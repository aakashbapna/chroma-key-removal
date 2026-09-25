"""V2 + trainable multi-scale U-Net alpha refinement, exported below 10 MB."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf
from product_data import dataset
from data import dataset as shapes
from train_v2 import alpha_loss,alpha_mae
ROOT=Path(__file__).resolve().parent

def build():
    base=tf.keras.models.load_model(ROOT/'models/v2/best.keras',compile=False);base.trainable=False
    inp=tf.keras.Input((None,None,3),name='rgb');prior=base(inp)
    x=tf.keras.layers.Concatenate()([inp,prior]);skips=[]
    def block(x,filters):
        for _ in range(2):x=tf.keras.layers.Conv2D(filters,3,padding='same',activation='relu',kernel_initializer='he_normal')(x)
        return x
    for filters in [32,64,128]:
        x=block(x,filters);skips.append(x);x=tf.keras.layers.AveragePooling2D(2)(x)
    x=block(x,256)
    for filters,skip in zip([128,64,32],reversed(skips)):
        x=tf.keras.layers.UpSampling2D(2,interpolation='nearest')(x)
        x=tf.keras.layers.Concatenate()([x,skip]);x=block(x,filters)
    delta=tf.keras.layers.Conv2D(1,1,activation='tanh',kernel_initializer='zeros',bias_initializer='zeros',name='alpha_correction')(x)
    alpha=tf.keras.layers.ReLU(max_value=1.,name='alpha')(tf.keras.layers.Add()([prior,delta]))
    return tf.keras.Model(inp,alpha,name='product_chroma_v3')

def export(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    @tf.function(input_signature=[tf.TensorSpec([1,256,256,3],tf.float32,name='rgb')])
    def infer(x):return model(x,training=False)
    frozen=convert_variables_to_constants_v2(infer.get_concrete_function())
    c=tf.lite.TFLiteConverter.from_concrete_functions([frozen]);c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS];c.optimizations=[tf.lite.Optimize.DEFAULT];c.target_spec.supported_types=[tf.float16]
    data=c.convert();assert len(data)<10_000_000,len(data)
    Path('models/v3/product_chroma.tflite').write_bytes(data)
    Path('models/v3/product_chroma.json').write_text(json.dumps({'tile':256,'halo':64,'stride':128,'alignment':8,'input':'float32 RGB [0,1]','output':'float32 alpha [0,1]','parameters':model.count_params(),'bytes':len(data)},indent=2))

def weak_images(size):
    from PIL import Image
    from scipy.ndimage import minimum_filter
    path=ROOT/'datasets/generated_v3/prepared_manifest.json'
    if not path.exists():return None,[]
    rows=[r for r in json.loads(path.read_text()) if r['split']=='train_weak' and r.get('background_qa_pass',False)];pairs=[]
    for row in rows:
        im=Image.open(ROOT/row['path']).convert('RGB');im.thumbnail((384,384));x=np.asarray(im,np.float32)/255
        strength=x[...,1]-np.maximum(x[...,0],x[...,2]);bg=(strength>.78)&(x[...,1]>.9);fg=strength<.08
        conf=minimum_filter((bg|fg).astype(np.float32),size=7)*.1
        for i in range(4):
            rng=np.random.default_rng(710000+row['id']*10+i);h,w=x.shape[:2];y=int(rng.integers(0,h-size+1));xx=int(rng.integers(0,w-size+1))
            pairs.append((x[y:y+size,xx:xx+size],np.stack([fg[y:y+size,xx:xx+size],conf[y:y+size,xx:xx+size]],-1)))
    return (tuple(map(np.stack,zip(*pairs))) if pairs else None),[r['id'] for r in rows]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=16);ap.add_argument('--count',type=int,default=1570);ap.add_argument('--size',type=int,default=96);ap.add_argument('--resume',action='store_true');ap.add_argument('--include-generated',action='store_true');args=ap.parse_args()
    tf.keras.utils.set_random_seed(2027);tf.config.threading.set_intra_op_parallelism_threads(6);tf.config.threading.set_inter_op_parallelism_threads(2)
    x,y=dataset('train',args.count,args.size,3000000);sx,sy=shapes(700000,256,args.size);x=np.concatenate([x,sx]);y=np.concatenate([y,sy]);y=np.concatenate([y,np.ones_like(y)],-1)
    weak_ids=[];weak_count=0
    if args.include_generated:
        weak,weak_ids=weak_images(args.size)
        if weak is not None:
            wx,wy=weak;weak_count=len(wx);x=np.concatenate([x,wx]);y=np.concatenate([y,wy])
    vx,vy=dataset('validation',144,args.size,4000000);vy=np.concatenate([vy,np.ones_like(vy)],-1)
    model=tf.keras.models.load_model('models/v3/best.keras',compile=False) if args.resume else build()
    # Restoring serialization must not unfreeze the previously trained V2 branch.
    for layer in model.layers:
        if isinstance(layer,tf.keras.Model):layer.trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(.0001 if args.resume else .0002,clipnorm=1.),loss=alpha_loss,metrics=[alpha_mae])
    initial=model.evaluate(vx,vy,verbose=0,return_dict=True)
    stage='finetune' if args.resume else 'training'
    # Preserve the starting checkpoint when further training fails to improve validation.
    model.save('models/v3/best.keras')
    checkpoint=tf.keras.callbacks.ModelCheckpoint('models/v3/best.keras',save_best_only=True,monitor='val_loss',initial_value_threshold=initial['loss'])
    start=time.time();history=model.fit(x,y,validation_data=(vx,vy),batch_size=12,epochs=args.epochs,verbose=2,callbacks=[checkpoint,tf.keras.callbacks.ReduceLROnPlateau(patience=2,factor=.5,min_lr=.0000125),tf.keras.callbacks.EarlyStopping(patience=5,monitor='val_loss')])
    model=tf.keras.models.load_model('models/v3/best.keras',compile=False);export(model)
    report={'parameters':model.count_params(),'trainable_parameters':int(sum(np.prod(w.shape) for w in model.trainable_weights)),'epochs':len(history.history['loss']),'product_composites':args.count,'shape_composites':256,'generated_weak_patches':weak_count,'generated_train_ids':weak_ids,'image_size':args.size,'validation_composites':144,'initial_validation':initial,'seconds':time.time()-start,'history':history.history,'resume':args.resume}
    Path(f'reports/v3/{stage}.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='history'})
if __name__=='__main__':main()
