"""Larger dilated CNN with genuine product-alpha supervision and optional weak labels."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf
from product_data import dataset,weak_generated
from data import dataset as shapes

@tf.keras.utils.register_keras_serializable()
def alpha_loss(y,p):
    alpha=y[...,:1];confidence=y[...,1:2]
    # Spatial boundary emphasis covers hard JPEG-contaminated outlines too.
    hi=tf.nn.max_pool2d(alpha,5,1,'SAME');lo=-tf.nn.max_pool2d(-alpha,5,1,'SAME')
    weight=confidence*(1+4*(hi-lo)+2*tf.cast((alpha>.02)&(alpha<.98),tf.float32))
    return tf.reduce_sum(weight*(tf.square(alpha-p)+.2*tf.abs(alpha-p)))/tf.maximum(tf.reduce_sum(weight),1.)

@tf.keras.utils.register_keras_serializable()
def alpha_mae(y,p):return tf.reduce_mean(tf.abs(y[...,:1]-p))

def build():
    inp=tf.keras.Input((None,None,3),name='rgb');x=inp
    for channels,dilation in [(24,1),(32,1),(48,2),(48,4),(32,2),(24,1)]:
        x=tf.keras.layers.Conv2D(channels,3,padding='same',dilation_rate=dilation,activation='relu')(x)
    out=tf.keras.layers.Conv2D(1,1,activation='sigmoid',name='alpha')(x)
    return tf.keras.Model(inp,out,name='product_chroma_v2')

def export(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    @tf.function(input_signature=[tf.TensorSpec([1,160,160,3],tf.float32,name='rgb')])
    def infer(x):return model(x,training=False)
    frozen=convert_variables_to_constants_v2(infer.get_concrete_function())
    c=tf.lite.TFLiteConverter.from_concrete_functions([frozen]);c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    c.optimizations=[tf.lite.Optimize.DEFAULT];c.target_spec.supported_types=[tf.float16]
    Path('models/v2/product_chroma.tflite').write_bytes(c.convert())
    Path('models/v2/product_chroma.json').write_text(json.dumps({'tile':160,'halo':12,'stride':136,'input':'float32 RGB [0,1]','output':'float32 alpha [0,1]','receptive_field':23},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=24);ap.add_argument('--count',type=int,default=1884);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    tf.keras.utils.set_random_seed(2026);tf.config.threading.set_intra_op_parallelism_threads(6);tf.config.threading.set_inter_op_parallelism_threads(2)
    x,y=dataset('train',args.count,96,17);sx,sy=shapes(500000,256,96)
    x=np.concatenate([x,sx]);y=np.concatenate([y,sy]);y=np.concatenate([y,np.ones_like(y)],-1)
    weak=weak_generated();weak_count=0
    if weak is not None:
        wx,wy=weak;weak_count=len(wx);x=np.concatenate([x,wx]);y=np.concatenate([y,wy])
    vx,vy=dataset('validation',144,96,1000000);vy=np.concatenate([vy,np.ones_like(vy)],-1)
    model=tf.keras.models.load_model('models/v2/best.keras',compile=False) if args.resume else build()
    model.compile(optimizer=tf.keras.optimizers.Adam(.0005 if args.resume else .001),loss=alpha_loss,metrics=[alpha_mae])
    start=time.time()
    hist=model.fit(x,y,validation_data=(vx,vy),batch_size=16,epochs=args.epochs,verbose=2,callbacks=[tf.keras.callbacks.ModelCheckpoint('models/v2/best.keras',monitor='val_loss',save_best_only=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=3,factor=.5),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=7,restore_best_weights=True)])
    model=tf.keras.models.load_model('models/v2/best.keras',compile=False);export(model)
    report={'parameters':model.count_params(),'product_composites':args.count,'shape_composites':256,'weak_generated_patches':weak_count,'validation_composites':144,'epochs':len(hist.history['loss']),'seconds':time.time()-start,'seed':2026,'history':hist.history,'resume':args.resume}
    Path('reports/v2/training.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='history'})
if __name__=='__main__':main()
