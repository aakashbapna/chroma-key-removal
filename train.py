"""Train locally on CPU, select by validation loss, export builtins-only TFLite."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf
from data import dataset


def loss(y,p):
    # Extra weight on soft edges and thin strokes, without ignoring flat background.
    edge=tf.cast((y>.02)&(y<.98),tf.float32)
    return tf.reduce_mean((1+4*edge)*tf.square(y-p))+ .15*tf.reduce_mean(tf.abs(y-p))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--count',type=int,default=768)
    args=ap.parse_args()
    tf.keras.utils.set_random_seed(73)
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    x,y=dataset(0,args.count); vx,vy=dataset(100000,96)
    inp=tf.keras.Input((None,None,3),name='rgb')
    z=inp
    for filters,k in [(12,3),(16,3),(12,3)]:
        z=tf.keras.layers.Conv2D(filters,k,padding='same',activation='relu')(z)
    out=tf.keras.layers.Conv2D(1,1,activation='sigmoid',name='alpha')(z)
    model=tf.keras.Model(inp,out,name='chroma_matte')
    model.compile(optimizer=tf.keras.optimizers.Adam(.002),loss=loss,metrics=['mae'])
    Path('models').mkdir(exist_ok=True); Path('reports').mkdir(exist_ok=True)
    t=time.time()
    history=model.fit(x,y,validation_data=(vx,vy),batch_size=16,epochs=args.epochs,verbose=2,callbacks=[tf.keras.callbacks.ModelCheckpoint('models/best.keras',save_best_only=True,monitor='val_loss'),tf.keras.callbacks.ReduceLROnPlateau(patience=2,factor=.5)])
    model=tf.keras.models.load_model('models/best.keras',custom_objects={'loss':loss})
    @tf.function(input_signature=[tf.TensorSpec([1,128,128,3],tf.float32,name='rgb')])
    def infer(x): return model(x,training=False)
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen=convert_variables_to_constants_v2(infer.get_concrete_function())
    converter=tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types=[tf.float16]
    Path('models/chroma_matte.tflite').write_bytes(converter.convert())
    report={'parameters':model.count_params(),'train_samples':args.count,'validation_samples':96,'train_seeds':[0,args.count-1],'validation_seeds':[100000,100095],'epochs':args.epochs,'training_seconds':time.time()-t,'history':history.history,'tensorflow':tf.__version__,'seed':73}
    Path('reports/training.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2))

if __name__=='__main__': main()
