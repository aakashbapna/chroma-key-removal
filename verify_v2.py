import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
from product_data import composite,manifest
from remove_v2 import ProductMatting
from remove import rgba
model=tf.keras.models.load_model('models/v2/best.keras',compile=False);runner=ProductMatting()
x,_,_=composite(manifest('validation')[0],987654,289)
p=np.pad(x,((12,12),(12,12),(0,0)),mode='edge')
a=model(p[None],training=False).numpy()[0,12:-12,12:-12,0];b=runner.alpha(x)
error=float(np.max(np.abs(a-b)));assert error<.005,error
for h,w in [(1,1),(7,301),(271,139)]:
    x=np.zeros((h,w,3),np.float32);x[...,1]=1
    a=runner.alpha(x);assert a.shape==(h,w);assert np.isfinite(a).all()
    assert rgba(x,a)[...,3].max()==0,float(a.max())
report={'keras_tflite_max_alpha_difference':error,'dimensions_and_tiling':'passed','pure_green_transparency':'passed','parameters':model.count_params(),'operators':sorted(set(op['op_name'] for op in runner.m._get_ops_details()))}
Path('reports/v2/verification.json').write_text(json.dumps(report,indent=2));print(report)
