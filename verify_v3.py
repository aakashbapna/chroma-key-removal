import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
from pathlib import Path
import tensorflow as tf
import numpy as np
from product_data import composite,manifest
from remove_v3 import LargeMatting
from remove import rgba
model=tf.keras.models.load_model('models/v3/best.keras',compile=False);runner=LargeMatting()
x,_,_=composite(manifest('validation')[0],987654,289)
h,w=x.shape[:2];ph=(-h-128)%8;pw=(-w-128)%8
p=np.pad(x,((64,64+ph),(64,64+pw),(0,0)),mode='edge')
a=model(p[None],training=False).numpy()[0,64:64+h,64:64+w,0];b=runner.alpha(x)
error=float(np.max(np.abs(a-b)));assert error<.005,error
for h,w in [(1,1),(7,301),(271,139)]:
    x=np.zeros((h,w,3),np.float32);x[...,1]=1
    a=runner.alpha(x);assert a.shape==(h,w);assert np.isfinite(a).all();assert rgba(x,a)[...,3].max()==0,float(a.max())
size=Path('models/v3/product_chroma.tflite').stat().st_size;assert size<10_000_000
report={'keras_tflite_max_alpha_difference':error,'dimensions_and_aligned_tiling':'passed','pure_green_transparency':'passed','parameters':model.count_params(),'bytes':size,'under_10_mb':True,'operators':sorted(set(op['op_name'] for op in runner.m._get_ops_details()))}
Path('reports/v3/verification.json').write_text(json.dumps(report,indent=2));print(report)
