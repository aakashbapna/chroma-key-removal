"""Verify the current compact model against its Keras checkpoint and odd input sizes."""
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from chroma import ROOT,Matting
m=Matting();keras=tf.keras.models.load_model(ROOT/'models/banner_latest/resume.keras',compile=False);x=np.asarray(Image.open(ROOT/'examples/banner_input.jpg').convert('RGB').resize((256,256)),np.float32)[None]/255
m.m.set_tensor(m.i,x);m.m.invoke();difference=float(np.max(np.abs(keras(x,training=False).numpy()-m.m.get_tensor(m.o))));assert difference<.02
checks=[]
for h,w in [(1,1),(7,9),(127,129),(261,133)]:
    x=np.zeros((h,w,3),np.float32);x[...,1]=1;a=m.alpha(x);assert a.shape==(h,w) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1;checks.append({'size':[w,h],'pure_green_max_alpha':float(a.max())})
result={'selected_model':json.loads((ROOT/'models/banner_latest/selection.json').read_text()),'keras_tflite_max_absolute_difference':difference,'size_checks':checks}
(ROOT/'reports/model_verification.json').write_text(json.dumps(result,indent=2));print(result)
