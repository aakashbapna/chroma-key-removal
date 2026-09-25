"""Canonical TFLite inference and conservative chroma-aware edge cleanup."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt,minimum_filter,maximum_filter
ROOT=Path(__file__).resolve().parent
DEFAULT_MODEL=ROOT/'models/banner_latest/product_chroma.tflite'
class Matting:
    def __init__(self,path=DEFAULT_MODEL):
        self.m=tf.lite.Interpreter(model_path=str(path),num_threads=2);self.m.allocate_tensors();self.i=self.m.get_input_details()[0]['index'];self.o=self.m.get_output_details()[0]['index']
    def alpha(self,rgb):
        h,w=rgb.shape[:2];out=np.empty((h,w),np.float32);padded=np.pad(rgb,((64,320),(64,320),(0,0)),mode='edge')
        for y in range(0,h,128):
            for x in range(0,w,128):
                self.m.set_tensor(self.i,padded[y:y+256,x:x+256][None].astype(np.float32));self.m.invoke();a=self.m.get_tensor(self.o)[0,64:192,64:192,0];hh,ww=min(128,h-y),min(128,w-x);out[y:y+hh,x:x+ww]=a[:hh,:ww]
        return out

def rgba(rgb,alpha):
    """Historical pure-green reconstruction, retained for benchmark comparability."""
    a=np.where(alpha<.01,0,np.where(alpha>.99,1,alpha));fg=np.clip((rgb-(1-a[...,None])*np.array([0,1,0]))/np.maximum(a[...,None],.03),0,1);fg[a==0]=0
    return np.uint8(np.round(np.concatenate([fg,a[...,None]],-1)*255))

def background_info(rgb):
    border=np.concatenate([rgb[:4].reshape(-1,3),rgb[-4:].reshape(-1,3),rgb[:,:4].reshape(-1,3),rgb[:,-4:].reshape(-1,3)])
    green=(border[:,1]-np.maximum(border[:,0],border[:,2])>.3)&(border[:,1]>.45);fraction=float(green.mean())
    key=np.median(border[green],axis=0) if green.any() else np.array([0,1,0],np.float32)
    spread=float(np.quantile(np.max(np.abs(border[green]-key),-1),.95)) if green.any() else 1.
    return {'green_border_fraction':fraction,'key':key.tolist(),'uniform_green':fraction>.85 and spread<.16,'has_green_background':fraction>.15,'spread':spread}

def refine_uniform(rgb):
    """Local foreground/background color projection; explicitly not a neural prediction."""
    ex=rgb[...,1]-np.maximum(rgb[...,0],rgb[...,2]);fg=ex<.025;bg=(ex>.35)&(rgb[...,1]>.5)
    if not fg.any():return np.zeros(rgb.shape[:2],np.float32),np.zeros_like(rgb)
    interior=minimum_filter(fg,size=5)>0
    if not interior.any():interior=fg
    _,fi=distance_transform_edt(~interior,return_indices=True);f=rgb[tuple(fi)]
    bg_inner=minimum_filter(bg,size=3)>0
    if not bg_inner.any():bg_inner=bg
    if not bg_inner.any():return np.ones(rgb.shape[:2],np.float32),rgb.copy()
    _,bi=distance_transform_edt(~bg_inner,return_indices=True);key=rgb[tuple(bi)];v=f-key
    a=np.clip(np.sum((rgb-key)*v,-1)/np.maximum(np.sum(v*v,-1),1e-6),0,1);a[fg]=1;d=distance_transform_edt(~fg);a[bg&(d>8)]=0;a[a<.01]=0;a[a>.99]=1
    # Stable inner foreground colors avoid both green and magenta inversion fringes.
    color=rgb.copy();band=(a>0)&(a<1);color[band]=f[band]
    edge=maximum_filter(a<.99,size=7)&(a>0);excess=np.maximum(color[...,1]-np.maximum(color[...,0],color[...,2]),0);color[...,1]-=edge*excess
    color[a==0]=0;return a.astype(np.float32),np.clip(color,0,1)

def process(rgb,model=None,mode='auto',model_path=DEFAULT_MODEL):
    info=background_info(rgb)
    if mode=='auto' and not info['has_green_background']:
        return np.uint8(np.round(np.concatenate([rgb,np.ones((*rgb.shape[:2],1))],-1)*255)),{**info,'route':'preserve_non_chroma'}
    if mode=='auto' and info['uniform_green']:
        a,color=refine_uniform(rgb);return np.uint8(np.round(np.concatenate([color,a[...,None]],-1)*255)),{**info,'route':'uniform_chroma_local_color_projection'}
    model=model or Matting(model_path);a=model.alpha(rgb)
    return rgba(rgb,a),{**info,'route':'neural_alpha'}
