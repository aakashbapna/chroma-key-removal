"""Model-only alpha inference. No color detection, routing, or edge correction."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
from pathlib import Path
import numpy as np
import tensorflow as tf
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
    """Encode input colors with predicted opacity; only PNG range conversion."""
    return np.uint8(np.round(np.clip(np.concatenate([rgb,alpha[...,None]],-1),0,1)*255))

def process(rgb,model=None,model_path=DEFAULT_MODEL):
    model=model or Matting(model_path)
    return rgba(rgb,model.alpha(rgb)),{'route':'model_only_alpha','rgb':'unchanged input colors'}
