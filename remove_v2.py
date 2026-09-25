"""Remove green at original resolution with the larger product model."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import tensorflow as tf
from remove import rgba
DEFAULT=Path(__file__).resolve().parent/'models/v2/product_chroma.tflite'
class ProductMatting:
    def __init__(self,path=DEFAULT):
        self.m=tf.lite.Interpreter(model_path=str(path),num_threads=2);self.m.allocate_tensors()
        details=self.m.get_input_details()[0];self.i=details['index'];self.o=self.m.get_output_details()[0]['index'];self.tile=int(details['shape'][1]);self.halo=12;self.stride=self.tile-24
    def alpha(self,rgb):
        h,w=rgb.shape[:2];out=np.empty((h,w),np.float32);pad=self.halo;s=self.stride;t=self.tile
        padded=np.pad(rgb,((pad,t+pad),(pad,t+pad),(0,0)),mode='edge')
        for y in range(0,h,s):
            for x in range(0,w,s):
                self.m.set_tensor(self.i,padded[y:y+t,x:x+t][None].astype(np.float32));self.m.invoke()
                a=self.m.get_tensor(self.o)[0,pad:pad+s,pad:pad+s,0];hh,ww=min(s,h-y),min(s,w-x);out[y:y+hh,x:x+ww]=a[:hh,:ww]
        return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('output');args=ap.parse_args()
    x=np.asarray(ImageOps.exif_transpose(Image.open(args.input)).convert('RGB'),np.float32)/255
    Image.fromarray(rgba(x,ProductMatting().alpha(x))).save(args.output,format='PNG');print('Saved',args.output)
