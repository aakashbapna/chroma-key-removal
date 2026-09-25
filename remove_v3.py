"""Native-resolution U-Net inference with aligned, halo-protected tiles."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import tensorflow as tf
from remove import rgba
DEFAULT=Path(__file__).resolve().parent/'models/v3/product_chroma.tflite'
class LargeMatting:
    def __init__(self,path=DEFAULT):
        self.m=tf.lite.Interpreter(model_path=str(path),num_threads=2);self.m.allocate_tensors();self.i=self.m.get_input_details()[0]['index'];self.o=self.m.get_output_details()[0]['index']
    def alpha(self,rgb):
        h,w=rgb.shape[:2];out=np.empty((h,w),np.float32)
        # Starts and halo are multiples of 8 so three encoder pooling stages align.
        padded=np.pad(rgb,((64,320),(64,320),(0,0)),mode='edge')
        for y in range(0,h,128):
            for x in range(0,w,128):
                self.m.set_tensor(self.i,padded[y:y+256,x:x+256][None].astype(np.float32));self.m.invoke()
                a=self.m.get_tensor(self.o)[0,64:192,64:192,0];hh,ww=min(128,h-y),min(128,w-x);out[y:y+hh,x:x+ww]=a[:hh,:ww]
        return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('output');args=ap.parse_args()
    x=np.asarray(ImageOps.exif_transpose(Image.open(args.input)).convert('RGB'),np.float32)/255
    Image.fromarray(rgba(x,LargeMatting().alpha(x))).save(args.output,format='PNG');print('Saved',args.output)
