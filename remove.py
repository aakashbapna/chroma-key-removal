"""Native-resolution tiled alpha inference and green-background uncompositing."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import tensorflow as tf
DEFAULT_MODEL=Path(__file__).resolve().parent/'models/chroma_matte.tflite'

class Matting:
    def __init__(self,path=DEFAULT_MODEL):
        self.m=tf.lite.Interpreter(model_path=str(path),num_threads=2)
        self.m.allocate_tensors()
        self.i=self.m.get_input_details()[0]['index']; self.o=self.m.get_output_details()[0]['index']
    def alpha(self,rgb):
        h,w=rgb.shape[:2]; output=np.empty((h,w),np.float32)
        # Discard 4 pixels at each tile edge; receptive radius is only 3.
        padded=np.pad(rgb,((4,132),(4,132),(0,0)),mode='edge')
        for y in range(0,h,120):
            for x in range(0,w,120):
                tile=padded[y:y+128,x:x+128]
                self.m.set_tensor(self.i,tile[None].astype(np.float32)); self.m.invoke()
                a=self.m.get_tensor(self.o)[0,4:124,4:124,0]
                hh,ww=min(120,h-y),min(120,w-x)
                output[y:y+hh,x:x+ww]=a[:hh,:ww]
        return output

def rgba(rgb,alpha):
    # Snap tiny sigmoid tails, then invert C=aF+(1-a)green to remove green fringes.
    a=np.where(alpha<.01,0,np.where(alpha>.99,1,alpha))
    fg=np.clip((rgb-(1-a[...,None])*np.array([0,1,0]))/np.maximum(a[...,None],.03),0,1)
    fg[a==0]=0
    return np.uint8(np.round(np.concatenate([fg,a[...,None]],-1)*255))

def main():
    p=argparse.ArgumentParser(); p.add_argument('input');p.add_argument('output');p.add_argument('--model',default=str(DEFAULT_MODEL)); args=p.parse_args()
    rgb=np.asarray(ImageOps.exif_transpose(Image.open(args.input)).convert('RGB'),np.float32)/255
    result=rgba(rgb,Matting(args.model).alpha(rgb));Image.fromarray(result,'RGBA').save(args.output,format='PNG')
    print(f'Saved transparent PNG: {args.output}')
if __name__=='__main__': main()
