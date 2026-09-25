"""Remove chroma with the validation-selected banner checkpoint at native resolution."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from remove_v3 import LargeMatting
from remove import rgba
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('output');args=ap.parse_args()
    x=np.asarray(ImageOps.exif_transpose(Image.open(args.input)).convert('RGB'),np.float32)/255
    Image.fromarray(rgba(x,LargeMatting(ROOT/'models/banner_latest/product_chroma.tflite').alpha(x))).save(args.output,format='PNG')
    print('Saved',args.output)
