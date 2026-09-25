"""Remove a green backdrop while preserving non-chroma banners intact."""
import argparse,json
import numpy as np
from PIL import Image,ImageOps
from chroma import Matting,process,DEFAULT_MODEL
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('output');ap.add_argument('--mode',choices=['auto','neural'],default='auto');ap.add_argument('--model',default=str(DEFAULT_MODEL));a=ap.parse_args()
    rgb=np.asarray(ImageOps.exif_transpose(Image.open(a.input)).convert('RGB'),np.float32)/255
    result,info=process(rgb,Matting(a.model) if a.mode=='neural' else None,a.mode,model_path=a.model)
    Image.fromarray(result).save(a.output,format='PNG');print(json.dumps(info))
