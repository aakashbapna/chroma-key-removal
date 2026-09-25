"""Native-resolution diagnostic comparison on 18 held-out product banners."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import json,time,argparse
from PIL import Image
import numpy as np
from banner_data import ROOT,records,load_sample
from compare_deeplab_banners import DeepLab,sample,scale,score,aggregate,OUT
from chroma import Matting as LargeMatting
ap=argparse.ArgumentParser();ap.add_argument('--deeplab',default='/Users/aakash/Downloads/deeplabv3.tflite');args=ap.parse_args()
r=json.loads((OUT/'metrics.json').read_text());norms=r['selected_normalization'];dl=DeepLab(args.deeplab,threads=2);model=LargeMatting(ROOT/'models/banner_latest/product_chroma.tflite');rows=records('test');ids=sorted({row['source_product_id'] for row in rows});chosen=[]
for k,pid in enumerate(ids):
 kind=['solid_green','green_shade','floor_gradient'][k%3];chosen.append(next(row for row in rows if row['source_product_id']==pid and row['background_kind']==kind))
selected=json.loads((ROOT/'models/banner_latest/selection.json').read_text())['selected_run'];out={name:[] for name in [selected,'deeplab_hard','deeplab_soft']}
for row in chosen:
 x,y=load_sample(row);h,w=x.shape[:2];start=time.perf_counter();p=model.alpha(x);sec=time.perf_counter()-start;out[selected].append(score(row,y,p,sec,'native'))
 for decoder,norm in norms.items():
  start=time.perf_counter();ratio=min(256/w,256/h);ww,hh=round(w*ratio),round(h*ratio);small=np.asarray(Image.fromarray(np.uint8(x*255)).resize((ww,hh),Image.Resampling.LANCZOS),np.float32)/255;small=np.pad(small,((0,256-hh),(0,256-ww),(0,0)),mode='edge');preds,_=dl.predict(small,norm);p=scale(preds[decoder][:hh,:ww],(w,h));sec=time.perf_counter()-start;out['deeplab_'+decoder].append(score(row,y,p,sec,'native'))
 print(row['id'],flush=True)
result={'scope':'18 unique held-out product assets, one banner per product, six per background. Diagnostic subset only; not additional independent test sources. Native labels retained. Banner uses native-resolution halo-protected tiles with two CPU threads; DeepLab uses one global 257px prediction with two CPU threads and mask upsampling. Different work makes these deployment-path timings, not equal-work model benchmarks. Both timings include their array preprocessing but exclude image-file loading.','methods':{k:{'overall':aggregate(v),'groups':{g:aggregate([i for i in v if i['group']==g]) for g in ['solid_green','green_shade','floor_gradient']},'samples':v} for k,v in out.items()}}
(OUT/'native_metrics.json').write_text(json.dumps(result,indent=2))
print({k:v['overall'] for k,v in result['methods'].items()},flush=True)
