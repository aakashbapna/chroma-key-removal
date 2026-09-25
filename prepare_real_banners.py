"""Local user examples: exact opaque makeup target and weak watch pseudo-labels.

These are training/acceptance examples, never independently scored test data.
"""
import io,json,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
from banner_data import ROOT
from chroma import refine_uniform
OUT=ROOT/'datasets/real_banners';rows=[]
for name,count in [('makeup',64),('watch',128)]:
    path=OUT/f'{name}.jpg';rgb=np.asarray(Image.open(path).convert('RGB'),np.float32)/255;h,w=rgb.shape[:2]
    if name=='makeup':alpha=np.ones((h,w),np.float32);confidence=np.ones_like(alpha)
    else:
        alpha,_=refine_uniform(rgb);sure=(alpha<.01)|(alpha>.99);confidence=minimum_filter(sure.astype(np.float32),size=9)*.15
    for k in range(count):
        rng=np.random.default_rng(62000+k+(1000 if name=='watch' else 0));size=int(rng.choice([128,192,256]));size=min(size,h,w)
        if name=='makeup' and k<32:cx,cy=int(w*.94),int(h*.14)
        elif name=='watch' and k<64:cx,cy=int(w*rng.uniform(.23,.76)),int(h*rng.uniform(.05,.47))
        else:cx,cy=int(rng.integers(w)),int(rng.integers(h))
        x0=int(np.clip(cx-size//2,0,w-size));y0=int(np.clip(cy-size//2,0,h-size));box=np.s_[y0:y0+size,x0:x0+size]
        patch=np.uint8(np.clip(rgb[box]*rng.uniform(.96,1.03)*255,0,255));im=Image.fromarray(patch).resize((128,128),Image.Resampling.LANCZOS);stem=f'{name}_{k:03d}';folder=OUT/'train';folder.mkdir(exist_ok=True)
        im.save(folder/f'{stem}.jpg',quality=int(rng.integers(60,99)),subsampling=2)
        for key,a in [('weak_alpha',alpha[box]),('confidence',confidence[box]),('text_alpha',np.zeros((size,size),np.float32))]:Image.fromarray(np.uint8(np.round(a*255))).resize((128,128),Image.Resampling.BILINEAR).save(folder/f'{stem}_{key}.png')
        rows.append({'id':stem,'split':'train','source':'user_supplied_'+name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'label_quality':'exact opaque preservation requested by user' if name=='makeup' else 'weak local-color pseudo-alpha; no independent ground truth','input':str((folder/f'{stem}.jpg').relative_to(ROOT)),**{key:str((folder/f'{stem}_{key}.png').relative_to(ROOT)) for key in ['weak_alpha','confidence','text_alpha']}})
(OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print('Prepared',len(rows),'training crops; originals excluded from held-out scores')
