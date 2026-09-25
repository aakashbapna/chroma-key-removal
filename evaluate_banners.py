"""Deterministic held-out banner smoke benchmark; does not train models."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from banner_data import records,load_sample,ROOT
from remove_v3 import LargeMatting
from remove_v2 import ProductMatting
from remove import rgba
OUT=ROOT/'reports/banners_v1/evaluation';OUT.mkdir(parents=True,exist_ok=True)
def checker(im):
    im=im.convert('RGBA');w,h=im.size;yy,xx=np.indices((h,w));g=np.where((xx//16+yy//16)%2,215,245).astype('uint8');bg=Image.fromarray(np.stack([g]*3,-1)).convert('RGBA');return Image.alpha_composite(bg,im).convert('RGB')
def panel(images,labels):
    out=Image.new('RGB',(400*len(images),260),'white');d=ImageDraw.Draw(out)
    for k,(im,label) in enumerate(zip(images,labels)):
        im=ImageOps.contain(im,(400,232));out.paste(im,(400*k+(400-im.width)//2,28+(232-im.height)//2));d.text((400*k+8,8),label,fill='black')
    return out
rows=records('test');ids=sorted({r['source_product_id'] for r in rows});selected=[]
for k,pid in enumerate(ids):
    kind=['solid_green','green_shade','floor_gradient'][k%3]
    selected.append(next(r for r in rows if r['source_product_id']==pid and r['background_kind']==kind))
models={'v3':LargeMatting(),'v2':ProductMatting()};results=[];panels=[]
for k,row in enumerate(selected):
    x,y=load_sample(row);gt=y[...,0];tm=y[...,2]>.02;pm=np.asarray(Image.open(ROOT/row['product_alpha']))>5;bg=gt<.01
    item={'id':row['id'],'background_kind':row['background_kind'],'input':row['input']};images=[Image.fromarray((x*255).astype('uint8')),checker(Image.open(ROOT/row['rgba']))]
    for name,model in models.items():
        t=time.perf_counter();a=model.alpha(x);elapsed=time.perf_counter()-t;err=np.abs(a-gt)
        item[name]={'alpha_mae':float(err.mean()),'text_alpha_mae':float(err[tm].mean()),'product_alpha_mae':float(err[pm].mean()),'background_leakage':float(a[bg].mean()),'seconds':elapsed}
        im=Image.fromarray(rgba(x,a));im.save(OUT/f'{row["id"]}_{name}.png');images.append(checker(im))
    p=panel(images,['JPEG input', 'Target transparency','V3 (4 MB)','V2']);p.save(OUT/f'{row["id"]}_comparison.jpg');panels.append(p)
    results.append(item);print(k+1,row['id'],row['background_kind'],item['v3'],flush=True)
summary={}
for kind in ['solid_green','green_shade','floor_gradient']:
    subset=[r for r in results if r['background_kind']==kind];summary[kind]={'count':len(subset)}
    for name in models:summary[kind][name]={key:float(np.mean([r[name][key] for r in subset])) for key in subset[0][name]}
(OUT/'metrics.json').write_text(json.dumps({'scope':'18 held-out products, 6 per background kind; existing models, no banner fine-tuning','summary':summary,'samples':results},indent=2))
contact=Image.new('RGB',(1600,780),'white')
for k,p in enumerate(panels[:3]):contact.paste(p,(0,k*260))
contact.save(OUT/'comparison.jpg')
print(json.dumps(summary,indent=2),flush=True)
