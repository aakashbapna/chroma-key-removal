"""Visual-only comparison of training/acceptance originals; not an accuracy test."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from banner_data import ROOT
from chroma import Matting,process,rgba
from compare_deeplab_banners import DeepLab,scale
OUT=ROOT/'reports/real_banners';model=Matting();dl=DeepLab('/Users/aakash/Downloads/deeplabv3.tflite');panels=[];report=[]
def checker(im):
    a=np.asarray(im.convert('RGBA'),np.float32)/255;h,w=a.shape[:2];yy,xx=np.indices((h,w));bg=np.where((xx//16+yy//16)%2,.72,.93)[...,None];return Image.fromarray(np.uint8(np.clip(a[...,:3]*a[...,3:]+bg*(1-a[...,3:]),0,1)*255))
for name in ['makeup','watch']:
    im=Image.open(ROOT/'datasets/real_banners'/f'{name}.jpg').convert('RGB');x=np.asarray(im,np.float32)/255;h,w=x.shape[:2];alpha=model.alpha(x);raw=Image.fromarray(rgba(x,alpha));auto,info=process(x);auto=Image.fromarray(auto)
    ratio=min(256/w,256/h);ww,hh=round(w*ratio),round(h*ratio);small=np.asarray(im.resize((ww,hh),Image.Resampling.LANCZOS),np.float32)/255;small=np.pad(small,((0,256-hh),(0,256-ww),(0,0)),mode='edge');pred,_=dl.predict(small,'minus_one_one');a=scale(pred['soft'][:hh,:ww],(w,h));deep=Image.fromarray(np.uint8(np.round(np.concatenate([x,a[...,None]],-1)*255)))
    raw.save(OUT/f'{name}_neural.png');deep.save(OUT/f'{name}_deeplab.png');images=[im,checker(raw),checker(auto),checker(deep)];ph=min(480,round(h/w*350));panel=Image.new('RGB',(1440,ph+40),'white');d=ImageDraw.Draw(panel)
    for k,(image,label) in enumerate(zip(images,['Input','Raw neural output','Automatic pipeline','DeepLab soft'])):
        image=ImageOps.contain(image,(350,ph));panel.paste(image,(k*360+(360-image.width)//2,30+(ph-image.height)//2));d.text((k*360+8,8),label,fill='black')
    panel.save(OUT/f'{name}_comparison.jpg');panels.append(panel);report.append({'image':name,'role':'training and acceptance only; no independent ground-truth alpha for watch','route':info['route'],'model_sha256':json.loads((ROOT/'models/banner_latest/selection.json').read_text())['sha256']});print(name,'complete',flush=True)
sheet=Image.new('RGB',(1440,sum(p.height for p in panels)),'white');top=0
for p in panels:sheet.paste(p,(0,top));top+=p.height
sheet.save(OUT/'comparison.jpg');(OUT/'comparison.json').write_text(json.dumps(report,indent=2))
