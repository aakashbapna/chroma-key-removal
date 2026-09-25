"""Real product alpha composites; held-out source products never enter training."""
import io,json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
from scipy.ndimage import maximum_filter,minimum_filter
ROOT=Path(__file__).resolve().parent

@lru_cache(maxsize=256)
def asset(path):
    return Image.open(ROOT/path).convert('RGBA')

def manifest(split):
    return [r for r in json.loads((ROOT/'datasets/products/manifest.json').read_text()) if r['split']==split]

def composite(row,seed,size=128):
    rng=np.random.default_rng(seed);s=size*2
    im=asset(row['path']).copy()
    im.thumbnail((int(rng.uniform(.48,.96)*s),int(rng.uniform(.48,.96)*s)),Image.Resampling.LANCZOS)
    if rng.random()<.5:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random()<.5:im=im.rotate(float(rng.uniform(-22,22)),resample=Image.Resampling.BICUBIC,expand=True)
    # Modest brightness augmentation retains real textures and natural green products.
    arr=np.asarray(im).copy();arr[...,:3]=np.clip(arr[...,:3]*rng.uniform(.8,1.12),0,255);im=Image.fromarray(arr)
    layer=Image.new('RGBA',(s,s))
    x=int(rng.integers(min(0,s-im.width),max(1,s-im.width+1)));y=int(rng.integers(min(0,s-im.height),max(1,s-im.height+1)))
    layer.alpha_composite(im,(x,y))
    if rng.random()<.35:
        d=ImageDraw.Draw(layer);font=ImageFont.load_default(size=int(rng.integers(13,35)))
        color=tuple(map(int,rng.integers(20,245,3)))+(255,)
        d.text((int(rng.integers(0,s//3)),int(rng.integers(0,s//5))),str(rng.choice(['SALE','NEW','50% OFF','SHOP'])),font=font,fill=color)
    layer=layer.resize((size,size),Image.Resampling.LANCZOS)
    rgba=np.asarray(layer,np.float32)/255;alpha=rgba[...,3:];fg=rgba[...,:3]
    if rng.random()<.15:alpha=alpha*rng.uniform(.6,.95)
    rgb=fg*alpha+np.array([0,1,0])*(1-alpha)
    quality=int(rng.integers(55,101));buf=io.BytesIO()
    Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(buf,format='JPEG',quality=quality,subsampling=int(rng.choice([0,2])))
    x=np.asarray(Image.open(buf),np.float32)/255
    return x,alpha.astype(np.float32),fg

def dataset(split,count,size=96,seed=17):
    rows=manifest(split);pairs=[composite(rows[i%len(rows)],seed+i,size)[:2] for i in range(count)]
    x,y=map(np.stack,zip(*pairs))
    return x,y

def weak_generated(size=96):
    path=ROOT/'datasets/generated/manifest.json'
    if not path.exists():return None
    rows=[r for r in json.loads(path.read_text()) if r['split']=='train_weak']
    pairs=[]
    for row in rows:
        im=Image.open(ROOT/row['path']).convert('RGB');im.thumbnail((512,512))
        x=np.asarray(im,np.float32)/255
        # Only supervise high-confidence background and non-green foreground.
        # Ambiguous green edges/foreground are excluded, never treated as exact alpha.
        strength=x[...,1]-np.maximum(x[...,0],x[...,2])
        bg=(strength>.72)&(x[...,1]>.85)
        fg=(strength<.12)
        confidence=minimum_filter((bg|fg).astype(np.float32),size=5)*.15
        label=fg.astype(np.float32)
        for i in range(8):
            rng=np.random.default_rng(row['id']*100+i)
            h,w=x.shape[:2];yy=int(rng.integers(0,h-size+1));xx=int(rng.integers(0,w-size+1))
            pairs.append((x[yy:yy+size,xx:xx+size],np.stack([label[yy:yy+size,xx:xx+size],confidence[yy:yy+size,xx:xx+size]],-1)))
    if not pairs:return None
    return tuple(map(np.stack,zip(*pairs)))
