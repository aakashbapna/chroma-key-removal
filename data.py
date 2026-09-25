"""Procedural, licensed asset-free chroma composites with exact alpha labels."""
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def sample(seed, size=96, stress=False):
    rng = np.random.default_rng(seed)
    scale = 3
    s = size * scale
    mask = Image.new('L', (s, s))
    d = ImageDraw.Draw(mask)
    for _ in range(int(rng.integers(2, 9))):
        x,y = rng.integers(-s//8, s*7//8, 2)
        w,h = rng.integers(s//12, s//2, 2)
        box = (int(x),int(y),int(x+w),int(y+h))
        opacity = int(rng.choice([255,255,255,180,110]))
        kind = rng.integers(4)
        if kind == 0: d.ellipse(box, fill=opacity)
        elif kind == 1: d.rounded_rectangle(box, radius=int(w//5), fill=opacity)
        elif kind == 2:
            points = [tuple(map(int,p)) for p in rng.integers(0,s,(4,2))]
            d.line(points, fill=opacity, width=int(rng.integers(2,12)))
        else:
            # Pillow's packaged default font avoids machine-specific font dependencies.
            font = ImageFont.load_default(size=int(rng.integers(s//12,s//4)))
            d.text((int(x),int(y)), str(rng.choice(['SALE','AI','2026','Banner','50%'])), font=font, fill=opacity)
    if rng.random()<.3: mask=mask.filter(ImageFilter.GaussianBlur(float(rng.uniform(.3,2)*scale)))
    a=np.asarray(mask.resize((size,size), Image.Resampling.LANCZOS),dtype=np.float32)/255
    c1=rng.uniform(.03,.98,3); c2=rng.uniform(.03,.98,3)
    # Saturated key green cannot be distinguished from genuine identical foreground.
    # Ordinary training avoids that impossible case; stress tests deliberately include it.
    for c in (c1,c2):
        if not stress and c[1]>max(c[0],c[2])+.15: c[1]=max(c[0],c[2])
    if stress: c1=np.array([.02,.95,.03]); c2=np.array([.1,.7,.15])
    yy,xx=np.mgrid[0:size,0:size]
    mix=((xx+yy)/(2*size))[...,None]
    foreground=c1*(1-mix)+c2*mix
    foreground=np.clip(foreground+rng.normal(0,.025,(size,size,1)),0,1)
    bg=np.array([0,1,0],dtype=np.float32)
    rgb=foreground*a[...,None]+bg*(1-a[...,None])
    quality=int(rng.integers(55,101))
    buf=io.BytesIO()
    Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(buf,format='JPEG',quality=quality,subsampling=int(rng.choice([0,2])))
    rgb=np.asarray(Image.open(buf).convert('RGB'),dtype=np.float32)/255
    return rgb,a[...,None],foreground.astype(np.float32)


def dataset(start,count,size=96):
    pairs=[sample(start+i,size)[:2] for i in range(count)]
    return tuple(np.stack(x) for x in zip(*pairs))
