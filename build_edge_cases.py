"""Deterministic stress banners. Preserve source-level train/validation/test splits."""
import io,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from banner_data import ROOT,records
KINDS=['tiny_text_lines','translucency','green_spill','heavy_jpeg','green_foreground','pale_floor','off_green','soft_shadow']
OUT=ROOT/'datasets/banner_edges'
def make(job):
    row,kind,idx=job;seed=17000000+row['source_product_id']*100+idx;rng=np.random.default_rng(seed)
    fg=Image.open(ROOT/row['rgba']).convert('RGBA');w,h=fg.size
    text=Image.open(ROOT/row['text_alpha']).convert('L');pa=Image.open(ROOT/row['product_alpha']).convert('L')
    yy,xx=np.mgrid[:h,:w];bg=np.zeros((h,w,3),np.float32);bg[...,1]=1;quality=90
    if kind=='tiny_text_lines':
        extra=Image.new('RGBA',fg.size);d=ImageDraw.Draw(extra);font=ImageFont.truetype(row['font'],int(rng.integers(8,13)))
        d.text((8,h-24),'SAVE 15% | FREE SHIPPING | LIMITED OFFER',font=font,fill=(245,245,245,255));d.line((8,h-6,w-8,h-6),fill=(255,210,35,255),width=1)
        fg=Image.alpha_composite(fg,extra);text=Image.fromarray(np.maximum(np.asarray(text),np.asarray(extra.getchannel('A'))))
    elif kind=='translucency':
        a=np.asarray(fg).copy();mask=np.asarray(pa)>0;a[...,3]=np.where(mask,np.round(a[...,3]*.45),a[...,3]);fg=Image.fromarray(a)
    elif kind=='green_spill':
        a=np.asarray(fg).copy();mask=(a[...,3]>0)&(a[...,3]<245);a[mask,:3]=np.uint8(.7*a[mask,:3]+.3*np.array([0,255,0]));fg=Image.fromarray(a)
    elif kind=='heavy_jpeg':quality=int(rng.integers(15,36))
    elif kind=='green_foreground':
        a=np.asarray(fg).copy();mask=np.asarray(pa)>10;a[mask,:3]=np.uint8(.18*a[mask,:3]+.82*np.array([0,255,0]));fg=Image.fromarray(a)
    elif kind=='pale_floor':
        start=float(rng.uniform(.25,.65));t=np.clip((yy/max(h-1,1)-start)/(1-start),0,1)[...,None]
        bg=bg*(1-t)+np.array([.92,.95,.9])*t
    elif kind=='off_green':
        bg[...,0]=rng.uniform(.015,.13);bg[...,2]=rng.uniform(.015,.13);bg[...,1]=rng.uniform(.55,.95)+.05*np.sin(xx/45)
    elif kind=='soft_shadow':
        shadow=Image.new('RGBA',fg.size);mask=pa.filter(ImageFilter.GaussianBlur(9));shadow.putalpha(mask.point(lambda a:int(a*.4)));fg=Image.alpha_composite(shadow,fg)
    a=np.asarray(fg,np.float32)/255;rgb=a[...,:3]*a[...,3:]+bg*(1-a[...,3:]);stem=f'e{row["source_product_id"]:04d}_{kind}';folder=OUT/row['split'];folder.mkdir(parents=True,exist_ok=True)
    paths={k:folder/f'{stem}{s}' for k,s in [('input','.jpg'),('alpha','_alpha.png'),('rgba','_rgba.png'),('text_alpha','_text.png'),('product_alpha','_product.png')]}
    Image.fromarray(np.uint8(np.clip(rgb*255,0,255))).save(paths['input'],quality=quality,subsampling=2);fg.save(paths['rgba']);fg.getchannel('A').save(paths['alpha']);text.save(paths['text_alpha']);pa.save(paths['product_alpha'])
    return {**{k:row[k] for k in ['split','source_product_id','source_sha256']},'id':stem,'edge_case':kind,'background_kind':kind,'seed':seed,'jpeg_quality':quality,'label_quality':'exact compositing alpha, inherited source cutout limitations',**{k:str(v.relative_to(ROOT)) for k,v in paths.items()}}
def main():
    jobs=[]
    for split in ['train','validation','test']:
        unique={}
        for row in records(split):unique.setdefault(row['source_product_id'],row)
        jobs.extend((row,kind,k) for row in unique.values() for k,kind in enumerate(KINDS))
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(make,jobs))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    splits={s:{r['source_sha256'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    assert not(splits['train']&splits['test'] or splits['train']&splits['validation'] or splits['test']&splits['validation'])
    print(json.dumps({'counts':{s:sum(r['split']==s for r in rows) for s in splits},'cases':KINDS,'source_split_overlap':False},indent=2))
if __name__=='__main__':main()
