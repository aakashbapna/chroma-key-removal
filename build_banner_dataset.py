"""One product + offer text banners with exact compositing labels and separate weak data."""
import io,json,hashlib,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import minimum_filter
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'datasets/banners_v1'
FONTS=[Path('/System/Library/Fonts/Supplemental')/n for n in ['Arial Bold.ttf','Arial Black.ttf','Arial Narrow Bold.ttf','Arial Bold Italic.ttf','Verdana Bold.ttf','Arial.ttf']]
OFFERS=['30% OFF','50% OFF','SAVE 25%','FLAT 40% OFF','UP TO 60% OFF','BUY 1 GET 1','SPECIAL PRICE','SAVE $20','20% OFF','LIMITED DEAL']
SUBLINES=['LIMITED TIME OFFER','WEEKEND SPECIAL','SHOP THE COLLECTION','ONLINE EXCLUSIVE','ENDS TONIGHT','NEW SEASON SAVINGS']
PALETTES=[((255,255,255),(20,20,30),(244,39,74)),((15,18,25),(255,255,255),(255,217,25)),((255,230,40),(255,255,255),(30,30,70)),((239,33,70),(255,255,255),(20,25,35)),((30,38,120),(255,255,255),(255,195,20))]

def fit_font(path,text,width,size):
    for px in range(max(10,int(size)),9,-1):
        font=ImageFont.truetype(str(path),px);b=font.getbbox(text)
        if b[2]-b[0]<=width:return font
    return ImageFont.truetype(str(path),10)

def design(size,seed,headline=None):
    rng=np.random.default_rng(seed);w,h=size;ss=2;W,H=w*ss,h*ss
    fontpath=FONTS[int(rng.integers(len(FONTS)))];offer=headline or str(rng.choice(OFFERS));sub=str(rng.choice(SUBLINES));cta=str(rng.choice(['SHOP NOW','GRAB THE DEAL','BUY NOW']))
    ink,subink,accent=PALETTES[int(rng.integers(len(PALETTES)))];margin=int(min(W,H)*.055)
    text=Image.new('RGBA',(W,H));dec=Image.new('RGBA',(W,H));d=ImageDraw.Draw(text);dd=ImageDraw.Draw(dec)
    wide=w/h>1.3
    if wide:
        right=bool(rng.integers(2));textleft=margin if right else int(W*.54)
        box=(int(W*.52) if right else margin,margin,int(W*.96) if right else int(W*.48),H-margin)
        area=(textleft,int(H*.18),textleft+int(W*.42),int(H*.88));layout='product_right' if right else 'product_left'
    else:
        box=(int(W*.12),int(H*.39),int(W*.88),int(H*.88));area=(margin,int(H*.06),W-margin,int(H*.34));layout='product_below_offer'
    x,y,x2,y2=area;maxwidth=x2-x;main_size=H*.15 if wide else H*.095
    main=fit_font(fontpath,offer,maxwidth,main_size);b=main.getbbox(offer);mainh=b[3]-b[1]
    # Bounding-box offsets prevent clipping ascenders or italic overhangs.
    d.text((x-b[0],y-b[1]),offer,font=main,fill=ink+(255,),stroke_width=0)
    y+=mainh+int(H*.055 if wide else H*.025)
    secondary=fit_font(fontpath,sub,maxwidth,H*(.045 if wide else .032));b=secondary.getbbox(sub)
    d.text((x-b[0],y-b[1]),sub,font=secondary,fill=subink+(255,))
    y+=b[3]-b[1]+int(H*.055 if wide else H*.03)
    buttonfont=fit_font(fontpath,cta,maxwidth*.75,H*(.05 if wide else .035));b=buttonfont.getbbox(cta);tw,th=b[2]-b[0],b[3]-b[1];pad=int(H*.022)
    dd.rounded_rectangle((x,y,x+tw+2*pad,y+th+2*pad),radius=max(2,pad//2),fill=accent+(255,))
    d.text((x+pad-b[0],y+pad-b[1]),cta,font=buttonfont,fill=(20,20,25,255))
    return text,dec,box,{'headline':offer,'subline':sub,'cta':cta,'font':str(fontpath),'font_sha256':hashlib.sha256(fontpath.read_bytes()).hexdigest(),'layout':layout}

def background(size,kind,seed):
    w,h=size;yy,xx=np.mgrid[:h,:w];rng=np.random.default_rng(seed)
    if kind=='solid_green':a=np.zeros((h,w,3),np.uint8);a[...,1]=255;return a
    if kind=='green_shade':
        t=xx/max(1,w-1) if rng.random()<.5 else yy/max(1,h-1);a=np.zeros((h,w,3),np.uint8);a[...,1]=np.uint8(190+65*t);return a
    t=np.clip((yy/max(1,h-1)-.42)/.58,0,1)
    return np.uint8(np.stack([t,np.ones_like(t),t],-1)*255)

def write_image(path,im):
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path)

def labeled(job):
    row,v,split=job;seed=9100000+row['id']*100+v;rng=np.random.default_rng(seed)
    # Most cases are horizontal; square/portrait remain realistic banner variations.
    size=[(640,320),(640,320),(640,320),(640,320),(640,320),(512,512),(384,512),(512,512)][((v*3)+(v//8)+row["id"])%8]
    kind='solid_green' if v%8<6 else 'green_shade' if v%8==6 else 'floor_gradient'
    w,h=size;text,dec,box,meta=design(size,seed);ss=2
    product=Image.open(ROOT/row['path']).convert('RGBA');product.thumbnail((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
    px=(box[0]+box[2]-product.width)//2;py=(box[1]+box[3]-product.height)//2
    product_layer=Image.new('RGBA',(w*ss,h*ss));product_layer.alpha_composite(product,(px,py))
    # Exactly one product asset. Text never overlaps the product box.
    fg=Image.alpha_composite(Image.alpha_composite(product_layer,dec),text).resize(size,Image.Resampling.LANCZOS)
    bg=background(size,kind,seed);rgba=np.asarray(fg,np.float32)/255;alpha=rgba[...,3:];rgb=rgba[...,:3]*alpha+(bg/255)*(1-alpha)
    quality=int(rng.integers(60,99));sampling=int(rng.choice([0,2]));stem=f'p{row["id"]:04d}_{v:02d}';folder=OUT/'labeled'/split
    folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(np.round(np.clip(rgb,0,1)*255))).save(folder/f'{stem}.jpg',quality=quality,subsampling=sampling)
    files={'input':folder/f'{stem}.jpg','rgba':folder/f'{stem}_rgba.png','alpha':folder/f'{stem}_alpha.png','product_alpha':folder/f'{stem}_product.png','text_alpha':folder/f'{stem}_text.png','decoration_alpha':folder/f'{stem}_decoration.png','background':folder/f'{stem}_background.png'}
    fg.save(files['rgba']);fg.getchannel('A').save(files['alpha']);product_layer.resize(size,Image.Resampling.LANCZOS).getchannel('A').save(files['product_alpha']);text.resize(size,Image.Resampling.LANCZOS).getchannel('A').save(files['text_alpha']);dec.resize(size,Image.Resampling.LANCZOS).getchannel('A').save(files['decoration_alpha']);Image.fromarray(bg).save(files['background'])
    return {'id':stem,'split':split,'source_product_id':row['id'],'source_sha256':row['sha256'],'category':row['category'],'product_count':1,'size':size,'background_kind':kind,'jpeg_quality':quality,'jpeg_subsampling':sampling,'seed':seed,'label_quality':'exact compositing alpha from source cutout and rendered text; source alpha not manually corrected',**meta,**{k:str(p.relative_to(ROOT)) for k,p in files.items()}}

def weak_banner(job):
    row,v=job;seed=9300000+row['id']*10+v;size=(640,320);w,h=size
    # Use a wide layout with product image on the right and text safely on the left.
    for offset in range(100):
        text,dec,_,meta=design(size,seed+offset)
        if meta['layout']=='product_right':break
    image=Image.open(ROOT/(row['path'] if row['background_qa_pass'] else row['original_path'])).convert('RGB');image.thumbnail((280,280),Image.Resampling.LANCZOS)
    x0=w-image.width-12;y0=(h-image.height)//2
    base=Image.new('RGB',size,(0,255,0));base.paste(image,(x0,y0));ink=Image.alpha_composite(dec,text).resize(size,Image.Resampling.LANCZOS)
    output=Image.alpha_composite(base.convert('RGBA'),ink).convert('RGB')
    hard=not row['background_qa_pass'];split='gradient_visual_test' if hard else 'train_weak' if row['split']=='train_weak' else 'solid_visual_test'
    stem=f'g{row["id"]:03d}_{v:02d}';folder=OUT/'gemini'/split;folder.mkdir(parents=True,exist_ok=True);output.save(folder/f'{stem}.jpg',quality=90,subsampling=2)
    result={'id':stem,'split':split,'source_generated_id':row['id'],'product_subject':row['subject'],'product_count':1,'size':size,'background_kind':'original_floor_or_white' if hard else 'normalized_green','input':str((folder/f'{stem}.jpg').relative_to(ROOT)),**meta,'label_quality':'visual only; no original product alpha'}
    if split=='train_weak':
        pix=np.asarray(image,np.float32)/255;strength=pix[...,1]-np.maximum(pix[...,0],pix[...,2]);fg=strength<.08;bg=(strength>.78)&(pix[...,1]>.9)
        mask=np.zeros((h,w),np.float32);conf=np.ones((h,w),np.float32)
        mask[y0:y0+image.height,x0:x0+image.width]=fg
        conf[y0:y0+image.height,x0:x0+image.width]=minimum_filter((fg|bg).astype(np.float32),size=7)*.1
        a=np.asarray(ink.getchannel('A'),np.float32)/255;mask=a+mask*(1-a)
        # All rendered text/decoration is outside the uncertain product region.
        Image.fromarray(np.uint8(np.round(mask*255))).save(folder/f'{stem}_weak_alpha.png');Image.fromarray(np.uint8(np.round(conf*255))).save(folder/f'{stem}_confidence.png')
        text.resize(size,Image.Resampling.LANCZOS).getchannel('A').save(folder/f'{stem}_text.png')
        result.update(label_quality='approximate product alpha; exact text alpha; use confidence map',weak_alpha=str((folder/f'{stem}_weak_alpha.png').relative_to(ROOT)),confidence=str((folder/f'{stem}_confidence.png').relative_to(ROOT)),text_alpha=str((folder/f'{stem}_text.png').relative_to(ROOT)))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--skip-generated",action="store_true");args=ap.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    sources=json.loads((ROOT/'datasets/products/manifest.json').read_text());sources=[r for r in sources if r['category'] not in ['groceries','vehicle','motorcycle']]
    jobs=[(r,i,r['split']) for r in sources for i in range(16 if r['split']=='train' else 8)]
    with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(labeled,jobs))
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    generated=[] if args.skip_generated else json.loads((ROOT/'datasets/generated_v3/prepared_manifest.json').read_text());jobs=[(r,i) for r in generated for i in range(4 if r['background_qa_pass'] and r['split']=='train_weak' else 2)]
    with ThreadPoolExecutor(max_workers=4) as pool:weak=list(pool.map(weak_banner,jobs))
    (OUT/'gemini_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in weak))
    report={'labeled_banners':len(rows),'source_products':len(sources),'labeled_splits':{s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},'source_splits':{s:sum(r['split']==s for r in sources) for s in ['train','validation','test']},'backgrounds':{s:sum(r['background_kind']==s for r in rows) for s in ['solid_green','green_shade','floor_gradient']},'gemini_derivatives':len(weak),'gemini_splits':{s:sum(r['split']==s for r in weak) for s in ['train_weak','solid_visual_test','gradient_visual_test']},'new_api_requests':0}
    (ROOT/'reports/banners_v1').mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/banners_v1/summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
