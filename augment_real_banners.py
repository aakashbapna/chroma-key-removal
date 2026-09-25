"""Additional local derivatives of user examples; training-only, no API requests.

Watch compositing masks inherit estimated source alpha, not hand-labeled truth.
Makeup is an exact opaque-preservation task, including all decorative cubes.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from chroma import ROOT
OUT=ROOT/'datasets/real_variants'
CASES=['pure_green','dark_green','off_green','floor_gradient','green_spill','heavy_jpeg','small_layout','soft_edges']
def main():
 OUT.mkdir(parents=True,exist_ok=True);rows=[];previews=[]
 for subject in ['makeup','watch']:
  source=ROOT/'datasets/real_banners'/f'{subject}.jpg';im=Image.open(source).convert('RGB');x=np.asarray(im,np.float32)/255
  if subject=='watch':
   source_rgba=Image.open(ROOT/'datasets/real_banners/annotations/watch.png').convert('RGBA')
  else:source_rgba=im.convert('RGBA')
  for k in range(128):
   seed=890000+(10000 if subject=='watch' else 0)+k;rng=np.random.default_rng(seed);case=CASES[k%len(CASES)] if subject=='watch' else ['lavender','cream','blue','pink'][k%4]
   size=(640,320) if subject=='makeup' else [(448,600),(512,640),(640,640),(384,512)][(k*3+k//8)%4];w,h=size
   angle=float(rng.uniform(-3,3));fg=source_rgba.rotate(angle,Image.Resampling.BICUBIC,expand=True);factor=float(rng.uniform(.7,.96)) if case=='small_layout' else float(rng.uniform(.9,.98));fg.thumbnail((int(w*factor),int(h*factor)),Image.Resampling.LANCZOS)
   layer=Image.new('RGBA',size);left=int(rng.integers(0,w-fg.width+1));top=int(rng.integers(0,h-fg.height+1));layer.alpha_composite(fg,(left,top));a=np.asarray(layer,np.float32)/255
   yy,xx=np.mgrid[:h,:w];bg=np.zeros((h,w,3),np.float32);bg[...,1]=1
   if subject=='makeup':
    palette={'lavender':[.84,.76,.96],'cream':[.96,.93,.84],'blue':[.65,.8,.97],'pink':[.95,.76,.85]};bg[:]=palette[case]
   elif case=='dark_green':bg[...,1]=rng.uniform(.55,.8)
   elif case=='off_green':bg[:]=[rng.uniform(.03,.15),rng.uniform(.75,.95),rng.uniform(.03,.15)]
   elif case=='floor_gradient':
    t=np.clip((yy/(h-1)-rng.uniform(.35,.6))/.6,0,1)[...,None];bg=bg*(1-t)+np.array([.93,.97,.96])*t
   elif case=='soft_edges':
    from PIL import ImageFilter
    layer.putalpha(layer.getchannel('A').filter(ImageFilter.GaussianBlur(float(rng.uniform(.35,.9)))));a=np.asarray(layer,np.float32)/255
   if case=='green_spill':
    from scipy.ndimage import maximum_filter
    edge=maximum_filter(a[...,3]<.99,size=5)&(a[...,3]>.01);a[edge,:3]=.8*a[edge,:3]+.2*np.array([0,1,0])
   rgb=a[...,:3]*a[...,3:]+bg*(1-a[...,3:]);target=a[...,3]
   if subject=='makeup':target=np.ones((h,w),np.float32)
   stem=f'{subject}_{k:03d}';quality=int(rng.integers(20,46)) if case=='heavy_jpeg' else int(rng.integers(65,99));Image.fromarray(np.uint8(np.round(np.clip(rgb,0,1)*255))).save(OUT/f'{stem}.jpg',quality=quality,subsampling=int(rng.choice([0,2])))
   Image.fromarray(np.uint8(np.round(target*255))).save(OUT/f'{stem}_alpha.png')
   # A conservative weight keeps teacher-derived watch labels from dominating true masks.
   conf=np.ones((h,w),np.float32) if subject=='makeup' else np.full((h,w),.15,np.float32)
   Image.fromarray(np.uint8(np.round(conf*255))).save(OUT/f'{stem}_confidence.png');Image.new('L',size).save(OUT/f'{stem}_text.png')
   row={'id':'variant_'+stem,'split':'train','source':'user_supplied_'+subject,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'case':case,'size':size,'rotation_degrees':angle,'seed':seed,'jpeg_quality':quality,'label_quality':'exact opaque preservation requested by user' if subject=='makeup' else 'composited from estimated source alpha; not independent ground truth','input':str((OUT/f'{stem}.jpg').relative_to(ROOT)),'weak_alpha':str((OUT/f'{stem}_alpha.png').relative_to(ROOT)),'confidence':str((OUT/f'{stem}_confidence.png').relative_to(ROOT)),'text_alpha':str((OUT/f'{stem}_text.png').relative_to(ROOT))};rows.append(row)
   if k<8:
    view=Image.open(OUT/f'{stem}.jpg');view=ImageOps.contain(view,(300,220));tile=Image.new('RGB',(320,250),'white');tile.paste(view,((320-view.width)//2,25+(220-view.height)//2));ImageDraw.Draw(tile).text((8,7),subject+' / '+case,fill='black');previews.append(tile)
 (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 report=ROOT/'reports/real_banners';sheet=Image.new('RGB',(1280,1000),'white')
 for k,tile in enumerate(previews):sheet.paste(tile,((k%4)*320,(k//4)*250))
 sheet.save(report/'synthetic_variants.jpg')
 summary={'new_images':len(rows),'makeup':128,'watch':128,'split':'training only; both source images previously used for training','new_api_calls':0,'makeup_contract':'No cropping; both cubes retained; exact fully opaque target','watch_label_limit':'Estimated source alpha reused, so masks remain weak supervision','weights_changed':False}
 (report/'synthetic_variants.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
