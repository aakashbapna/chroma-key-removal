"""Visual-only tests of pre-existing V3 on generated product offer banners."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from remove_v3 import LargeMatting
from remove import rgba
ROOT=Path(__file__).resolve().parent
out=ROOT/'reports/banners_v1/evaluation';out.mkdir(parents=True,exist_ok=True)
rows=list(map(json.loads,(ROOT/'datasets/banners_v1/gemini_manifest.jsonl').read_text().splitlines()))
m=LargeMatting();sheet=Image.new('RGB',(1280,696),'white');d=ImageDraw.Draw(sheet)
for k,split in enumerate(['solid_visual_test','gradient_visual_test']):
    row=next(r for r in rows if r['split']==split);im=Image.open(ROOT/row['input']).convert('RGB');x=np.asarray(im,np.float32)/255
    result=Image.fromarray(rgba(x,m.alpha(x)));result.save(out/f'{row["id"]}_v3.png')
    yy,xx=np.indices((im.height,im.width));g=np.where((xx//16+yy//16)%2,215,245).astype('uint8');bg=Image.fromarray(np.stack([g]*3,-1)).convert('RGBA');preview=Image.alpha_composite(bg,result).convert('RGB')
    y=k*348;d.text((8,y+8),f'{split}: JPEG input',fill='black');d.text((648,y+8),'Existing V3 transparency',fill='black');sheet.paste(im,(0,y+28));sheet.paste(preview,(640,y+28))
sheet.save(out/'generated_comparison.jpg')
