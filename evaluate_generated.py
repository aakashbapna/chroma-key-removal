"""Held-out generated RGBA sheet: exact compositing alpha, not a recovered pseudo-mask."""
import io,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from remove import Matting,rgba
from remove_v2 import ProductMatting
from evaluate import metrics,key

def main():
    source=Image.open('datasets/generated/builtin_product_sheet.png').convert('RGBA');source.thumbnail((640,640),Image.Resampling.LANCZOS)
    arr=np.asarray(source,np.float32)/255;fg=arr[...,:3];a=arr[...,3];clean=fg*a[...,None]+np.array([0,1,0])*(1-a[...,None])
    methods={'larger_v2':ProductMatting().alpha,'small_v1':Matting().alpha,'color_key':key};records={k:[] for k in methods};panels=[]
    for quality in [60,80,95]:
        buf=io.BytesIO();Image.fromarray(np.uint8(clean*255)).save(buf,format='JPEG',quality=quality,subsampling=2)
        rgb=np.asarray(Image.open(buf).convert('RGB'),np.float32)/255
        Image.open(buf).save(f'examples/v2/generated_green_q{quality}.png')
        yy,xx=np.mgrid[:a.shape[0],:a.shape[1]];bg=(.65+.2*((xx//20+yy//20)%2))[...,None]*np.ones(3)
        row=[rgb,fg*a[...,None]+bg*(1-a[...,None])]
        for name,fn in methods.items():
            pred=fn(rgb);records[name].append(metrics(a,pred));out=rgba(rgb,pred)
            Image.fromarray(out).save(f'examples/v2/generated_{name}_q{quality}.png')
            out=out/255;row.append(out[...,:3]*out[...,3:]+bg*(1-out[...,3:]))
        panels.append(np.concatenate(row,1))
    report={'independent_generated_sources':1,'products_in_sheet':6,'jpeg_qualities':[60,80,95],'training_use':'None; entire generated source held out','ground_truth':'Actual generated PNG alpha, preserved before green compositing; not a hand-corrected real-world mask','methods':{k:{m:float(np.mean([r[m] for r in rs])) for m in rs[0]} for k,rs in records.items()}}
    Path('reports/v2/generated_metrics.json').write_text(json.dumps(report,indent=2))
    sheet=Image.fromarray(np.uint8(np.clip(np.concatenate(panels),0,1)*255));sheet.thumbnail((1600,1600))
    labeled=Image.new('RGB',(sheet.width,sheet.height+32),'white');labeled.paste(sheet,(0,32));draw=ImageDraw.Draw(labeled)
    for i,label in enumerate(['Green JPEG (Q60/80/95)','Preserved source alpha','Larger v2','Small v1','Color key']):draw.text((i*sheet.width//5+6,8),label,fill='black')
    labeled.save('reports/v2/generated_comparison.png');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
