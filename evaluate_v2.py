"""Compare larger/small/DeepLab/color key on held-out ecommerce source products."""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from product_data import manifest,composite
from data import sample
from remove import Matting,rgba
from remove_v2 import ProductMatting
from evaluate import DeepLab,key,metrics

def mean(records):return {k:float(np.mean([r[k] for r in records])) for k in records[0]}

def main():
    new=ProductMatting();old=Matting();deep=DeepLab('/Users/aakash/Downloads/deeplabv3.tflite')
    methods={'larger_v2':new.alpha,'small_v1':old.alpha,'color_key':key,'deeplab':lambda x:deep.alpha(x,'minus_one_one')}
    records={k:[] for k in methods};rows=manifest('test');per_product=[];panels=[]
    for j,row in enumerate(rows):
        product={k:[] for k in methods}
        for variant in range(3):
            x,y,fg=composite(row,2000000+j*3+variant,256);y=y[...,0];preds={}
            for name,fn in methods.items():
                start=time.perf_counter();a=fn(x);ms=(time.perf_counter()-start)*1000;r=metrics(y,a)
                r['native_ms_256']=ms
                out=rgba(x,a).astype(np.float32)/255
                # Color reconstruction matters independently of alpha.
                truth=fg*y[...,None]+(1-y[...,None])*.5
                pred=out[...,:3]*out[...,3:]+(1-out[...,3:])*.5
                r['gray_composite_rgb_mae']=float(np.abs(truth-pred).mean())
                records[name].append(r);product[name].append(r);preds[name]=out
            if variant==0:
                Image.fromarray(np.uint8(x*255)).save(f'examples/v2/product_{row["id"]}_input.png')
                Image.fromarray(np.uint8(np.round(preds['larger_v2']*255))).save(f'examples/v2/product_{row["id"]}_result.png')
                Image.fromarray(np.uint8(y*255)).save(f'examples/v2/product_{row["id"]}_alpha.png')
                if j<12:
                    yy,xx=np.mgrid[:256,:256];bg=(.65+.2*((xx//16+yy//16)%2))[...,None]*np.ones(3)
                    items=[x,fg*y[...,None]+bg*(1-y[...,None])]
                    for name in ['larger_v2','small_v1','color_key','deeplab']:
                        out=preds[name];items.append(out[...,:3]*out[...,3:]+bg*(1-out[...,3:]))
                    panels.append(np.concatenate(items,axis=1))
        green_fraction=float(np.mean((fg[...,1]>np.maximum(fg[...,0],fg[...,2])+.15)&(y>.8)))
        per_product.append({'green_foreground_fraction':green_fraction,'id':row['id'],'title':row['title'],'category':row['category'],'methods':{k:mean(v) for k,v in product.items()}})
        print('Evaluated product',j+1,'/',len(rows),flush=True)
    shape_records={k:[] for k in ['larger_v2','small_v1','color_key']}
    for i in range(32):
        x,y,_=sample(600000+i,128)
        for name in shape_records:shape_records[name].append(metrics(y[...,0],methods[name](x)))
    # Bootstrap over independent source products, not correlated augmentation variants.
    diffs=np.array([r['methods']['small_v1']['alpha_mae']-r['methods']['larger_v2']['alpha_mae'] for r in per_product]);rng=np.random.default_rng(72)
    boot=np.mean(rng.choice(diffs,(5000,len(diffs))),axis=1)
    report={'test_products':len(rows),'variants_per_product':3,'image_size':256,'models':{k:mean(v) for k,v in records.items()},'procedural_regression':{k:mean(v) for k,v in shape_records.items()},'v1_minus_v2_alpha_mae_product_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'per_product':per_product,'model_bytes':Path('models/v2/product_chroma.tflite').stat().st_size,'test_label_source':'Original downloaded RGBA alpha; source quality not manually corrected','deeplab_preprocessing':'[-1,1], class 0 background (same assumptions as v1 evaluation)','generated_images':'No generated images included in quantitative test without independent alpha labels.'}
    Path('reports/v2/metrics.json').write_text(json.dumps(report,indent=2))
    sheet=Image.new('RGB',(1536,len(panels)*256+32),'white');sheet.paste(Image.fromarray(np.uint8(np.clip(np.concatenate(panels),0,1)*255)),(0,32));d=ImageDraw.Draw(sheet)
    for i,label in enumerate(['Green/JPEG input','Original alpha target','Larger v2','Small v1','Color key','DeepLab']):d.text((i*256+8,8),label,fill='black')
    sheet.save('reports/v2/product_comparison.png')
    sheet.resize((960,round(sheet.height*960/sheet.width))).save('reports/v2/product_comparison_preview.png')
    print(json.dumps({k:v for k,v in report.items() if k!='per_product'},indent=2))
if __name__=='__main__':main()
