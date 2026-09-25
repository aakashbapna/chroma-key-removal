"""V3 regression benchmark; generated Gemini inputs are visual-only, never exact labels."""
import json,time,io
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from product_data import manifest,composite
from data import sample
from remove import rgba
from remove_v2 import ProductMatting
from remove_v3 import LargeMatting
from evaluate import metrics,key
ROOT=Path(__file__).resolve().parent

def mean(rows):return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
def render_row(x,truth,preds):
    h,w=x.shape[:2];yy,xx=np.mgrid[:h,:w];bg=(.65+.2*((xx//16+yy//16)%2))[...,None]*np.ones(3)
    out=[x]
    if truth is not None:
        f,a=truth;out.append(f*a[...,None]+bg*(1-a[...,None]))
    for a in preds.values():
        result=rgba(x,a)/255;out.append(result[...,:3]*result[...,3:]+bg*(1-result[...,3:]))
    return np.concatenate(out,1)
def save_sheet(rows,labels,path):
    a=Image.fromarray(np.uint8(np.clip(np.concatenate(rows),0,1)*255));a.thumbnail((1600,2400));im=Image.new('RGB',(a.width,a.height+32),'white');im.paste(a,(0,32));d=ImageDraw.Draw(im)
    for i,label in enumerate(labels):d.text((i*im.width//len(labels)+5,8),label,fill='black')
    im.save(path)

def known_alpha_test(methods):
    # Known-alpha generated source from prior run, never used in training.
    source=Image.open('datasets/generated/builtin_product_sheet.png').convert('RGBA');source.thumbnail((640,640),Image.Resampling.LANCZOS);f=np.asarray(source,np.float32)/255;a=f[...,3];f=f[...,:3]
    generated={k:[] for k in methods}
    for q in [60,80,95]:
        x=np.asarray(Image.open(f'examples/v2/generated_green_q{q}.png'),np.float32)/255
        for name,fn in methods.items():generated[name].append(metrics(a,fn(x)))
    Path('reports/v3/known_alpha_generated_metrics.json').write_text(json.dumps({'independent_sources':1,'methods':{k:mean(v) for k,v in generated.items()}},indent=2))

def main():
    methods={'v3':LargeMatting().alpha,'v2':ProductMatting().alpha,'color_key':key};records={k:[] for k in methods};per=[];sheet=[]
    for j,row in enumerate(manifest('test')):
        prs={k:[] for k in methods}
        for variant in range(3):
            x,a,f=composite(row,2000000+j*3+variant,256);a=a[...,0];preds={}
            for name,fn in methods.items():
                p=fn(x);preds[name]=p;r=metrics(a,p);result=rgba(x,p)/255
                truth=f*a[...,None]+.5*(1-a[...,None]);output=result[...,:3]*result[...,3:]+.5*(1-result[...,3:]);r['gray_composite_rgb_mae']=float(np.abs(truth-output).mean());records[name].append(r);prs[name].append(r)
            if variant==0:
                Image.fromarray(rgba(x,preds['v3'])).save(f'examples/v3/product_{row["id"]}.png')
                if j<12:sheet.append(render_row(x,(f,a),preds))
        per.append({'id':row['id'],'methods':{k:mean(v) for k,v in prs.items()}})
        print('Test product',j+1,flush=True)
    procedural={k:[] for k in methods}
    for i in range(32):
        x,a,_=sample(600000+i,128)
        for name,fn in methods.items():procedural[name].append(metrics(a[...,0],fn(x)))
    # Controlled floor-gradient benchmark with independently known alpha.
    floor={k:[] for k in methods};floor_sheet=[]
    for j,row in enumerate(manifest('test')):
        _,a,f=composite(row,8000000+j,256);a=a[...,0]
        yy=np.arange(256,dtype=np.float32)[:,None,None];mix=np.clip((yy/255-.4)/.6,0,1)
        backdrop=np.array([0,1,0])*(1-mix)+np.ones(3)*mix
        clean=f*a[...,None]+backdrop*(1-a[...,None]);buf=io.BytesIO()
        Image.fromarray(np.uint8(np.clip(clean,0,1)*255)).save(buf,format='JPEG',quality=85,subsampling=2)
        x=np.asarray(Image.open(buf),np.float32)/255;preds={k:fn(x) for k,fn in methods.items()}
        for name,pred in preds.items():floor[name].append(metrics(a,pred))
        if j<6:floor_sheet.append(render_row(x,(f,a),preds))
    Path('reports/v3/controlled_gradient_metrics.json').write_text(json.dumps({'source_products':23,'background':'Pure green upper 40%, linear ramp to white floor below; JPEG quality 85','label_source':'Known original product alpha, not pseudo-labels','methods':{k:mean(v) for k,v in floor.items()}},indent=2))
    save_sheet(floor_sheet,['Gradient JPEG','Source alpha target','V3','V2','Color key'],'reports/v3/controlled_gradient_comparison.png')
    diffs=np.array([r['methods']['v2']['alpha_mae']-r['methods']['v3']['alpha_mae'] for r in per]);rng=np.random.default_rng(73);bootstrap=np.mean(rng.choice(diffs,(5000,len(diffs))),axis=1)
    report={'test_products':len(per),'variants_per_product':3,'models':{k:mean(v) for k,v in records.items()},'procedural':{k:mean(v) for k,v in procedural.items()},'v2_minus_v3_alpha_mae_bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist(),'per_product':per,'label_source':'Source-provided downloaded alpha; identical test setup to V2.'}
    Path('reports/v3/metrics.json').write_text(json.dumps(report,indent=2));save_sheet(sheet,['Green JPEG','Source alpha target','V3','V2','Color key'],'reports/v3/product_comparison.png')
    print(json.dumps({k:v for k,v in report.items() if k!='per_product'},indent=2))
    known_alpha_test(methods)
    # New Gemini data: visual-only results on held-out subjects.
    rows=json.loads(Path('datasets/generated_v3/prepared_manifest.json').read_text());rows=[r for r in rows if r['background_qa_pass'] and r['split']=='test_visual_only'];sheet=[]
    for i,row in enumerate(rows):
        im=Image.open(row['path']).convert('RGB');im.thumbnail((384,384));x=np.asarray(im,np.float32)/255;preds={k:fn(x) for k,fn in methods.items()}
        Image.fromarray(rgba(x,preds['v3'])).save(f'examples/v3/gemini_{row["id"]:03d}.png')
        if i<12:sheet.append(render_row(x,None,preds))
        if i==0:im.save('examples/v3/demo_input.png')
    if sheet:save_sheet(sheet,['Gemini green input','V3','V2','Color key'],'reports/v3/gemini_visual_comparison.png')
    # User-requested hard cases: retain original floor gradients / white-background deviations.
    all_rows=json.loads(Path('datasets/generated_v3/prepared_manifest.json').read_text())
    hard=[r for r in all_rows if not r['background_qa_pass']];hard_sheet=[]
    for row in hard:
        im=Image.open(row['original_path']).convert('RGB');im.thumbnail((384,384));x=np.asarray(im,np.float32)/255
        preds={k:fn(x) for k,fn in methods.items()}
        Image.fromarray(rgba(x,preds['v3'])).save(f'examples/v3/gradient_{row["id"]:03d}.png')
        hard_sheet.append(render_row(x,None,preds))
    if hard_sheet:save_sheet(hard_sheet,['Original gradient/floor','V3','V2','Color key'],'reports/v3/gradient_visual_comparison.png')
    Path('reports/v3/gradient_visual_test.json').write_text(json.dumps({'count':len(hard),'ids':[r['id'] for r in hard],'purpose':'User-requested floor-gradient and nonuniform-background hard cases; these images were excluded from weak training, but two subjects have another training view','quantitative_alpha_metrics':None,'reason':'No independently known product masks; visual evaluation only.'},indent=2))
    Path('reports/v3/gemini_visual_test.json').write_text(json.dumps({'count':len(rows),'ids':[r['id'] for r in rows],'quantitative_alpha_metrics':None,'reason':'No independent alpha labels exist for green-background Gemini outputs.'},indent=2))
if __name__=='__main__':
    import sys
    if '--known-alpha-only' in sys.argv:known_alpha_test({'v3':LargeMatting().alpha,'v2':ProductMatting().alpha,'color_key':key})
    else:main()
