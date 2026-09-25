"""Paired evaluation against the user-supplied DeepLab TFLite artifact.

DeepLab input normalization is chosen on validation only. Test inputs/targets
match banner_experiments.py. Semantic outputs are converted using both hard
non-background labels and 1 - softmax(background); neither is calibrated alpha.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse,json,time,hashlib,platform,csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import tensorflow as tf
from banner_data import ROOT,records
from banner_experiments import resized,edge_records,metrics
OUT=ROOT/'reports/deeplab_comparison'
NORMS=['minus_one_one','zero_one','raw']

def scale(a,size):return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size,Image.Resampling.BILINEAR))
class DeepLab:
    def __init__(self,path,threads=4):
        self.m=tf.lite.Interpreter(model_path=str(path),num_threads=threads);self.m.allocate_tensors();self.i=self.m.get_input_details()[0];self.o=self.m.get_output_details()[0]
        assert tuple(self.i['shape'])==(1,257,257,3) and tuple(self.o['shape'])==(1,257,257,21)
    def predict(self,x,norm):
        x=np.asarray(Image.fromarray(np.uint8(np.round(np.clip(x,0,1)*255))).resize((257,257),Image.Resampling.BILINEAR),np.float32)
        if norm=='minus_one_one':x=x/127.5-1
        elif norm=='zero_one':x=x/255
        start=time.perf_counter();self.m.set_tensor(self.i['index'],x[None]);self.m.invoke();logits=self.m.get_tensor(self.o['index'])[0];elapsed=time.perf_counter()-start
        hard=(logits.argmax(-1)!=0).astype(np.float32);z=logits-logits.max(-1,keepdims=True);prob=np.exp(z);soft=1-prob[...,0]/prob.sum(-1)
        return {'hard':scale(hard,(256,256)),'soft':scale(soft,(256,256))},elapsed
class Banner:
    def __init__(self,path):
        self.m=tf.lite.Interpreter(model_path=str(path),num_threads=4);self.m.allocate_tensors();self.i=self.m.get_input_details()[0]['index'];self.o=self.m.get_output_details()[0]['index']
    def predict(self,x):
        start=time.perf_counter();self.m.set_tensor(self.i,x[None]);self.m.invoke();a=self.m.get_tensor(self.o)[0,...,0];return a,time.perf_counter()-start

def sample(row):
    x,y=resized(row,256);h,w=x.shape[:2];return np.pad(x,((0,256-h),(0,256-w),(0,0)),mode='edge'),y,h,w

def score(row,y,p,seconds,suite):
    a=y[...,0];pred=p>=.5;gt=a>=.5;err=np.abs(p-a);soft=(a>.02)&(a<.98);text=y[...,2]>.02;bg=a<.01;tc=(y[...,2]>=.5)&gt
    return {'id':row['id'],'source_product_id':row['source_product_id'],'suite':suite,'group':row.get('edge_case',row['background_kind']),'counts':[int(np.sum(pred&gt)),int(np.sum(pred&~gt)),int(np.sum(~pred&gt)),int(np.sum(~pred&~gt))],'alpha_mae':float(err.mean()),'soft_edge_mae':float(err[soft].mean()) if soft.any() else None,'text_alpha_mae':float(err[text].mean()) if text.any() else None,'background_leakage':float(p[bg].mean()) if bg.any() else None,'seconds':seconds,'text_counts':[int(np.sum(pred&tc)),int(tc.sum())]}
def aggregate(rows):
    m=metrics(rows);t=np.sum([r['text_counts'] for r in rows],axis=0);m['text_core_recall']=float(t[0]/max(t[1],1));m['median_inference_ms']=float(np.median([r['seconds'] for r in rows])*1000);m['p95_inference_ms']=float(np.quantile([r['seconds'] for r in rows],.95)*1000);return m

def checker(rgb,a):
    h,w=a.shape;yy,xx=np.indices((h,w));bg=np.where((xx//12+yy//12)%2,.82,.96)[...,None];return np.uint8(np.clip((rgb*a[...,None]+bg*(1-a[...,None]))*255,0,255))
def display(row,x,y,preds):
    h,w=y.shape[:2];rgba=np.asarray(Image.open(ROOT/row['rgba']).convert('RGBA').resize((w,h),Image.Resampling.LANCZOS),np.float32)/255
    panels=[np.uint8(x[:h,:w]*255),checker(rgba[...,:3],y[...,0])]
    # Same observed RGB for all prediction previews: isolates the masks; no despill.
    panels += [checker(x[:h,:w],a) for a in preds.values()]
    labels=['JPEG input','Target']+[{'banner_run1':'Banner Run 1','banner_run2':'Banner Run 2','banner_real':'Real-image fine-tune','deeplab_hard':'DeepLab hard','deeplab_soft':'DeepLab soft'}[n] for n in preds]
    out=Image.new('RGB',(288*len(panels),190),'white');d=ImageDraw.Draw(out)
    for k,(a,label) in enumerate(zip(panels,labels)):
        im=ImageOps.contain(Image.fromarray(a),(280,156));out.paste(im,(288*k+(288-im.width)//2,28+(156-im.height)//2));d.text((288*k+7,7),label,fill='black')
    out.save(OUT/f'{row["id"]}.jpg');return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deeplab',default='/Users/aakash/Downloads/deeplabv3.tflite');args=ap.parse_args();path=Path(args.deeplab);OUT.mkdir(parents=True,exist_ok=True)
    dl=DeepLab(path);bm={name:Banner(ROOT/'models'/name/'product_chroma.tflite') for name in [n for n in ['banner_run1','banner_run2','banner_real'] if (ROOT/'models'/n/'product_chroma.tflite').exists()]}
    dummy=np.zeros((256,256,3),np.float32);dummy[...,1]=1
    for _ in range(3):
        dl.predict(dummy,'minus_one_one')
        for model in bm.values():model.predict(dummy)
    val={n:{d:[] for d in ['hard','soft']} for n in NORMS}
    validation=records('validation')+edge_records('validation')
    for row in validation:
        x,y,h,w=sample(row)
        for norm in NORMS:
            preds,sec=dl.predict(x,norm)
            for decoder,p in preds.items():val[norm][decoder].append(score(row,y,p[:h,:w],sec,'validation'))
    validation_summary={n:{d:aggregate(v) for d,v in dec.items()} for n,dec in val.items()}
    selected={d:max(NORMS,key=lambda n:validation_summary[n][d]['iou']) for d in ['hard','soft']}
    print('Selected normalization:',selected,flush=True)
    all_results={k:[] for k in [*bm,'deeplab_hard','deeplab_soft']};pictures=[];seen=set()
    for suite,rows in [('banners',records('test')),('edge_cases',edge_records('test'))]:
        for row in rows:
            x,y,h,w=sample(row);preds={}
            for name,model in bm.items():p,sec=model.predict(x);p=p[:h,:w];preds[name]=p;all_results[name].append(score(row,y,p,sec,suite))
            dl_preds={n:dl.predict(x,n) for n in set(selected.values())}
            for decoder,norm in selected.items():
                name='deeplab_'+decoder;p=dl_preds[norm][0][decoder][:h,:w];preds[name]=p;all_results[name].append(score(row,y,p,dl_preds[norm][1],suite))
            group=row.get('edge_case',row['background_kind'])
            if group not in seen:seen.add(group);pictures.append(display(row,x,y,preds))
        print(suite,{name:aggregate([r for r in rows_ if r['suite']==suite]) for name,rows_ in all_results.items()},flush=True)
    sheet=Image.new('RGB',(pictures[0].width,190*len(pictures)),'white')
    for k,p in enumerate(pictures):sheet.paste(p,(0,k*190))
    sheet.save(OUT/'visual_comparison.jpg')
    result={'protocol':{'test_banners':144,'test_edge_cases':144,'test_product_assets':18,'validation_images':len(validation),'shared_test_sources_between_suites':True,'preprocessing':'Same aspect-preserving resize to fit 256 and bottom/right edge padding as earlier banner tests. DeepLab padded canvas rescaled to 257; output masks bilinear-rescaled to 256; padding excluded. Input normalized per validation selection.','threshold':.5,'metrics':'Micro pixel confusion metrics; per-image average alpha, edge and text errors. Text-core recall uses target glyph alpha >= .5 and combined target alpha >= .5.','deeplab_assumption':'Output channel 0 is background, channels 1–20 foreground. No class metadata supplied; softmax probabilities are not calibrated opacity.','selection':'Normalization chosen separately for hard and soft decoding by highest validation micro foreground IoU; no threshold or normalization selection on test.','timing':'Four CPU threads per interpreter, three warm-ups. Set input + invoke + get output only; excludes preprocessing, resizing, softmax, file I/O and browser runtime. Models are measured sequentially on the same inputs.','system':platform.platform(),'tensorflow':tf.__version__},'models':{'deeplab':{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input':[1,257,257,3],'output':[1,257,257,21]},**{n:{'bytes':(ROOT/'models'/n/'product_chroma.tflite').stat().st_size,'sha256':hashlib.sha256((ROOT/'models'/n/'product_chroma.tflite').read_bytes()).hexdigest(),'input':[1,256,256,3],'output':[1,256,256,1]} for n in bm}},'normalization_validation':validation_summary,'selected_normalization':selected,'results':{name:{'suites':{s:aggregate([r for r in rows if r['suite']==s]) for s in ['banners','edge_cases']},'groups':{g:aggregate([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})},'samples':rows} for name,rows in all_results.items()}}
    # Product-cluster bootstrap respects the correlated variants from a source.
    rng=np.random.default_rng(20260925);cis={};selected_model=json.loads((ROOT/'models/banner_latest/selection.json').read_text())['selected_run']
    for suite in ['banners','edge_cases']:
        for dl_name in ['deeplab_hard','deeplab_soft']:
            a={r['id']:r for r in all_results[dl_name] if r['suite']==suite};b=[r for r in all_results[selected_model] if r['suite']==suite];ids=sorted({r['source_product_id'] for r in b});delta=np.array([np.mean([a[r['id']]['alpha_mae']-r['alpha_mae'] for r in b if r['source_product_id']==pid]) for pid in ids]);boot=rng.choice(delta,(5000,len(delta)),replace=True).mean(1);cis[f'{suite}:{dl_name}_minus_{selected_model}']={'mean_alpha_mae_difference':float(delta.mean()),'product_cluster_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'clusters':len(ids)}
    result['paired_alpha_error_differences']=cis;result['selected_model']=selected_model
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    with (OUT/'summary.csv').open('w') as f:
        keys=['method','suite','pixel_accuracy','precision','recall','f1','iou','alpha_mae','soft_edge_mae','text_alpha_mae','text_core_recall','median_inference_ms'];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for name,r in result['results'].items():
            for suite,m in r['suites'].items():writer.writerow({k:{'method':name,'suite':suite,**m}[k] for k in keys})
    print('Finished',OUT,flush=True)
if __name__=='__main__':main()
