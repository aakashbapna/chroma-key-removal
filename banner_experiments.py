"""Banner fine-tuning and fixed, reproducible TFLite evaluation.

Evaluation uses all held-out banners scaled to fit 256x256, preserving aspect
ratio. Edge padding supplies the rest of the fixed TFLite input; padding is never
scored. These are standardized-resolution model metrics, not native tiled metrics.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from banner_data import ROOT,records,load_sample
from training_loss import alpha_loss

def edge_records(split):
    return [r for r in map(json.loads,(ROOT/'datasets/banner_edges/manifest.jsonl').read_text().splitlines()) if r['split']==split]
def resized(row,size):
    x,y=load_sample(row);h,w=x.shape[:2];scale=min(size/w,size/h);ww,hh=max(1,round(w*scale)),max(1,round(h*scale))
    x=np.asarray(Image.fromarray(np.uint8(x*255)).resize((ww,hh),Image.Resampling.LANCZOS),np.float32)/255
    y=np.stack([np.asarray(Image.fromarray(y[...,k],mode='F').resize((ww,hh),Image.Resampling.BILINEAR)) for k in range(y.shape[-1])],-1)
    return x,y

def training_arrays(rows,seed,size=128):
    rng=np.random.default_rng(seed);xs=[];ys=[]
    for row in rows:
        # Include global context as well as native-resolution text/boundary crops.
        if rng.random()<.5:
            x,y=resized(row,size);h,w=x.shape[:2];x=np.pad(x,((0,size-h),(0,size-w),(0,0)),mode='edge');y=np.pad(y,((0,size-h),(0,size-w),(0,0)))
        else:
            x,y=load_sample(row);h,w=x.shape[:2]
            if rng.random()<.65:
                coords=np.argwhere((y[...,2]>.05)|((y[...,0]>.02)&(y[...,0]<.98)))
                if len(coords):cy,cx=coords[int(rng.integers(len(coords)))];top=int(np.clip(cy-size//2,0,h-size));left=int(np.clip(cx-size//2,0,w-size))
                else:top=int(rng.integers(h-size+1));left=int(rng.integers(w-size+1))
            else:top=int(rng.integers(h-size+1));left=int(rng.integers(w-size+1))
            x=x[top:top+size,left:left+size];y=y[top:top+size,left:left+size]
        xs.append(x);ys.append(y)
    return np.stack(xs),np.stack(ys)

@tf.keras.utils.register_keras_serializable()
def banner_loss(y,p):
    # Existing boundary loss plus text-aware supervision; confidence masks padding.
    a=y[...,:1];c=y[...,1:2];t=y[...,2:3]
    text=tf.reduce_sum(c*t*tf.abs(a-p))/tf.maximum(tf.reduce_sum(c*t),1.)
    return alpha_loss(y,p)+.08*text

def export(model,folder):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    @tf.function(input_signature=[tf.TensorSpec([1,256,256,3],tf.float32,name='rgb')])
    def infer(x):return model(x,training=False)
    frozen=convert_variables_to_constants_v2(infer.get_concrete_function());c=tf.lite.TFLiteConverter.from_concrete_functions([frozen]);c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS];c.optimizations=[tf.lite.Optimize.DEFAULT];c.target_spec.supported_types=[tf.float16]
    data=c.convert();assert len(data)<10_000_000;path=folder/'product_chroma.tflite';path.write_bytes(data)
    (folder/'product_chroma.json').write_text(json.dumps({'tile':256,'halo':64,'stride':128,'alignment':8,'input':'float32 RGB [0,1]','output':'float32 alpha [0,1]','parameters':model.count_params(),'bytes':len(data)},indent=2))
    return path

def metrics(items):
    counts=np.sum([r['counts'] for r in items],axis=0);tp,fp,fn,tn=map(float,counts);eps=1e-12
    precision=tp/max(tp+fp,eps);recall=tp/max(tp+fn,eps);specificity=tn/max(tn+fp,eps)
    return {'images':len(items),'pixel_accuracy':(tp+tn)/max(sum(counts),eps),'precision':precision,'recall':recall,'specificity':specificity,'balanced_accuracy':(recall+specificity)/2,'f1':2*tp/max(2*tp+fp+fn,eps),'iou':tp/max(tp+fp+fn,eps),**{k:float(np.mean([r[k] for r in items if r[k] is not None])) if any(r[k] is not None for r in items) else None for k in ['alpha_mae','soft_edge_mae','text_alpha_mae','background_leakage','seconds']}}
def evaluate(path,run):
    out=ROOT/'reports/banner_runs'/run;out.mkdir(parents=True,exist_ok=True)
    m=tf.lite.Interpreter(model_path=str(path),num_threads=4);m.allocate_tensors();i=m.get_input_details()[0]['index'];o=m.get_output_details()[0]['index'];samples=[]
    for suite,rows in [('banners',records('test')),('edge_cases',edge_records('test'))]:
        for row in rows:
            x,y=resized(row,256);h,w=x.shape[:2];x=np.pad(x,((0,256-h),(0,256-w),(0,0)),mode='edge');start=time.perf_counter();m.set_tensor(i,x[None]);m.invoke();p=m.get_tensor(o)[0,:h,:w,0];elapsed=time.perf_counter()-start;a=y[...,0];pred=p>=.5;gt=a>=.5;err=np.abs(p-a);soft=(a>.02)&(a<.98);text=y[...,2]>.02;bg=a<.01
            samples.append({'id':row['id'],'suite':suite,'group':row.get('edge_case',row['background_kind']),'counts':[int(np.sum(pred&gt)),int(np.sum(pred&~gt)),int(np.sum(~pred&gt)),int(np.sum(~pred&~gt))],'alpha_mae':float(err.mean()),'soft_edge_mae':float(err[soft].mean()) if soft.any() else None,'text_alpha_mae':float(err[text].mean()) if text.any() else None,'background_leakage':float(p[bg].mean()) if bg.any() else None,'seconds':elapsed})
        print(run,suite,metrics([r for r in samples if r['suite']==suite]),flush=True)
    result={'run':run,'model':str(path.relative_to(ROOT)),'model_bytes':path.stat().st_size,'protocol':'All 144 held-out banners and 144 edge cases; resize to fit 256, aspect preserved, edge-pad bottom/right, padding excluded; binary threshold .5 for prediction and target; micro confusion metrics, per-image mean alpha metrics; no test examples used in training or checkpoint selection. Edge cases share the 18 test products with banner tests.','suites':{s:metrics([r for r in samples if r['suite']==s]) for s in ['banners','edge_cases']},'groups':{g:metrics([r for r in samples if r['group']==g]) for g in sorted({r['group'] for r in samples})},'samples':samples}
    (out/'metrics.json').write_text(json.dumps(result,indent=2));return result

def train(args):
    tf.keras.utils.set_random_seed(args.seed);tf.config.threading.set_intra_op_parallelism_threads(6);tf.config.threading.set_inter_op_parallelism_threads(2)
    rows=records('train',include_weak=not args.no_weak)+(edge_records('train') if args.edges else [])
    real_rows=list(map(json.loads,(ROOT/'datasets/real_banners/manifest.jsonl').read_text().splitlines())) if args.real else []
    rows+=real_rows
    # Fixed validation for comparable selection across runs, including edge cases.
    vx,vy=training_arrays(records('validation')+edge_records('validation'),99021)
    model=tf.keras.models.load_model(ROOT/args.source,compile=False)
    for layer in model.layers:
        if isinstance(layer,tf.keras.Model):layer.trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr,clipnorm=1.),loss=banner_loss)
    folder=ROOT/'models'/args.run;folder.mkdir(parents=True,exist_ok=True);reportdir=ROOT/'reports/banner_runs'/args.run;reportdir.mkdir(parents=True,exist_ok=True)
    initial=float(model.evaluate(vx,vy,batch_size=12,verbose=0));best=initial;model.save(folder/'best.keras');history=[];start=time.time()
    for epoch in range(args.epochs):
        x,y=training_arrays(rows,args.seed+epoch)
        progress=tf.keras.callbacks.LambdaCallback(on_train_batch_end=lambda batch,logs: print('BATCH',batch+1,'loss',float(logs['loss']),flush=True) if (batch+1)%50==0 else None)
        h=model.fit(x,y,validation_data=(vx,vy),batch_size=12,epochs=1,verbose=2,callbacks=[progress]).history
        val=float(h['val_loss'][0]);history.append({'epoch':epoch+1,'loss':float(h['loss'][0]),'val_loss':val})
        if val<best:best=val;model.save(folder/'best.keras')
        (reportdir/'training.json').write_text(json.dumps({'source':args.source,'parameters':model.count_params(),'real_example_crops':len(real_rows),'training_examples_per_epoch':len(rows),'weak_examples':sum('weak_alpha' in r and r.get('source')!='user_supplied_makeup' for r in rows),'real_exact_opaque_crops':sum(r.get('source')=='user_supplied_makeup' for r in rows),'real_weak_crops':sum(r.get('source')=='user_supplied_watch' for r in rows),'edge_examples':1024 if args.edges else 0,'initial_validation_loss':initial,'best_validation_loss':best,'selection':'minimum fixed validation loss including unchanged starting checkpoint','epochs':history,'seconds':time.time()-start,'seed':args.seed,'learning_rate':args.lr},indent=2))
        print('EPOCH_REPORT',json.dumps(history[-1]),flush=True)
    model=tf.keras.models.load_model(folder/'best.keras',compile=False);path=export(model,folder);evaluate(path,args.run)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','evaluate']);ap.add_argument('--source',default='models/v3/resume.keras');ap.add_argument('--run',default='banner_run1');ap.add_argument('--epochs',type=int,default=2);ap.add_argument('--seed',type=int,default=4100);ap.add_argument('--lr',type=float,default=.0001);ap.add_argument('--edges',action='store_true');ap.add_argument('--no-weak',action='store_true');ap.add_argument('--real',action='store_true');args=ap.parse_args()
    if args.mode=='train':train(args)
    else:evaluate(ROOT/args.source,args.run)
