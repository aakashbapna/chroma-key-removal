"""Untouched synthetic test seeds; report limitations and baselines explicitly."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import tensorflow as tf
from data import sample
from remove import Matting,rgba

def metrics(y,p):
    edge=(y>.02)&(y<.98); a=y>.5; b=p>.5
    return {'alpha_mae':float(np.abs(y-p).mean()),'soft_edge_mae':float(np.abs(y-p)[edge].mean()) if edge.any() else 0.,'foreground_iou':float((a&b).sum()/max(1,(a|b).sum()))}

def key(rgb):
    # A deliberately useful continuous chroma-difference baseline, not exact RGB equality.
    return np.clip(1-(rgb[...,1]-np.maximum(rgb[...,0],rgb[...,2])),0,1)

class DeepLab:
    def __init__(self,path):
        self.m=tf.lite.Interpreter(model_path=path,num_threads=2);self.m.allocate_tensors()
        self.i=self.m.get_input_details()[0];self.o=self.m.get_output_details()[0]
    def alpha(self,rgb,normalization):
        h,w=self.i['shape'][1:3]
        x=np.asarray(Image.fromarray(np.uint8(rgb*255)).resize((int(w),int(h)),Image.Resampling.BILINEAR),np.float32)
        if normalization=='minus_one_one': x=x/127.5-1
        elif normalization=='zero_one': x=x/255
        self.m.set_tensor(self.i['index'],x[None].astype(self.i['dtype']));self.m.invoke()
        logits=self.m.get_tensor(self.o['index'])[0]
        a=(np.argmax(logits,axis=-1)!=0).astype(np.float32)
        return np.asarray(Image.fromarray(a).resize((rgb.shape[1],rgb.shape[0]),Image.Resampling.BILINEAR))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deeplab',default='/Users/aakash/Downloads/deeplabv3.tflite');ap.add_argument('--count',type=int,default=64);args=ap.parse_args()
    model=Matting();dl=DeepLab(args.deeplab)
    # No supplied metadata: select among common float normalizations on validation only.
    norms={}
    for norm in ['minus_one_one','zero_one','raw']:
        norms[norm]=float(np.mean([metrics(a[...,0],dl.alpha(x,norm))['alpha_mae'] for x,a,_ in [sample(100000+i) for i in range(8)]]))
    norm=min(norms,key=norms.get)
    records={k:[] for k in ['model','color_key','deeplab']};times={k:[] for k in records}
    contact=[]
    for i in range(args.count):
        x,a,f=sample(200000+i,128);a=a[...,0]
        preds={}
        for name,fn in [('model',model.alpha),('color_key',key),('deeplab',lambda x:dl.alpha(x,norm))]:
            t=time.perf_counter();p=fn(x);times[name].append((time.perf_counter()-t)*1000);records[name].append(metrics(a,p));preds[name]=p
        if i<6:
            Image.fromarray(np.uint8(x*255)).save(f'examples/input_{i}.png')
            Image.fromarray(rgba(x,preds['model'])).save(f'examples/result_{i}.png')
            Image.fromarray(np.uint8(a*255)).save(f'examples/alpha_truth_{i}.png')
            yy,xx=np.mgrid[:128,:128];checker=(.65+.2*((xx//12+yy//12)%2))[...,None]*np.ones(3)
            panels=[x,f*a[...,None]+checker*(1-a[...,None])]
            for name in ['model','color_key','deeplab']:
                out=rgba(x,preds[name])/255;panels.append(out[...,:3]*out[...,3:]+checker*(1-out[...,3:]))
            contact.append(np.concatenate(panels,axis=1))
    stress=[]
    for i in range(16):
        x,a,_=sample(300000+i,128,stress=True);stress.append(metrics(a[...,0],model.alpha(x)))
    report={'test_count':args.count,'test_seeds':[200000,200000+args.count-1],'scope':'Procedural shapes/text, not real AI banners. Green foreground stress reported separately.','deeplab_assumptions':'Class 0 background, all other classes foreground; input normalization chosen on 8 validation samples; no metadata supplied.','deeplab_validation_normalization_mae':norms,'deeplab_normalization':norm,'models':{k:{**{m:float(np.mean([r[m] for r in v])) for m in v[0]},'median_native_ms_128x128':float(np.median(times[k]))} for k,v in records.items()},'green_foreground_stress':{m:float(np.mean([r[m] for r in stress])) for m in stress[0]},'model_bytes':Path('models/chroma_matte.tflite').stat().st_size,'deeplab_bytes':Path(args.deeplab).stat().st_size}
    Path('reports/metrics.json').write_text(json.dumps(report,indent=2))
    sheet=Image.new('RGB',(640,6*128+30),'white');sheet.paste(Image.fromarray(np.uint8(np.concatenate(contact)*255)),(0,30));d=ImageDraw.Draw(sheet)
    for j,name in enumerate(['JPEG input','Ground truth','Trained model','Color key','DeepLab']):d.text((j*128+5,8),name,fill='black')
    sheet.save('reports/comparison.png');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
