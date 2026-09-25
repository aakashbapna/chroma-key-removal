"""Check export fidelity and native-size results of the selected banner model."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image,ImageDraw,ImageOps
from banner_data import ROOT,records,load_sample
from banner_experiments import edge_records,metrics
from remove_v3 import LargeMatting
from remove import rgba
OUT=ROOT/'reports/banner_runs/native';OUT.mkdir(parents=True,exist_ok=True)
def checker(im):
    im=im.convert('RGBA');w,h=im.size;yy,xx=np.indices((h,w));g=np.where((xx//16+yy//16)%2,210,245).astype('uint8');bg=Image.fromarray(np.stack([g]*3,-1)).convert('RGBA');return Image.alpha_composite(bg,im).convert('RGB')
def panel(images,labels):
    out=Image.new('RGB',(400*len(images),260),'white');d=ImageDraw.Draw(out)
    for k,(im,label) in enumerate(zip(images,labels)):
        im=ImageOps.contain(im,(400,232));out.paste(im,(400*k+(400-im.width)//2,28+(232-im.height)//2));d.text((400*k+8,8),label,fill='black')
    return out
models={'baseline':LargeMatting(),'latest':LargeMatting(ROOT/'models/banner_latest/product_chroma.tflite')}
selected=[next(r for r in records('test') if r['background_kind']==kind) for kind in ['solid_green','green_shade','floor_gradient']]
selected += [next(r for r in edge_records('test') if r['edge_case']==kind) for kind in ['tiny_text_lines','translucency','green_spill','heavy_jpeg','green_foreground','pale_floor','off_green','soft_shadow']]
results=[];panels=[]
for row in selected:
    x,y=load_sample(row);gt=y[...,0];images=[Image.fromarray(np.uint8(x*255)),checker(Image.open(ROOT/row['rgba']))];item={'id':row['id'],'group':row['background_kind']}
    for name,model in models.items():
        a=model.alpha(x);assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
        pred=a>=.5;truth=gt>=.5;tp=int(np.sum(pred&truth));fp=int(np.sum(pred&~truth));fn=int(np.sum(~pred&truth));tn=int(np.sum(~pred&~truth));item[name]={'alpha_mae':float(np.abs(a-gt).mean()),'pixel_accuracy':(tp+tn)/gt.size,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'iou':tp/max(tp+fp+fn,1)}
        im=Image.fromarray(rgba(x,a));images.append(checker(im))
        if name=='latest':im.save(OUT/f'{row["id"]}.png')
    p=panel(images,['Input','Target transparency','Original V3','Retrained']);p.save(OUT/f'{row["id"]}.jpg');panels.append(p);results.append(item);print(row['id'],item,flush=True)
contact=Image.new('RGB',(1600,780),'white')
for k,p in enumerate(panels[:3]):contact.paste(p,(0,k*260))
contact.save(OUT/'comparison.jpg')
model=tf.keras.models.load_model(ROOT/'models/banner_latest/resume.keras',compile=False);lite=models['latest'].m;i=lite.get_input_details()[0]['index'];o=lite.get_output_details()[0]['index'];x=np.asarray(Image.open(ROOT/selected[0]['input']).convert('RGB').resize((256,256)),np.float32)[None]/255;lite.set_tensor(i,x);lite.invoke();diff=float(np.max(np.abs(model(x,training=False).numpy()-lite.get_tensor(o))));assert diff<.02,diff
checks=[]
for h,w in [(1,1),(7,9),(127,129),(261,133)]:
    x=np.zeros((h,w,3),np.float32);x[...,1]=1;a=models['latest'].alpha(x);assert a.shape==(h,w) and np.isfinite(a).all();checks.append({'size':[w,h],'pure_green_max_alpha':float(a.max())})
(OUT/'verification.json').write_text(json.dumps({'scope':'11 diagnostic examples; not an aggregate accuracy estimate. Some examples share product assets. Native-size tiling used by CLI/browser.','samples':results,'keras_tflite_max_abs_difference':diff,'size_checks':checks},indent=2))
