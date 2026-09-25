"""Reject wrong backdrops; normalize confidently identified green exterior only."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_propagation
ROOT=Path(__file__).resolve().parent

def main():
    folder=ROOT/'datasets/generated_v3';rows=json.loads((folder/'manifest.json').read_text());out=folder/'prepared';out.mkdir(exist_ok=True);results=[]
    for row in rows:
        im=Image.open(ROOT/row['path']).convert('RGB');x=np.asarray(im,np.float32)/255
        border=np.concatenate([x[:8].reshape(-1,3),x[-8:].reshape(-1,3),x[:,:8].reshape(-1,3),x[:,-8:].reshape(-1,3)])
        key=np.median(border,axis=0);strong=(border[:,1]-np.maximum(border[:,0],border[:,2])>.55)&(border[:,1]>.65)
        fraction=float(strong.mean());spread=float(np.quantile(np.max(np.abs(border-key),axis=-1),.95))
        accepted=fraction>.985 and spread<.15
        result={**row,'background_border_green_fraction':fraction,'background_border_spread':spread,'estimated_key':key.tolist(),'background_qa_pass':accepted,'original_path':row['path']}
        if accepted:
            # No product mask is claimed here. Near-product mixed edge colors remain unchanged.
            candidates=(np.max(np.abs(x-key),axis=-1)<.1)&((x[...,1]-np.maximum(x[...,0],x[...,2]))>.65)
            seeds=np.zeros(candidates.shape,bool);seeds[[0,-1],:]=True;seeds[:,[0,-1]]=True
            exterior=binary_propagation(seeds&candidates,mask=candidates)
            x[exterior]=[0,1,0];target=out/f"{row['id']:03d}.png";Image.fromarray(np.uint8(np.round(x*255))).save(target)
            result['path']=str(target.relative_to(ROOT));result['normalized_exterior_fraction']=float(exterior.mean())
        results.append(result)
    (folder/'prepared_manifest.json').write_text(json.dumps(results,indent=2))
    accepted=[r for r in results if r['background_qa_pass']]
    report={'raw_images':len(rows),'accepted':len(accepted),'rejected_ids':[r['id'] for r in results if not r['background_qa_pass']],'accepted_by_split':{s:sum(r['split']==s for r in accepted) for s in ['train_weak','test_visual_only']},'normalization':'Only connected high-confidence exterior set to exact RGB(0,255,0); no ground-truth product alpha inferred.'}
    (ROOT/'reports/v3/generated_quality.json').write_text(json.dumps(report,indent=2))
    chosen=accepted[:36];sheet=Image.new('RGB',(900,((len(chosen)+5)//6)*170),'white');d=ImageDraw.Draw(sheet)
    for i,row in enumerate(chosen):
        im=Image.open(ROOT/row['path']);im.thumbnail((150,150));xx=i%6*150;yy=i//6*170;sheet.paste(im,(xx,yy));d.text((xx+3,yy+151),f"{row['id']} {row['split']}",fill='black')
    sheet.save(ROOT/'reports/v3/generated_contact_sheet.png');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
