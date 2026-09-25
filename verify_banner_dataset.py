"""Check source splits, label completeness, single-product composition, and text labels."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from banner_data import records,load_sample,patches
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/banners_v1'

def main():
    allrows={s:records(s) for s in ['train','validation','test']}
    ids={s:{r['source_product_id'] for r in rows} for s,rows in allrows.items()};hashes={s:{r['source_sha256'] for r in rows} for s,rows in allrows.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not ids[a]&ids[b];assert not hashes[a]&hashes[b]
    checked=0
    for split,rows in allrows.items():
        for r in rows:
            assert r['product_count']==1
            for key in ['input','rgba','alpha','product_alpha','text_alpha','decoration_alpha','background']:
                with Image.open(ROOT/r[key]) as im:
                    assert im.size==tuple(r['size']);im.verify()
            if checked%17==0:
                x,y=load_sample(r);rgba=np.asarray(Image.open(ROOT/r['rgba']))
                assert np.array_equal(np.uint8(np.round(y[...,0]*255)),rgba[...,3])
                assert np.count_nonzero(y[...,2]>.5)>50
                product=np.asarray(Image.open(ROOT/r['product_alpha']))
                assert product.max()==255
                assert not np.any((product>128)&(y[...,2]>.5)),r['id']
            checked+=1
    weak=records('train',True)[len(allrows['train']):]
    for r in weak:
        x,y=load_sample(r);assert np.isfinite(y).all();assert y[...,1].min()>=0 and y[...,1].max()<=1
        # Weak data remain identified and separated from exact labels.
        assert 'alpha' not in r and 'weak_alpha' in r
    for x,y in list(zip(range(3),patches(size=128,include_weak=True))):
        rgb,labels=y;assert rgb.shape==(128,128,3) and labels.shape==(128,128,3)
    # A contact sheet pairs JPEG input with the exact target on checkerboard.
    selected=[]
    for split in ['train','validation','test']:
        rs=allrows[split]
        for kind in ['solid_green','green_shade','floor_gradient']:
            selected.append(next(r for r in rs if r['background_kind']==kind and r['layout']=='product_below_offer'))
            selected.append(next(r for r in rs if r['background_kind']==kind and r['layout']!='product_below_offer'))
    tiles=[]
    for r in selected:
        x,y=load_sample(r);fg=np.asarray(Image.open(ROOT/r['rgba']),np.float32)/255;h,w=x.shape[:2];yy,xx=np.mgrid[:h,:w];bg=(.65+.2*((xx//16+yy//16)%2))[...,None]
        target=fg[...,:3]*fg[...,3:]+bg*(1-fg[...,3:]);pair=Image.fromarray(np.uint8(np.clip(np.concatenate([x,target],1),0,1)*255));pair.thumbnail((600,230));tile=Image.new('RGB',(600,260),'#eee');tile.paste(pair,((600-pair.width)//2,20));d=ImageDraw.Draw(tile);d.text((5,3),f"{r['split']} | {r['background_kind']} | {r['headline']}",fill='black');tiles.append(tile)
    sheet=Image.new('RGB',(1200,((len(tiles)+1)//2)*260),'white')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*600,(i//2)*260))
    sheet.save(OUT/'labeled_preview.jpg',quality=90)
    gen=[json.loads(l) for l in (ROOT/'datasets/banners_v1/gemini_manifest.jsonl').read_text().splitlines()]
    selected=[next(r for r in gen if r['split']==s) for s in ['train_weak','solid_visual_test','gradient_visual_test']]
    sheet=Image.new('RGB',(640,3*350),'white');d=ImageDraw.Draw(sheet)
    for i,r in enumerate(selected):sheet.paste(Image.open(ROOT/r['input']),(0,i*350+25));d.text((5,i*350+5),r['split'],fill='black')
    sheet.save(OUT/'gemini_preview.jpg',quality=90)
    result={'labeled_files_checked':checked,'source_id_and_hash_split_leakage':False,'product_text_overlap_in_sampled_rows':False,'exact_alpha_matches_rgba':'passed','weak_records_checked':len(weak),'loader_shapes':'passed','new_api_calls':0}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
