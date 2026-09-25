"""Check generated edge-case labels, source splits, and extreme foreground cases."""
import json
from PIL import Image
import numpy as np
from banner_data import ROOT,records,load_sample
from build_edge_cases import KINDS
rows=list(map(json.loads,(ROOT/'datasets/banner_edges/manifest.jsonl').read_text().splitlines()))
source_splits={r['source_sha256']:s for s in ['train','validation','test'] for r in records(s)}
seen=set()
for r in rows:
    assert r['id'] not in seen;seen.add(r['id']);assert source_splits[r['source_sha256']]==r['split']
    x,y=load_sample(r);a=np.asarray(Image.open(ROOT/r['rgba']).getchannel('A'))
    assert x.shape[:2]==a.shape==y.shape[:2]
    assert np.max(np.abs(y[...,0]-a/255))<1e-6
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert np.any(a==0) and np.any(a>0)
    if r['edge_case']=='translucency':
        pm=np.asarray(Image.open(ROOT/r['product_alpha']))>250
        assert np.any(pm) and float(y[...,0][pm].mean())<.5
assert len(rows)==1240
assert all({r['edge_case'] for r in rows if r['split']==s}==set(KINDS) for s in ['train','validation','test'])
result={'images_checked':len(rows),'cases':KINDS,'labels_match_rgba':True,'source_split_preserved':True,'translucent_product_check':True}
(ROOT/'reports/banners_v1/edge_verification.json').write_text(json.dumps(result,indent=2));print(result)
