"""Acceptance checks for user examples, not independent model accuracy."""
import json
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter,minimum_filter
from banner_data import ROOT
from chroma import process
out=ROOT/'reports/real_banners';results={}
for name in ['makeup','watch']:
    original=np.asarray(Image.open(ROOT/'datasets/real_banners'/f'{name}.jpg').convert('RGB'));rgb=original.astype(np.float32)/255;image,info=process(rgb);a=image[...,3];mask=a>0
    Image.fromarray(image).save(out/(f'{name}_model_only.png'))
    if name=='makeup':
        assert np.array_equal(image[...,:3],original)
        results[name]={'all_rgb_pixels_unchanged':True,'all_alpha_opaque':bool(np.all(a==255)),'opaque_pixel_fraction':float(np.mean(a==255)),'route':info['route']}
    else:
        edge=(maximum_filter(a<253,size=7))&mask;ex=image[...,1].astype(int)-np.maximum(image[...,0],image[...,2]).astype(int)
        points={'green_corner':(10,10),'strap_opening':(478,153),'blue_card':(400,800),'watch_face':(340,330)}
        values={key:int(a[y,x]) for key,(x,y) in points.items()}
        results[name]={'route':info['route'],'maximum_positive_green_excess_in_boundary':int(max(0,ex[edge].max())),'alpha_at_acceptance_points':values,'note':'Structural/color checks only; no hand-labeled alpha ground truth.'}
    original.tofile('/tmp/chroma_'+name+'.rgba_rgb')
    rgba=np.concatenate([original,np.full((*original.shape[:2],1),255,np.uint8)],-1);rgba.tofile('/tmp/chroma_'+name+'.rgba')
(out/'acceptance.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
