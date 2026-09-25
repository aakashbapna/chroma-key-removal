"""Run after training/evaluation, without concurrent model workloads."""
import json,time
from pathlib import Path
import numpy as np
from product_data import composite,manifest
from remove import Matting
from remove_v2 import ProductMatting
from evaluate import DeepLab,key
x,_,_=composite(manifest('test')[0],333333,256)
v2=ProductMatting();v1=Matting();dl=DeepLab('/Users/aakash/Downloads/deeplabv3.tflite')
methods={'larger_v2':v2.alpha,'small_v1':v1.alpha,'color_key':key,'deeplab':lambda x:dl.alpha(x,'minus_one_one')}
report={'image_size':[256,256],'runs':12,'threads_per_interpreter':2,'scope':'Native CPU end-to-end alpha inference including preprocessing/tiling, excluding PNG saving. Not browser timing. Warmed up; no concurrent training or evaluation.'}
for name,fn in methods.items():
    fn(x);ms=[]
    for i in range(12):
        t=time.perf_counter();fn(x);ms.append((time.perf_counter()-t)*1000)
    report[name]={'median_ms':float(np.median(ms)),'p95_ms':float(np.quantile(ms,.95))}
Path('reports/v2/benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
