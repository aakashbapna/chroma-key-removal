import json,time
from pathlib import Path
import numpy as np
from remove_v3 import LargeMatting
from remove_v2 import ProductMatting
from product_data import composite,manifest
x,_,_=composite(manifest('test')[0],333333,256);report={'input':[256,256],'warm_runs':8,'threads':2,'scope':'Standalone native CPU, includes tiling, not browser latency'}
for name,runner in [('v3',LargeMatting()),('v2',ProductMatting())]:
    runner.alpha(x);times=[]
    for _ in range(8):
        start=time.perf_counter();runner.alpha(x);times.append((time.perf_counter()-start)*1000)
    report[name]={'median_ms':float(np.median(times))}
Path('reports/v3/benchmark.json').write_text(json.dumps(report,indent=2));print(report)
