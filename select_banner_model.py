"""Select only by validation loss and publish compact checkpoints for inference/resume."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import json,shutil,hashlib
import tensorflow as tf
from banner_data import ROOT,records
runs=[name for name in ['banner_run1','banner_run2','banner_real'] if (ROOT/'reports/banner_runs'/name/'metrics.json').exists()];reports={name:json.loads((ROOT/'reports/banner_runs'/name/'training.json').read_text()) for name in runs}
selected=min(runs,key=lambda name:reports[name]['best_validation_loss'])
folder=ROOT/'models/banner_latest';folder.mkdir(parents=True,exist_ok=True)
for name in ['product_chroma.tflite','product_chroma.json']:shutil.copy2(ROOT/'models'/selected/name,folder/name)
model=tf.keras.models.load_model(ROOT/'models'/selected/'best.keras',compile=False);model.save(folder/'resume.keras')
# Original compact baseline is preserved; no local optimizer checkpoint required.
meta={'selected_run':selected,'reason':'lowest fixed validation loss, not test metrics','validation_losses':{name:reports[name]['best_validation_loss'] for name in runs},'model_bytes':(folder/'product_chroma.tflite').stat().st_size,'sha256':hashlib.sha256((folder/'product_chroma.tflite').read_bytes()).hexdigest()}
(folder/'selection.json').write_text(json.dumps(meta,indent=2))
row=next(r for r in records('test') if r['background_kind']=='solid_green');shutil.copy2(ROOT/row['input'],ROOT/'examples/banner_input.jpg')
print(json.dumps(meta,indent=2))
