"""Check export accuracy and tile boundaries against full-frame Keras inference."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from data import sample
from remove import Matting,rgba

model=tf.keras.models.load_model('models/best.keras',compile=False)
small=Matting()
x,_,_=sample(400000,257)
padded=np.pad(x,((4,4),(4,4),(0,0)),mode='edge')
a=model(padded[None],training=False).numpy()[0,4:-4,4:-4,0]
b=small.alpha(x)
error=float(np.max(np.abs(a-b)))
assert error<.003,error
for h,w in [(1,1),(7,241),(240,121)]:
    image=np.zeros((h,w,3),np.float32);image[...,1]=1
    output=rgba(image,small.alpha(image))
    assert output.shape==(h,w,4)
    assert output[...,3].max()==0
Image.fromarray(rgba(x,b)).save('examples/large_result.png')
Path('reports/verification.json').write_text(json.dumps({'max_keras_tflite_alpha_difference':error,'tile_seams_checked_on':'257x257','tiny_and_non_multiple_dimensions':'passed','pure_green_fully_transparent':'passed'},indent=2))
print('Export accuracy, tiling, dimensions, and green transparency checks passed.')
