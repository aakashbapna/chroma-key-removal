"""Guard the inference contract: every input uses predicted alpha unchanged."""
import numpy as np
from chroma import process
class Probe:
    calls=0
    def alpha(self,rgb):
        self.calls+=1
        return np.linspace(0,1,rgb.shape[0]*rgb.shape[1],dtype=np.float32).reshape(rgb.shape[:2])
for color in [[0,255,0],[210,190,245],[20,20,20]]:
    source=np.broadcast_to(np.array(color,np.uint8),(11,17,3)).copy()
    model=Probe();output,info=process(source.astype(np.float32)/255,model)
    assert model.calls==1
    assert np.array_equal(output[...,:3],source)
    expected=np.round(np.linspace(0,1,187,dtype=np.float32).reshape(11,17)*255).astype(np.uint8)
    assert np.array_equal(output[...,3],expected)
print('PASS: all backgrounds use model alpha; RGB unchanged; no opacity snapping.')
