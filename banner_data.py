"""Streaming training loader for single-product promotional banners.

Y channels: alpha, confidence, text-alpha. Adapt alpha_loss to use channels 0/1;
channel 2 is available for text-specific validation or extra loss weighting.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent

def records(split='train',include_weak=False):
    folder=ROOT/'datasets/banners_v1'
    rows=[json.loads(line) for line in (folder/'manifest.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['split']==split]
    if include_weak:
        if split!='train':raise ValueError('Weak pseudo-labels cannot enter validation/test.')
        rows += [r for r in map(json.loads,(folder/'gemini_manifest.jsonl').read_text().splitlines()) if r['split']=='train_weak']
    return rows

def load_sample(row):
    x=np.asarray(Image.open(ROOT/row['input']).convert('RGB'),np.float32)/255
    alpha=np.asarray(Image.open(ROOT/row.get('alpha',row.get('weak_alpha'))).convert('L'),np.float32)/255
    confidence=np.asarray(Image.open(ROOT/row['confidence']).convert('L'),np.float32)/255 if 'confidence' in row else np.ones_like(alpha)
    text=np.asarray(Image.open(ROOT/row['text_alpha']).convert('L'),np.float32)/255
    return x,np.stack([alpha,confidence,text],-1)

def patches(split='train',size=256,seed=2028,include_weak=False):
    """One shuffled crop per banner per iterator; half are centered on offer text."""
    rows=records(split,include_weak);rng=np.random.default_rng(seed)
    if split=='train':rng.shuffle(rows)
    for row in rows:
        x,y=load_sample(row);h,w=x.shape[:2]
        if h<size or w<size:
            dh,dw=max(0,size-h),max(0,size-w)
            x=np.pad(x,((0,dh),(0,dw),(0,0)),mode='edge')
            y=np.pad(y,((0,dh),(0,dw),(0,0)),mode='constant')
            # Padded labels have zero confidence; they never contribute to loss.
            h,w=x.shape[:2]
        if rng.random()<.5 and np.any(y[...,2]>.5):
            coords=np.argwhere(y[...,2]>.5);cy,cx=coords[int(rng.integers(len(coords)))];top=int(np.clip(cy-size//2,0,h-size));left=int(np.clip(cx-size//2,0,w-size))
        else:top=int(rng.integers(0,h-size+1));left=int(rng.integers(0,w-size+1))
        yield x[top:top+size,left:left+size],y[top:top+size,left:left+size]

def tensorflow_dataset(split='train',size=256,batch_size=8,seed=2028,include_weak=False):
    """Stream one finite dataset pass; vary seed across epochs for fresh crops."""
    import tensorflow as tf
    spec=(tf.TensorSpec((size,size,3),tf.float32),tf.TensorSpec((size,size,3),tf.float32))
    return tf.data.Dataset.from_generator(lambda:patches(split,size,seed,include_weak),output_signature=spec).batch(batch_size).prefetch(1)
