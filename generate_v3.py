"""Bounded 100-image Gemini run. No credentials are stored; failed requests are not retried."""
import argparse,base64,io,json,os,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
from generate_products import SUBJECTS
ROOT=Path(__file__).resolve().parent
MORE=['red toaster with chrome levers','white ceramic teapot','brown hiking boot with rugged sole','blue tennis racket with fine strings','yellow bicycle helmet with ventilation holes','silver bicycle with thin spokes','black game controller','pink hair dryer','white standing fan with wire grille','silver kitchen whisk','red scissors with open handles','brown woven basket','blue umbrella opened fully','silver fountain pen','black microphone with mesh grille','white charging cable coiled loosely','purple yoga mat rolled up','red insulated water bottle','tan leather belt with metal buckle','black office chair with mesh back','blue suitcase with extended handle','white sneaker viewed from the sole','bronze desk clock with thin hands','red ceramic vase','black tripod with slender legs','orange safety vest with reflective strips']
SUBJECTS100=[(s,v) for s in SUBJECTS+MORE for v in ['front three-quarter view','side view with fine product details visible']]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=100);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--model',default='gemini-3.1-flash-lite-image');args=ap.parse_args()
    if not 1<=args.count<=100:raise SystemExit('Count must be within 1–100.')
    key=os.environ.get('GEMINI_API_KEY')
    if not key:raise SystemExit('Set GEMINI_API_KEY locally.')
    root=ROOT/'datasets/generated_v3';root.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.json';ledger=root/'requests.json'
    rows=json.loads(manifest.read_text()) if manifest.exists() else [];attempts=json.loads(ledger.read_text()) if ledger.exists() else []
    lock=threading.Lock();stop=threading.Event()
    def save(path,obj):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
    def run(idx):
        subject,view=SUBJECTS100[idx]
        prompt=f'Generate ONE flat digital chroma-key test plate. The ENTIRE square canvas is one perfectly uniform bright green color #00FF00, all four corners and the ENTIRE bottom third included. Paste a photorealistic cutout of {subject}, {view}, in the center, floating in the green field. The product is fully visible, realistic material textures, occupies at most 60 percent of canvas height and width. Leave wide uniform pure green margins on ALL four sides. Render no floor, no surface, no horizon, no scene, no gradient, no glow, no white or gray anywhere outside the product, no cast shadows, no text. All gaps and holes in the product reveal exactly the SAME uniform green canvas. This is a digital collage of one cutout on a solid #00FF00 rectangle. Background must stay equally bright green from top to bottom.'
        with lock:
            if stop.is_set() or any(r['id']==idx for r in rows) or any(r['id']==idx for r in attempts):return
            if len(attempts)>=100:stop.set();return
            entry={'id':idx,'model':args.model,'started':time.time(),'status':'pending'};attempts.append(entry);save(ledger,attempts)
        try:
            response=requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{args.model}:generateContent',headers={'x-goog-api-key':key},json={'contents':[{'parts':[{'text':prompt}]}],'generationConfig':{'responseModalities':['IMAGE']}},timeout=180)
            if not response.ok:
                err=response.json().get('error',{});reason=str(err.get('message','')).replace(key,'[REDACTED]')[:700]
                with lock:entry.update(status='failed',http_status=response.status_code,reason=reason);save(ledger,attempts)
                stop.set();print('Generation stopped:',response.status_code,reason,flush=True);return
            j=response.json();images=[]
            for c in j.get('candidates',[]):
                for part in c.get('content',{}).get('parts',[]):
                    inline=part.get('inlineData',{})
                    if not part.get('thought') and inline.get('mimeType','').startswith('image/'):images.append(inline['data'])
            if not images:raise ValueError('No output image')
            im=Image.open(io.BytesIO(base64.b64decode(images[0]))).convert('RGB');outfile=root/f'{idx:03d}.png';im.save(outfile)
            # Both views of each product share a split to avoid subject leakage.
            split='test_visual_only' if (idx//2)%5==0 else 'train_weak'
            row={'id':idx,'subject':subject,'view':view,'prompt':prompt,'split':split,'path':str(outfile.relative_to(ROOT)),'model':args.model,'size':im.size,'usage':j.get('usageMetadata',{}),'returned_images':len(images),'alpha_source':'none; not quantitative ground truth'}
            with lock:rows.append(row);entry.update(status='complete',returned_images=len(images));save(manifest,sorted(rows,key=lambda r:r['id']));save(ledger,attempts)
            print('Saved',idx,subject,flush=True)
            if len(images)>1:stop.set();print('Multiple images returned unexpectedly; stopping to preserve output cap.',flush=True)
        except Exception as e:
            with lock:entry.update(status='uncertain_or_failed',reason=type(e).__name__);save(ledger,attempts)
            stop.set();print('Stopped after',type(e).__name__,'without retry.',flush=True)
    with ThreadPoolExecutor(max_workers=max(1,min(2,args.workers))) as pool:list(pool.map(run,range(args.count)))
    print('Saved images:',len(rows),'total attempted requests:',len(attempts),flush=True)
if __name__=='__main__':main()
