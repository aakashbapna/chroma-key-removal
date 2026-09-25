"""Generate a bounded Gemini product set. Credential is read only from the environment."""
import argparse,base64,io,json,os,time
from pathlib import Path
import requests
from PIL import Image
ROOT=Path(__file__).resolve().parent
SUBJECTS=['white running shoe with woven mesh and red laces','black over-ear headphones with thin cable','amber perfume bottle with gold cap','red leather handbag with curved handles','stainless steel wristwatch and metal bracelet','white ceramic mug with a large open handle','navy backpack with straps and mesh pockets','black camera with textured grip','silver laptop open at a three-quarter angle','blue glass skincare bottle with dropper','beige knitted sweater folded neatly','black sunglasses with transparent gray lenses','red electric kettle with chrome spout','wooden dining chair with thin legs','cream plush teddy bear with fuzzy fur','gold necklace with a delicate chain','purple athletic shoe with white laces','black desk lamp with thin articulated arms','orange cordless drill','white wireless earbuds and open case','clear drinking glass with water','green shampoo bottle with white label','yellow handbag with a gold chain','dark green potted houseplant in a white pot']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=24);ap.add_argument('--model',default='gemini-3.1-flash-lite-image');args=ap.parse_args()
    if not 1<=args.count<=min(100,len(SUBJECTS)):raise SystemExit('Count must be 1–24; absolute safety cap is 100.')
    key=os.environ.get('GEMINI_API_KEY')
    if not key:raise SystemExit('Set GEMINI_API_KEY in your local environment.')
    root=ROOT/'datasets/generated';root.mkdir(parents=True,exist_ok=True)
    path=root/'manifest.json';rows=json.loads(path.read_text()) if path.exists() else []
    ledger=root/'requests.json';attempts=json.loads(ledger.read_text()) if ledger.exists() else []
    for idx,subject in enumerate(SUBJECTS[:args.count]):
        if any(r['id']==idx for r in rows):continue
        if len(attempts)>=100:raise SystemExit('100-request cap reached; no further requests made.')
        prompt=f'Use case: product-mockup. Create exactly ONE image: photorealistic ecommerce studio product photograph of {subject}. Isolated centered product, entirely visible, occupying 65 percent of the frame. Background must be flat uniform pure chroma green RGB(0,255,0), including holes and empty space around the object. No floor, no background gradient, no drop shadow, no watermark, no extra objects. Preserve realistic material texture and fine edges. Square composition.'
        split='test_visual_only' if idx>=18 else 'train_weak'
        attempts.append({'id':idx,'model':args.model,'time':time.time()});ledger.write_text(json.dumps(attempts,indent=2))
        # No automatic retries: an ambiguous timeout could otherwise incur duplicate charges.
        try:
            r=requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{args.model}:generateContent',headers={'x-goog-api-key':key},json={'contents':[{'parts':[{'text':prompt}]}],'generationConfig':{'responseModalities':['IMAGE']}},timeout=180)
            if not r.ok:
                err=r.json().get('error',{});print('Generation stopped:',r.status_code,err.get('status'),str(err.get('message','')).replace(key,'[REDACTED]')[:250],flush=True);break
            j=r.json();images=[]
            for c in j.get('candidates',[]):
                for p in c.get('content',{}).get('parts',[]):
                    if p.get('thought'):continue
                    inline=p.get('inlineData',p.get('inline_data',{}))
                    if inline.get('mimeType',inline.get('mime_type','')).startswith('image/'):images.append(inline['data'])
            if not images:print('No image returned; stopping.',flush=True);break
            im=Image.open(io.BytesIO(base64.b64decode(images[0]))).convert('RGB');outfile=root/f'{idx:03d}.png';im.save(outfile)
            rows.append({'id':idx,'subject':subject,'prompt':prompt,'split':split,'path':str(outfile.relative_to(ROOT)),'model':args.model,'size':im.size,'usage':j.get('usageMetadata',{}),'returned_images':len(images),'alpha_source':'none; green-key pseudo-labels allowed only for weak training'})
            path.write_text(json.dumps(rows,indent=2));print('Generated',idx+1,'of',args.count,subject,flush=True)
        except requests.RequestException as e:print('Generation stopped after network failure:',type(e).__name__,flush=True);break
    print('Saved',len(rows),'images; total requests',len(attempts),flush=True)
if __name__=='__main__':main()
