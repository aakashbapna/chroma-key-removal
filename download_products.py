"""Download one alpha-bearing image per product; split by product before augmentation."""
import hashlib,json,io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image
ROOT=Path(__file__).resolve().parent

def download(p):
    url=p['images'][0];path=ROOT/'datasets/products'/f"{p['id']:04d}.png"
    try:
        if not path.exists():
            r=requests.get(url,timeout=45);r.raise_for_status()
            im=Image.open(io.BytesIO(r.content));im.load()
            if im.mode!='RGBA' or im.getchannel('A').getextrema()!=(0,255):return None
            box=im.getchannel('A').getbbox();im=im.crop(box);im.thumbnail((640,640));im.save(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        # Identical source bytes always go to the same split.
        bucket=int(digest[:8],16)%10
        split='test' if bucket==0 else 'validation' if bucket==1 else 'train'
        return {'id':p['id'],'title':p['title'],'category':p['category'],'url':url,'path':str(path.relative_to(ROOT)),'sha256':digest,'split':split,'alpha_source':'original source alpha; not manually verified','source':'DummyJSON product catalog'}
    except Exception as e:print(f"Skipped product {p['id']}: {type(e).__name__}",flush=True);return None

if __name__=='__main__':
    (ROOT/'datasets/products').mkdir(parents=True,exist_ok=True)
    catalog=requests.get('https://dummyjson.com/products?limit=0',timeout=40);catalog.raise_for_status()
    products=catalog.json()['products']
    # Spread across categories; one photo per product avoids alternate-view leakage.
    with ThreadPoolExecutor(max_workers=8) as pool: rows=[x for x in pool.map(download,products) if x]
    # Remove byte-identical duplicates entirely.
    unique={r['sha256']:r for r in rows};rows=list(unique.values())
    (ROOT/'datasets/products/manifest.json').write_text(json.dumps(rows,indent=2))
    print('Downloaded',len(rows),'products:',{s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},flush=True)
