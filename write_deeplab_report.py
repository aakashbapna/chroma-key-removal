"""Generate one portable comparison report (Markdown + standalone HTML)."""
import json,html,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'reports/deeplab_comparison'
r=json.loads((OUT/'metrics.json').read_text());native=json.loads((OUT/'native_metrics.json').read_text());chosen=r['selected_model']
names={'banner_run1':'Banner Run 1','banner_run2':'Banner Run 2','banner_real':'Real-image fine-tune','deeplab_hard':'DeepLab hard','deeplab_soft':'DeepLab soft'};names[chosen]+=' (current)';methods=list(r['results']);md=[];blocks=[]
def para(s):md.extend([s,'']);blocks.append('<p>'+html.escape(s)+'</p>')
def heading(s,level=2):md.extend(['#'*level+' '+s,'']);blocks.append(f'<h{level}>'+html.escape(s)+f'</h{level}>')
def table(headers,rows):
 md.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]+['']);blocks.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>')
def figure(path,caption):
 file=ROOT/path;rel=__import__('os').path.relpath(file,OUT);md.extend([f'![{caption}]({rel})','',caption,'']);data=base64.b64encode(file.read_bytes()).decode();mime='image/png' if file.suffix=='.png' else 'image/jpeg';blocks.append('<figure><img src="data:'+mime+';base64,'+data+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
def pct(v):return f'{100*v:.2f}%'
heading('Green-background removal: current model vs supplied DeepLab V3',1)
para('25 September 2026 · Frozen, paired tests · 144 held-out offer banners + 144 synthetic stress images · 18 held-out product assets.')
a=r['results'][chosen]['suites']['banners'];d=r['results']['deeplab_soft']['suites']['banners']
heading('Findings')
para(f'The current neural model achieves {pct(a["iou"])} foreground IoU versus {pct(d["iou"])} for DeepLab’s soft-mask conversion on held-out banners. Its alpha error is {a["alpha_mae"]:.5f} versus {d["alpha_mae"]:.5f}, a reduction of {(1-a["alpha_mae"]/d["alpha_mae"])*100:.1f}%. DeepLab is smaller and roughly {a["median_inference_ms"]/d["median_inference_ms"]:.1f}× faster at similar input size with four CPU threads.')
para('These tables compare raw TFLite model outputs. The application now uses model-predicted alpha for every image, with unchanged input RGB and no color rules or despill. DeepLab has semantic class outputs rather than a transparency channel; both hard and soft foreground conversions are evaluated.')
for suite,title in [('banners','Held-out offer banners'),('edge_cases','Held-out synthetic edge cases')]:
 heading(title);table(['Method','Accuracy','Precision','Recall','F1','IoU','Alpha MAE ↓'],[[names[n]]+[pct(v['suites'][suite][k]) for k in ['pixel_accuracy','precision','recall','f1','iou']]+[f'{v["suites"][suite]["alpha_mae"]:.5f}'] for n,v in r['results'].items()])
para('Binary foreground uses alpha ≥0.5. Confusion metrics aggregate pixel counts; alpha errors average per-image errors on a 0–1 scale. Large background areas inflate pixel accuracy, so foreground recall, IoU and alpha error are more informative than accuracy alone.')
heading('Offer text and soft boundaries')
table(['Method','Suite','Text-core recall ↑','Text alpha MAE ↓','Soft-edge MAE ↓'],[[names[n],s,pct(v['suites'][s]['text_core_recall']),f'{v["suites"][s]["text_alpha_mae"]:.5f}',f'{v["suites"][s]["soft_edge_mae"]:.5f}'] for n,v in r['results'].items() for s in ['banners','edge_cases']])
para('Text-core recall measures opaque glyph pixels retained as foreground. It is not OCR accuracy or a guarantee of legibility. Soft-edge error covers target alpha from 0.02 to 0.98. DeepLab removes most offer lettering under the tested class-to-foreground mapping.')
heading('Background and stress breakdown')
table(['Case']+[names[n]+' MAE ↓' for n in methods],[[g]+[f'{r["results"][n]["groups"][g]["alpha_mae"]:.5f}' for n in methods] for g in r['results'][chosen]['groups']])
heading('Size and speed')
table(['Model','File size','Input','Output','Median / p95 CPU inference'],[[names[n],f'{r["models"][n if n.startswith("banner") else "deeplab"]["bytes"]/1e6:.3f} MB','256×256 RGB' if n.startswith('banner') else '257×257 RGB','1 alpha channel' if n.startswith('banner') else '21 class scores',f'{r["results"][n]["suites"]["banners"]["median_inference_ms"]:.2f} / {r["results"][n]["suites"]["banners"]["p95_inference_ms"]:.2f} ms'] for n in methods if n!='deeplab_hard'])
para(r['protocol']['timing']);para('Runtime: TensorFlow '+r['protocol']['tensorflow']+' on '+r['protocol']['system']+'. Sizes exclude browser runtime downloads. No cloud inference or paid image generation was used for this comparison.')
heading('Native-resolution diagnostic subset')
para(native['scope'])
table(['Method','Accuracy','Precision','Recall','IoU','Alpha MAE ↓','Median processing'],[[names[n],pct(v['overall']['pixel_accuracy']),pct(v['overall']['precision']),pct(v['overall']['recall']),pct(v['overall']['iou']),f'{v["overall"]["alpha_mae"]:.5f}',f'{v["overall"]["median_inference_ms"]:.1f} ms'] for n,v in native['methods'].items()])
table(['Background']+[names[n]+' MAE ↓' for n in native['methods']],[[g]+[f'{v["groups"][g]["alpha_mae"]:.5f}' for v in native['methods'].values()] for g in ['solid_green','green_shade','floor_gradient']])
para('These are 18 reused held-out sources, not 18 additional independent products. Native tiled inference does much more work than one global DeepLab mask. Use the fixed-size table for the cleaner model-speed comparison.')
heading('Visual comparison on held-out images')
para('Columns are input, target, Run 1, Run 2, real-image fine-tune, DeepLab hard and DeepLab soft. Predictions use the same observed JPEG colors with each alpha mask, without despill, to isolate masking. Targets use original foreground colors. Examples are the first test item per case, not handpicked by results.')
for file,title in [('p0008_00.jpg','Solid green and offer text'),('p0008_07.jpg','Floor gradient'),('e0008_green_foreground.jpg','Green foreground stress')]:figure('reports/deeplab_comparison/'+file,title)
heading('User-supplied banners: training and acceptance cases')
para('The makeup image is a lavender banner, not a green-backdrop input. The user explicitly requested that it remain intact, including both decorative cubes. The application preserves decoded RGB but predicts opacity with the model; complete preservation is not guaranteed. The watch should retain its blue card, logo, offer text and watch while removing the exterior green and green opening in the strap.')
para('Training added 64 exact all-opaque makeup crops and 128 weak watch crops. Watch targets are estimated from chroma colors, with confidence 0.15 away from uncertain edges; they are not independent ground truth. The follow-up used one epoch at learning rate 0.00002. These originals are excluded from held-out accuracy scores, and improved test scores cannot be attributed to the real images alone because the run also repeats existing banner/edge training.')
if (ROOT/'reports/real_banners/comparison.jpg').exists():figure('reports/real_banners/comparison.jpg','Real examples: input, model-only alpha, DeepLab soft. These images were used for training; visual acceptance only.')
accept=json.loads((ROOT/'reports/real_banners/acceptance.json').read_text())
para('Current model-only acceptance measurements (8-bit alpha): '+json.dumps(accept)+'. These are training examples, not independent accuracy measurements. Positive boundary green excess exposes remaining spill; no correction is applied.')
heading('Protocol, assumptions and uncertainty')
para(r['protocol']['preprocessing'])
para('Training/validation/test product hashes remain disjoint. Both main test suites share 18 source products; 288 variants are not 288 independent products. Compositing alpha is exact, but source product cutouts were not manually corrected. Neither test suite uses weak Gemini masks or the newly supplied originals as ground truth.')
para('The supplied file has float32 input [1,257,257,3] and float32 output [1,257,257,21]. No class-label or preprocessing documentation accompanied it. Channel 0 is assumed background. Hard decoding takes argmax !=0; soft decoding uses 1−softmax(background). Semantic probability is not physical opacity. This is a comparison of the provided artifact under these disclosed assumptions, not all DeepLab variants.')
para('Input normalization was selected independently for hard and soft decoding by validation foreground IoU on 72 validation banners plus 72 validation stress images. Both selected [-1,1]. Threshold stays 0.5. No test-based tuning was performed.')
table(['Normalization','Decoder','Validation IoU','Validation MAE ↓'],[[norm,decoder,pct(v['iou']),f'{v["alpha_mae"]:.5f}'] for norm,decoders in r['normalization_validation'].items() for decoder,v in decoders.items()])
table(['Paired comparison','Mean alpha-error advantage','95% product-cluster interval'],[[k,f'{v["mean_alpha_mae_difference"]:.5f}',f'[{v["product_cluster_bootstrap_95ci"][0]:.5f}, {v["product_cluster_bootstrap_95ci"][1]:.5f}]'] for k,v in r['paired_alpha_error_differences'].items()])
para('Positive differences favor the current model. Intervals use 5,000 paired bootstrap resamples of 18 product clusters, preserving correlated variants. They do not estimate uncertainty over arbitrary future banners. Matching green foreground/background is inherently ambiguous; unseen gradients and fine translucency remain limitations. The alpha-only model does not reconstruct foreground RGB, so residual green fringes remain possible.')
heading('Reproduce and inspect')
para('Run compare_deeplab_banners.py, then compare_deeplab_native.py with --deeplab /path/to/deeplabv3.tflite, then write_deeplab_report.py. Machine-readable evidence is in metrics.json, native_metrics.json and summary.csv. The original DeepLab file is evaluated in place and is not republished. Its SHA-256 is '+r['models']['deeplab']['sha256']+'.')
(OUT/'REPORT.md').write_text('\n'.join(md))
css='body{font:16px/1.6 system-ui;color:#17232f;background:#f4f7fa;margin:0}main{max-width:1200px;margin:28px auto;background:white;padding:40px;border-radius:14px}h1{font-size:34px;line-height:1.2}h2{margin-top:38px;color:#16486a}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px;margin:18px 0}th{background:#e8f1f7;text-align:left}th,td{padding:9px 11px;border-bottom:1px solid #dce3e9;white-space:nowrap}tr:nth-child(even){background:#f8fafc}td{font-variant-numeric:tabular-nums}figure{margin:24px 0}img{width:100%;height:auto;border:1px solid #e2e7ec}figcaption{font-size:13px;color:#536476}@media(max-width:700px){main{padding:16px;margin:0}h1{font-size:27px}}@media print{main{padding:0}h2{break-after:avoid}tr,figure{break-inside:avoid}}'
(OUT/'REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Banner transparency vs DeepLab</title><style>'+css+'</style><main>'+''.join(blocks)+'</main></html>')
print('Wrote REPORT.md and standalone REPORT.html')
