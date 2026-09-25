"""Publish per-run metrics without rounding away soft-alpha errors."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
runs=[]
for name in ['baseline','banner_run1','banner_run2','banner_real']:
    path=ROOT/'reports/banner_runs'/name/'metrics.json'
    if path.exists():runs.append(json.loads(path.read_text()))
lines=['# Banner retraining results','','All runs use the same 144 held-out banners and 144 synthetic edge cases from 18 held-out product assets. Each input is resized to fit 256×256, aspect ratio preserved; padded pixels are excluded. Binary metrics threshold prediction and target alpha at 0.5. Accuracy, precision, recall, F1 and IoU aggregate pixel counts; alpha errors average per-image errors. These are standardized-resolution model tests, not native-resolution browser accuracy or independent real-world validation.','','| Run | Test set | Accuracy | Precision | Recall | F1 | IoU | Alpha MAE ↓ | Text MAE ↓ | Soft-edge MAE ↓ |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in runs:
    for suite,m in r['suites'].items():
        lines.append('| '+r['run']+' | '+suite+' | '+' | '.join(f'{m[k]*100:.2f}%' for k in ['pixel_accuracy','precision','recall','f1','iou'])+' | '+' | '.join(f'{m[k]:.5f}' for k in ['alpha_mae','text_alpha_mae','soft_edge_mae'])+' |')
for r in runs:
    lines+=['',f'## {r["run"]}: detailed groups','','| Group | Precision | Recall | IoU | Alpha MAE ↓ |','|---|---:|---:|---:|---:|']
    for group,m in r['groups'].items():lines.append(f'| {group} | {m["precision"]*100:.2f}% | {m["recall"]*100:.2f}% | {m["iou"]*100:.2f}% | {m["alpha_mae"]:.5f} |')
lines+=['','## Training and limitations','','Run 1 uses 2,048 exact-label banners plus 304 low-confidence generated-image derivatives. Run 2 adds 1,024 edge-case examples. The real-image follow-up uses one additional epoch, 64 fully opaque makeup crops and 128 low-confidence watch crops, retaining the original independent test split. Each run uses two epochs, a 2,010,258-parameter V3 refinement network, 128×128 training inputs (a mix of global views and native crops), text-aware boundary loss and a frozen V2 branch. Each epoch changes the sampling seed. Best weights are selected using the same fixed validation set (72 banners and 72 edge cases), including the option to retain the starting checkpoint. Test results do not select checkpoints.','','No new paid image-generation calls were made. The generated product regions have approximate labels with reduced confidence, while composited banners have exact compositing alpha. Source product alpha was not manually corrected. Stress cases are synthetic; RGB cannot unambiguously recover green foreground that matches the background. The RGB uncompositing step still assumes a pure-green background, so successful alpha removal does not guarantee faithful colors on pale floors or translucent objects.','','Native-resolution diagnostics and observed regressions are documented in [the current DeepLab report](../deeplab_comparison/REPORT.md).', '', 'Per-image metrics, confusion counts, full model sizes and evaluation timings are in each run’s `metrics.json`; optimization history and validation selection are in `training.json`. Historical results elsewhere in the repository use other evaluation protocols and should not be directly compared numerically.']
(ROOT/'reports/banner_runs/RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Updated reports/banner_runs/RESULTS.md')
