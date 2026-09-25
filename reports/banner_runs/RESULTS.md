# Banner retraining results

All runs use the same 144 held-out banners and 144 synthetic edge cases from 18 held-out product assets. Each input is resized to fit 256×256, aspect ratio preserved; padded pixels are excluded. Binary metrics threshold prediction and target alpha at 0.5. Accuracy, precision, recall, F1 and IoU aggregate pixel counts; alpha errors average per-image errors. These are standardized-resolution model tests, not native-resolution browser accuracy or independent real-world validation.

| Run | Test set | Accuracy | Precision | Recall | F1 | IoU | Alpha MAE ↓ | Text MAE ↓ | Soft-edge MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | banners | 96.01% | 75.56% | 98.91% | 85.67% | 74.93% | 0.05309 | 0.06758 | 0.08125 |
| baseline | edge_cases | 94.99% | 71.70% | 89.21% | 79.50% | 65.97% | 0.07629 | 0.07327 | 0.10654 |
| banner_run1 | banners | 99.61% | 98.18% | 98.61% | 98.39% | 96.83% | 0.00716 | 0.05268 | 0.05499 |
| banner_run1 | edge_cases | 98.23% | 94.48% | 88.98% | 91.65% | 84.58% | 0.03099 | 0.06185 | 0.08424 |
| banner_run2 | banners | 99.60% | 97.88% | 98.83% | 98.35% | 96.75% | 0.00862 | 0.05406 | 0.05724 |
| banner_run2 | edge_cases | 98.18% | 90.58% | 92.98% | 91.76% | 84.78% | 0.02513 | 0.06096 | 0.07938 |
| banner_real | banners | 99.55% | 97.51% | 98.80% | 98.15% | 96.36% | 0.00657 | 0.05555 | 0.05931 |
| banner_real | edge_cases | 98.99% | 94.50% | 96.36% | 95.42% | 91.25% | 0.01505 | 0.06214 | 0.07450 |

## baseline: detailed groups

| Group | Precision | Recall | IoU | Alpha MAE ↓ |
|---|---:|---:|---:|---:|
| floor_gradient | 29.20% | 99.09% | 29.12% | 0.29477 |
| green_foreground | 96.02% | 31.42% | 31.02% | 0.06888 |
| green_shade | 95.18% | 99.39% | 94.63% | 0.09434 |
| green_spill | 99.27% | 94.23% | 93.58% | 0.00808 |
| heavy_jpeg | 97.66% | 98.60% | 96.32% | 0.00665 |
| off_green | 90.16% | 99.88% | 90.06% | 0.22958 |
| pale_floor | 30.35% | 99.24% | 30.28% | 0.27323 |
| soft_shadow | 98.36% | 98.98% | 97.37% | 0.00712 |
| solid_green | 98.24% | 98.80% | 97.08% | 0.00593 |
| tiny_text_lines | 95.99% | 98.89% | 94.97% | 0.00883 |
| translucency | 75.69% | 98.25% | 74.68% | 0.00791 |

## banner_run1: detailed groups

| Group | Precision | Recall | IoU | Alpha MAE ↓ |
|---|---:|---:|---:|---:|
| floor_gradient | 97.58% | 97.00% | 94.72% | 0.01007 |
| green_foreground | 96.35% | 31.30% | 30.93% | 0.07139 |
| green_shade | 97.43% | 99.12% | 96.60% | 0.01474 |
| green_spill | 99.45% | 94.17% | 93.69% | 0.00736 |
| heavy_jpeg | 97.78% | 98.62% | 96.46% | 0.00638 |
| off_green | 91.97% | 99.86% | 91.85% | 0.12938 |
| pale_floor | 97.47% | 97.83% | 95.40% | 0.00953 |
| soft_shadow | 98.42% | 98.95% | 97.40% | 0.00704 |
| solid_green | 98.40% | 98.79% | 97.23% | 0.00541 |
| tiny_text_lines | 96.54% | 98.80% | 95.42% | 0.00826 |
| translucency | 63.60% | 98.37% | 62.93% | 0.00861 |

## banner_run2: detailed groups

| Group | Precision | Recall | IoU | Alpha MAE ↓ |
|---|---:|---:|---:|---:|
| floor_gradient | 95.09% | 98.70% | 93.92% | 0.01220 |
| green_foreground | 97.99% | 59.92% | 59.20% | 0.04809 |
| green_shade | 97.71% | 99.04% | 96.80% | 0.02543 |
| green_spill | 99.43% | 94.14% | 93.64% | 0.00748 |
| heavy_jpeg | 97.77% | 98.65% | 96.49% | 0.00613 |
| off_green | 68.76% | 99.61% | 68.57% | 0.10491 |
| pale_floor | 95.69% | 98.76% | 94.56% | 0.01155 |
| soft_shadow | 98.13% | 99.01% | 97.18% | 0.00729 |
| solid_green | 98.39% | 98.81% | 97.24% | 0.00522 |
| tiny_text_lines | 96.35% | 98.81% | 95.24% | 0.00806 |
| translucency | 72.76% | 98.55% | 71.98% | 0.00757 |

## banner_real: detailed groups

| Group | Precision | Recall | IoU | Alpha MAE ↓ |
|---|---:|---:|---:|---:|
| floor_gradient | 97.36% | 98.60% | 96.03% | 0.00846 |
| green_foreground | 97.94% | 84.63% | 83.14% | 0.03027 |
| green_shade | 98.06% | 98.79% | 96.90% | 0.00894 |
| green_spill | 98.53% | 94.30% | 92.99% | 0.00769 |
| heavy_jpeg | 96.58% | 98.64% | 95.32% | 0.00689 |
| off_green | 91.00% | 99.50% | 90.58% | 0.04095 |
| pale_floor | 97.35% | 98.78% | 96.20% | 0.00807 |
| soft_shadow | 96.90% | 99.06% | 96.01% | 0.00772 |
| solid_green | 97.44% | 98.83% | 96.32% | 0.00586 |
| tiny_text_lines | 96.00% | 98.85% | 94.94% | 0.00938 |
| translucency | 68.07% | 98.55% | 67.39% | 0.00939 |

## Training and limitations

Run 1 uses 2,048 exact-label banners plus 304 low-confidence generated-image derivatives. Run 2 adds 1,024 edge-case examples. The real-image follow-up uses one additional epoch, 64 fully opaque makeup crops and 128 low-confidence watch crops, retaining the original independent test split. Each run uses two epochs, a 2,010,258-parameter V3 refinement network, 128×128 training inputs (a mix of global views and native crops), text-aware boundary loss and a frozen V2 branch. Each epoch changes the sampling seed. Best weights are selected using the same fixed validation set (72 banners and 72 edge cases), including the option to retain the starting checkpoint. Test results do not select checkpoints.

No new paid image-generation calls were made. The generated product regions have approximate labels with reduced confidence, while composited banners have exact compositing alpha. Source product alpha was not manually corrected. Stress cases are synthetic; RGB cannot unambiguously recover green foreground that matches the background. The RGB uncompositing step still assumes a pure-green background, so successful alpha removal does not guarantee faithful colors on pale floors or translucent objects.

Native-resolution diagnostics and observed regressions are documented in [the current DeepLab report](../deeplab_comparison/REPORT.md).

Per-image metrics, confusion counts, full model sizes and evaluation timings are in each run’s `metrics.json`; optimization history and validation selection are in `training.json`. Historical results elsewhere in the repository use other evaluation protocols and should not be directly compared numerically.
