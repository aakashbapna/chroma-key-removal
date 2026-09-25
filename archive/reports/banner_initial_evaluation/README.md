# Existing model test on offer banners

This is a smoke benchmark, not retraining: V3 and V2 weights are unchanged. Eighteen held-out product assets were selected deterministically, six per background type. Different background groups use different products, so differences between groups are not a paired background ablation.

| Background | V3 alpha MAE | V2 alpha MAE | V3 text alpha MAE |
| --- | ---: | ---: | ---: |
| Solid green | 0.00341 | 0.00465 | 0.03777 |
| Shaded green | 0.09250 | 0.08214 | 0.05960 |
| Floor gradient | 0.30140 | 0.29672 | 0.03877 |

Alpha is on a 0–1 scale; lower error is better. Text metrics include antialiased glyph pixels. V3 reduces overall alpha error by 26.7% versus V2 on the six solid-green cases. It performs worse on both background variation groups. Both models leave substantial floor residue. Reconstruction assumes pure-green background, which also creates color artifacts when the true background is shaded or pale. This experiment does not establish superiority over DeepLab.

V3 native-resolution inference took roughly 3.0–3.3 seconds per banner on this machine, versus 0.32–0.35 seconds for V2. These are Python CPU timings, not browser benchmarks. V3 TFLite size is 4,043,076 bytes.

`metrics.json` contains individual results and timing. `comparison.jpg` shows one example of each background, with target transparency. `generated_comparison.jpg` shows qualitative generated-image tests with no ground-truth product alpha. Transparent outputs are saved beside the reports.
