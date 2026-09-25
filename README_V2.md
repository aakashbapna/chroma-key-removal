# Larger ecommerce chroma model

Trained locally on 192 downloaded product cutouts, with source products split before augmentation. The new model has 63,089 parameters (16.5× v1) and a 23×23 receptive field. The browser-ready float16 TFLite file is **131,964 bytes (132 KB / 129 KiB)**, about 21× smaller than the supplied DeepLab file. It improves alpha estimation over v1 on both held-out ecommerce composites and one generated product sheet, at a significant speed cost.

## Use

From `/Users/aakash/projects/green-screen-removal`:

```sh
.venv/bin/python remove_v2.py banner.jpg transparent.png
.venv/bin/python serve.py
```

Open [the larger-model demo](http://localhost:8766/web/?model=v2). The `Try example` button processes the generated product sheet after green compositing and JPEG compression. The actual TFLite model was verified in the browser producing a 640×640 transparent PNG. The original model remains available at `/web/` without the query parameter.

- Model: `models/v2/product_chroma.tflite`
- Python checkpoint: `models/v2/best.keras`
- Contract: `models/v2/product_chroma.json`
- Input: float32 `[1,160,160,3]`, RGB in [0,1].
- Output: float32 `[1,160,160,1]`, alpha in [0,1].
- Tiling: 12-pixel halo, retain the central 136×136 pixels, replicate image boundaries. Do not reuse the v1 4-pixel halo.
- Exported operators: Conv2D, Dequantize, Logistic; no custom/Flex operations.
- The Python and browser wrappers reconstruct foreground colors by inverting the green composite, with the same stabilization and endpoint snapping as v1.

## Held-out product results

23 source products × 3 variants = 69 composites at 256×256. Products were excluded from training and validation. The target is each downloaded image's original alpha, preserved before adding green and JPEG compression. It is not manually corrected alpha. See `datasets/products/manifest.json` for URLs, hashes, categories and split assignments.

| Method | Alpha MAE ↓ | Soft-edge MAE ↓ | Foreground IoU ↑ | RGB error on gray ↓ |
|---|---:|---:|---:|---:|
| Larger v2 | **0.00754** | 0.05124 | 0.94284 | 0.00503 |
| Small v1 | 0.01605 | 0.10504 | 0.93826 | 0.00776 |
| Color key | 0.01097 | **0.04874** | **0.95201** | **0.00382** |
| DeepLab | 0.07975 | 0.31486 | 0.25098 | 0.02119 |

V2 reduces mean alpha error by **53.0% versus v1** and **90.5% versus DeepLab** on this benchmark. The source-product bootstrap 95% interval for v1 minus v2 alpha MAE is [0.00591, 0.01114]. This interval reflects only this small source collection, not arbitrary production banners.

The color-key baseline still has better average soft-edge error, IoU and reconstructed RGB error on these ecommerce composites. V2 is not universally superior to color keying. Genuine green foreground products visibly lose green or become tinted in some outputs. These failures are included in the benchmark and contact sheet.

The original procedural-text/shape task also improved: alpha MAE 0.01890 versus v1's 0.02375 on 32 untouched synthetic samples.

DeepLab uses [-1,1] normalization and combines all nonzero semantic classes as foreground, retaining the documented v1 assumptions. It is a semantic-segmentation model, not a dedicated alpha-matting model.

## Generated-image test and Gemini status

Gemini authentication succeeded, but generation failed with HTTP 429 `RESOURCE_EXHAUSTED` for both `gemini-3.1-flash-lite-image` and `gemini-2.5-flash-image`. Two requests, zero returned Gemini images, no automatic retries.

As a disclosed fallback, the built-in image generator produced **one image containing six products** on actual transparency. The file is `datasets/generated/builtin_product_sheet.png`; the exact prompt and provenance are in `datasets/generated/builtin_manifest.json`. It was never used in training. The sheet was composited onto RGB(0,255,0) and JPEG-compressed at qualities 60, 80 and 95. The retained alpha is the exact compositing target, including any imperfections in the generated cutout.

| Method | Generated-sheet alpha MAE ↓ | Soft-edge MAE ↓ | Foreground IoU ↑ |
|---|---:|---:|---:|
| Larger v2 | **0.00703** | **0.06764** | **0.99553** |
| Small v1 | 0.02450 | 0.09615 | 0.99342 |
| Color key | 0.01547 | 0.11280 | 0.99182 |

V2's alpha error is 71.3% lower than v1 here. This is **one independent generated source**, so it is a useful smoke test, not evidence of broad performance across AI-generated banners. Total successfully generated images: **1**, below the requested cap of 100.

For a future Gemini batch after quota is enabled, set `GEMINI_API_KEY` locally, then run:

```sh
.venv/bin/python generate_products.py --count 24
```

The script records prompts and usage without saving credentials. It stops on API errors and caps attempted requests at 100. Its current prompt list supports up to 24 images. Generated green-background images have no independent alpha labels; optional training uses only low-weight high-confidence pixels, and they are excluded from quantitative alpha tests. This completed run used **zero weak-label generated patches**.

## Speed, checks and training

Warmed native CPU timing, 12 runs at 256×256, two interpreter threads, no concurrent training/evaluation:

| Method | Median inference time |
|---|---:|
| Larger v2 | 87.29 ms |
| Small v1 | 10.65 ms |
| Color key | 0.069 ms |
| DeepLab | 12.80 ms |

V2 is approximately 8.2× slower than v1 in this measurement. Timings include each method's resizing/tiling, exclude PNG saving, and are not browser latency claims. The browser WASM runtime download remains much larger than the model itself. The demo retains the tested pinned CDN runtime from v1; vendor it for offline operation.

Training: 157 source products, 1,884 augmented product composites, 256 procedural composites, 144 validation composites from 12 different products, 24 epochs, batch 16, Adam, seed 2026. Training/export took 755.6 seconds locally on CPU. All quantitative test sources were excluded from fitting and checkpoint selection.

Verification passed for full-frame Keras versus tiled TFLite (maximum alpha difference 0.000558), awkward/tiny dimensions, and completely transparent pure green. Source IDs/hashes are unique across splits. Native CLI output and browser TFLite processing were also checked. Source catalog near-duplicates or category similarities are not ruled out by byte hashing.

## Reproduce

Use the existing `.venv` or install `requirements.txt` with Python 3.10 as described in `README.md`.

```sh
.venv/bin/python download_products.py
.venv/bin/python train_v2.py
.venv/bin/python evaluate_v2.py
.venv/bin/python verify_v2.py
.venv/bin/python evaluate_generated.py
.venv/bin/python benchmark_v2.py
```

The saved source manifest/hashes and downloaded files are the reproducible dataset snapshot; fetching the online catalog again may change it. Generated imagery is preserved locally and is not deterministic from the prompt alone. Run the benchmark separately from other model workloads.

Evidence: `reports/v2/metrics.json`, `generated_metrics.json`, `training.json`, `verification.json`, `benchmark.json`, `data_integrity.json`, `product_comparison.png`, and `generated_comparison.png`. Example input/result PNGs are in `examples/v2/`.

Dataset source: [DummyJSON product documentation](https://dummyjson.com/docs/products). Third-party product images retain their existing rights; source URLs are recorded in the manifest.
