# Single-product banner retraining

The current experiment fine-tunes the 2,010,258-parameter V3 network for single-product ecommerce banners with offer text. Model exports remain below 10 MB. Full per-run measurements are in [the results report](reports/banner_runs/RESULTS.md).

## Selected result

Run 2 has the lowest fixed validation loss (0.02627 versus Run 1’s 0.03270). Its 4,043,076-byte TFLite export is published as `models/banner_latest/product_chroma.tflite`.

| Held-out suite | Pixel accuracy | Precision | Recall | F1 | IoU | Alpha MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Banners (144) | 99.60% | 97.88% | 98.83% | 98.35% | 96.75% | 0.00862 |
| Edge cases (144) | 98.18% | 90.58% | 92.98% | 91.76% | 84.78% | 0.02513 |

Run 1 is slightly better on ordinary banner precision and alpha error; Run 2 improves edge-case recall and alpha error. Neither is uniformly better. Green-product recall remains only 59.9%, and off-green backgrounds still cause false foreground. Both checkpoints and complete measurements are retained.

## Use the selected model

```sh
.venv/bin/python remove_banner.py input.jpg transparent.png
.venv/bin/python serve.py
```

Open `http://localhost:8766/web/?model=banner`. Processing runs locally in the browser. This URL uses `models/banner_latest/product_chroma.tflite`, with the same 256-pixel input, 64-pixel halo and 128-pixel stride as V3. The browser runtime download is separate from model size.

## Data and training

- Exact-label banners: 2,048 training, 72 validation, 144 test.
- Reused Gemini derivatives: 304 low-confidence training examples; no new image API requests for these experiments.
- New synthetic stress set: 1,024 training, 72 validation, 144 test, spanning eight edge cases.
- Product sources remain separated across train, validation and test. Test suites share the same 18 held-out product assets and are correlated.
- Run 1: two epochs on exact-label banners plus weak generated derivatives.
- Run 2: two epochs with the additional synthetic stress examples.
- Each epoch uses both global views and native text/boundary crops. The V2 branch remains frozen. Validation loss chooses the checkpoint, including the option to keep the starting weights.

```sh
.venv/bin/python build_banner_dataset.py
.venv/bin/python build_edge_cases.py
.venv/bin/python verify_banner_edges.py
.venv/bin/python banner_experiments.py evaluate --source models/v3/product_chroma.tflite --run baseline
.venv/bin/python banner_experiments.py train --source models/v3/resume.keras --run banner_run1 --epochs 2
.venv/bin/python banner_experiments.py train --source models/banner_run1/best.keras --run banner_run2 --epochs 2 --edges --seed 5100 --lr 0.00005
.venv/bin/python summarize_banner_runs.py
```

The local dataset is not included in Git. To rebuild the exact-label portion from a fresh checkout, run `download_products.py`, then `build_banner_dataset.py --skip-generated`, then `build_edge_cases.py`. Run training with `--no-weak` when generated source images are unavailable. That is a different training set from the reported runs. Existing font paths target macOS; adapt `FONTS` in `build_banner_dataset.py` on other platforms. Recorded asset hashes and source URLs support provenance; a changed upstream catalog can change the regenerated split and metrics.

The compact `models/v3/resume.keras` checkpoint supports retraining from the original V3 weights. `models/banner_latest/resume.keras` supports further training from the selected banner model. Optimizer state and raw data are ignored by Git. Image-generation scripts require an environment variable for credentials and must be invoked explicitly; no key is embedded. No new API calls are needed to use the published model.

## What the scores mean

Binary foreground metrics use alpha ≥0.5. Pixel accuracy is heavily affected by the large background area; use precision, recall, F1 and IoU alongside soft-alpha error. The main benchmark fits images to 256 pixels and excludes padding. Native-resolution diagnostics are reported separately. No claim is made that these synthetic results prove superiority over DeepLab or production-banner performance.

Native diagnostics found remaining floor residue and an off-green regression: see [native checks](reports/banner_runs/native/README.md). Run 1 was better than Run 2 on the sampled native floor. Green foreground, translucent products, heavily compressed text, shadows and pale floors remain difficult. Color recovery still assumes pure-green background, so inspect output colors as well as transparency. Original downloaded cutout masks were not manually corrected.
