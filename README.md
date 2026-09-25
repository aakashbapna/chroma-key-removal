# Green-background removal for product banners

A **4.04 MB TFLite model** plus conservative chroma cleanup for banners containing a product, offer text and decorative artwork. Python and browser processing stay local.

## Start here

```sh
.venv/bin/python remove.py input.jpg transparent.png
.venv/bin/python serve.py
```

Open **http://localhost:8766/web/** for the browser demo. `remove.py` is the single supported removal command. Use `--mode neural` to inspect the raw neural route; default `--mode auto`:

1. Preserves banners without a green border intact. Lavender backgrounds and green decorations inside non-green banners are not removed.
2. Uses local foreground/background color projection and boundary despill for uniform green backdrops, including internal cutouts.
3. Uses the selected neural model for nonuniform green backgrounds, with 256-pixel tiles, 64-pixel halo and 128-pixel stride.

The uniform-green cleanup is an explicit image-processing step, not a learned-model improvement. Python uses Euclidean nearest colors; the browser uses grid-nearest colors. Both pass the supplied-image acceptance checks, but their boundary pixels need not be identical.

## Results and real examples

- [Complete DeepLab comparison](reports/deeplab_comparison/REPORT.md) — all model versions, precision, recall, IoU, alpha/text/edge errors, timing, assumptions and visual comparisons.
- [Standalone HTML report](reports/deeplab_comparison/REPORT.html).
- [Training-run history](reports/banner_runs/RESULTS.md).
- [Watch PNG](reports/real_banners/watch_clean.png) — green exterior and strap opening removed; card and lettering retained.
- [Makeup PNG](reports/real_banners/makeup_preserved.png) — lavender, products, lettering and both cubes preserved; decoded RGB unchanged.

The current model is selected by validation loss and stored in `models/banner_latest/`. Its metadata and selection evidence are beside the TFLite file. The follow-up trained on 192 crops from the supplied images alongside existing data. The watch uses weak estimated masks; these supplied originals are **training/acceptance examples, not independent accuracy tests**.

Main test scores use 144 banners and 144 synthetic edge cases from 18 held-out product assets, fitted to 256 pixels. Native-resolution checks are separate. Reports compare raw neural outputs with the supplied DeepLab file, not the complete automatic pipeline. High background-area pixel accuracy can hide poor foreground recall.

## Installation

Python 3.10 is used locally:

```sh
uv venv --python python3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

The browser loads pinned TensorFlow.js/TFLite WASM assets from jsDelivr; their download size is separate from the model. Serve via `serve.py` for the required cross-origin isolation headers. No API key is needed for inference.

## Training, data and evaluation

[Data provenance, splits and labeling](docs/DATA.md). Raw downloaded/generated/user images and optimizer checkpoints are ignored by Git. Original V3 and subsequent TFLite checkpoints remain as named experimental baselines; only `banner_latest` is the deployed default. Compact `resume.keras` checkpoints support additional training.

```sh
# Existing local dataset
.venv/bin/python build_banner_dataset.py
.venv/bin/python build_edge_cases.py
.venv/bin/python prepare_real_banners.py
.venv/bin/python banner_experiments.py train --source models/banner_latest/resume.keras --run my_run --epochs 1 --edges --real --lr 0.00002

# Independent comparison with your own supplied DeepLab file
.venv/bin/python compare_deeplab_banners.py --deeplab /path/to/deeplabv3.tflite
.venv/bin/python compare_deeplab_native.py --deeplab /path/to/deeplabv3.tflite
.venv/bin/python write_deeplab_report.py
```

On a fresh checkout, use `download_products.py`, `build_banner_dataset.py --skip-generated`, and `build_edge_cases.py`, then train with `--no-weak` and without `--real` unless those local images are available. This produces a different training set from the recorded experiments. Fonts in the builder currently target macOS; adapt `FONTS` for another OS. Upstream asset changes can alter hashes and splits.

`generate_v3.py` is the only image-generation command retained. It requires `GEMINI_API_KEY` in the environment and is never invoked automatically. The earlier 100-request generation budget has been consumed; no further generation was used for the current work.

## Verification and limits

Run `verify_banner_dataset.py`, `verify_banner_edges.py`, and `verify_real_banners.py` for local data/acceptance checks. `verify_banner_model.py` checks export fidelity and native inference. The report records the evaluated model hashes.

Green foreground matching a green key cannot always be recovered from RGB. Nonuniform backgrounds, pale floors, translucent products, and unfamiliar thin details remain difficult. The automatic border test is conservative; it may preserve an image whose green backdrop barely touches the border. Pure-green uncompositing on the neural fallback approximates foreground color and can leave color errors on gradients.

Superseded commands and duplicate top-level READMEs were removed. Historical evidence is under `archive/`; the prior code remains recoverable from Git commit `5ec7875`. The main README and current comparison report are the sources of truth.
