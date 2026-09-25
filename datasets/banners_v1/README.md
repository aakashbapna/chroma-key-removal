# Single-product offer banners

Generated with `build_banner_dataset.py` from the existing product cutouts and previously generated Gemini images. No new image API requests were made.

## Exact compositing labels

2,264 JPEG banners: 2,048 training, 72 validation, 144 test. The 155 product source assets are split into 128/9/18 products, with no source ID or SHA-256 overlap across splits. Each banner uses one product asset, one offer headline, supporting text and a CTA. A source product photograph may depict multiple components of that product.

Backgrounds: 1,698 solid green, 283 shaded green and 283 green-to-white floor gradients. Targets remove the entire background, including the floor. Banner sizes are 640×320, 512×512 and 384×512. JPEG quality, chroma subsampling, typography, placement, foreground colors and offer wording vary.

`manifest.jsonl` records the JPEG input, RGBA target, combined alpha, product alpha, text glyph alpha, decoration alpha and original background, plus source and rendering metadata. Labels are exact for the compositing operation; source product cutouts were not manually corrected. Font paths and hashes are recorded, and fonts are not redistributed.

## Generated-image derivatives

`gemini_manifest.jsonl` contains 348 offer banners made from existing Gemini products: 304 weak training examples, 38 solid-background visual tests and 6 floor-gradient visual tests. Training labels use low-confidence product pseudo-labels and exact rendered text labels; uncertain product pixels have zero confidence. These are not ground-truth product masks. Visual tests have no product alpha labels and must not be included in quantitative accuracy scores. Some original gradient failures share a subject with a training view; they are qualitative hard cases, not an independent subject benchmark.

## Loading and verification

Use `banner_data.records`, `banner_data.load_sample` or `banner_data.tensorflow_dataset`. Targets have three channels: alpha, confidence and text alpha. The loader prevents weak examples from entering validation or test. Vary the seed across training epochs for fresh crops. Banner-specific fine-tuning runs are documented in `README_BANNERS.md` and `reports/banner_runs/RESULTS.md`. The original V3 weights are retained unchanged for baseline comparisons.

Run `.venv/bin/python verify_banner_dataset.py` to check files, splits, sampled compositing labels and loader shapes. Reports and previews are in `reports/banners_v1`. Run `.venv/bin/python evaluate_banners.py` for a deterministic smoke benchmark of the existing V3 and V2 models on 18 held-out products (six per background type). This small benchmark is not a claim of general superiority over DeepLab.
