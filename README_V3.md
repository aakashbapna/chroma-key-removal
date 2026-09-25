# V3: larger refinement model and Gemini product data

This experiment expands V2 with a multi-scale U-Net and keeps its pretrained chroma branch frozen. The complete model has **2,010,258 parameters and 22 convolution layers**. Its exported float16-weight TFLite file is **4,043,076 bytes (4.04 MB)**, below the 10 MB limit. It has 31.9× the parameter count of V2. Detailed quality results are recorded in `reports/v3/`.

## Use

From this project directory:

```sh
.venv/bin/python remove_v3.py banner.jpg transparent.png
.venv/bin/python serve.py
```

Open [the V3 browser demo](http://localhost:8766/web/?model=v3). Model: `models/v3/product_chroma.tflite`. Input is float32 RGB [0,1], shape `[1,256,256,3]`; output is float32 alpha `[1,256,256,1]`. Use a 64-pixel halo and retain the central 128×128 region. Tile starts must align to multiples of eight; `remove_v3.py` and the browser implement this. The original V1 and V2 remain available.

Only model storage is constrained by 10 MB. The browser runtime download and working memory are additional, and inference speed must be measured separately. The demo uses the existing pinned TensorFlow.js/TFLite WASM runtime.

## Generated images

The replacement Gemini credential successfully generated product imagery with `gemini-3.1-flash-lite-image`. Exactly 100 requests were attempted/reserved: **98 images were saved** and two requests were interrupted with unknown outcomes while the background prompt was improved. They were not retried, keeping the cap intact. No credential is stored in project files.

All 98 original images are retained in `datasets/generated_v3/`; `manifest.json` contains the exact prompts, model, source paths and usage metadata. The first prompt sometimes generated a pale floor or backdrop. A stricter digital-cutout-on-green prompt greatly reduced that behavior.

- **95 solid-green examples:** 76 available for weak training and 19 for visual testing, split by product subject before augmentation.
- **3 gradient/white-floor hard cases:** IDs 0, 9 and 15. Retained as requested and excluded from weak training. Two share a product subject with an accepted training view; these test a changed background, not unseen-subject recognition.
- Confidently identified connected exterior green is normalized to exact RGB(0,255,0) in `prepared/`. Raw originals are unchanged. Near-edge mixed pixels are retained; no exact product mask is claimed.

Estimated cost for the 98 completed requests is approximately **$3.35**, using returned token counts and the documented model rates. The two interrupted requests may incur additional charges. This is an estimate, not a billing statement. See `reports/v3/generation_usage.json` and [Google's pricing](https://ai.google.dev/gemini-api/docs/pricing).

The actual saved image count is 98. Prepared copies, crops, contact sheets and PNG removal results are derivatives, not additional image-generation requests.

## Training

The frozen V2 output is fed into a trainable U-Net with RGB. Encoder widths 32/64/128 and bottleneck width 256 provide wider spatial context. Decoder skip connections preserve fine detail. A zero-initialized correction head starts from V2's prediction and learns an alpha adjustment.

Initial supervised refinement uses the existing source-product split (157 train / 12 validation / 23 test), 1,570 product composites and 256 procedural examples at 96×96. When generation finished, training transitioned from the best checkpoint to a 128×128 mixed-data stage: 785 product composites, 256 procedural examples, and four weak patches from each of 76 generated training images. This second stage ran four epochs.

Generated labels supervise only eroded high-confidence foreground/background pixels, at confidence 0.1. Edges and ambiguous colors are ignored. They are approximate labels and never supply quantitative test ground truth. Checkpoint selection uses validation loss and preserves the starting checkpoint if fine-tuning fails to improve it.

`reports/v3/training.json` records completed initial epochs and the transition; `finetune.json` records the final pass and the exact generated image IDs used. The checkpoint before generated-data fine-tuning is retained as `models/v3/pre_generated.keras`.

## Test sets and limitations

1. The same 69 held-out product composites used for V2, with original source alpha.
2. The same 32 procedural text/shape examples.
3. The earlier transparent generated product sheet, one independent source with known alpha after green compositing.
4. The 19 new Gemini solid-green test images, evaluated visually only.
5. The three original Gemini gradient/white-floor images, evaluated visually only.
6. A controlled 23-product gradient-floor benchmark with known alpha: green above, fading to a white floor, then JPEG compression. This adds measurable evidence for the requested hard cases.

The new Gemini images have no independent alpha masks, so no numerical alpha-accuracy claim is made for them. The generated transparent sheet is only one independent source. Downloaded masks are source-provided, not manually corrected. Genuine green foreground remains ambiguous. The PNG foreground-color recovery assumes a pure green backdrop, which is a limitation on floor-gradient inputs even when alpha improves.

Detailed experiment decisions: `reports/v3/EXPERIMENT.md`.

## Measured quality

On the 69 held-out product composites, V3 reduced alpha MAE from 0.00754 to **0.00604**, a **19.9% reduction versus V2**. Soft-edge error also fell from 0.05124 to **0.03991**. These are controlled composites from held-out downloaded products, not accuracy scores for the new Gemini inputs.

| Method | Alpha MAE ↓ | Soft-edge MAE ↓ | Foreground IoU ↑ | RGB error on gray ↓ |
|---|---:|---:|---:|---:|
| V3 | 0.00604 | 0.03991 | 0.94910 | 0.00462 |
| V2 | 0.00754 | 0.05124 | 0.94284 | 0.00503 |
| Color key | 0.01097 | 0.04874 | 0.95201 | 0.00382 |

Color keying still has the best IoU and average RGB reconstruction error here. Green foreground discoloration remains visible. On procedural examples, alpha error improves (0.01716 versus V2's 0.01890) but IoU declines slightly (0.8551 versus 0.8601). The product-bootstrap 95% interval for V2 minus V3 alpha MAE is [0.00118, 0.00185]; this describes this dataset only.

**Floor-gradient failure:** on the controlled 23-product green-to-white-floor test, V3 alpha MAE is **0.30447**, versus V2 **0.29981** and color key **0.28864**. None of these methods reliably removes the pale floor. V3 is slightly worse on this test. The actual Gemini floor examples visibly retain much of the white backdrop. Merely increasing model size did not solve the change in background type; training here targets green chroma, and the color recovery also assumes green.

See `reports/v3/product_comparison.png`, `gemini_visual_comparison.png`, `gradient_visual_comparison.png`, and `controlled_gradient_comparison.png`. All gradient originals were retained and tested as requested.

Export verification passed: full-frame Keras versus tiled TFLite maximum alpha difference **0.000586**, tiny/awkward dimensions passed, and pure green becomes fully transparent. The actual exported TFLite loaded in the browser and produced a 384×384 transparent PNG.

## Speed and generated-source check

Warmed standalone native CPU inference for a 256×256 input (8 runs, two interpreter threads, including tiling): V3 median **857 ms**, V2 **92 ms**. V3 is approximately **9.4× slower** here. This is not browser latency; browser loading and PNG output were smoke-tested separately. The larger model trades compute for the measured improvement and is not presented as a real-time model.

On the one earlier generated product sheet with preserved alpha, averaged over JPEG qualities 60/80/95, V3 alpha MAE is **0.00674**, versus V2 **0.00703** and color key **0.01547**. Only this preserved-alpha source permits numerical testing; these figures do not describe the 98 new Gemini images.

Native CLI output exactly matches its evaluated PNG. Verification details are in `reports/v3/verification.json`; timing is in `reports/v3/benchmark.json`.

## Reproduce

Use the existing Python 3.10 environment and `requirements.txt`.

```sh
# Fresh initial stage (original stage transitioned early; consult its report):
.venv/bin/python train_v3.py --epochs 5
# Final mixed-data stage:
.venv/bin/python train_v3.py --resume --include-generated --size 128 --count 785 --epochs 4
.venv/bin/python verify_v3.py
.venv/bin/python evaluate_v3.py
.venv/bin/python benchmark_v3.py
```

To prepare the already saved Gemini images again, run `prepare_generated_v3.py`. `generate_v3.py` requires `GEMINI_API_KEY` in the local environment and skips every attempted ID, including uncertain attempts; the completed 100-request ledger prevents additional generation in this batch. Do not erase that ledger to bypass the cap.
