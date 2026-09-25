# Green chroma matting

> **Latest: banner retraining** — [training, usage and limitations](README_BANNERS.md) · [accuracy, precision, recall, IoU and alpha error after each run](reports/banner_runs/RESULTS.md).

> **Previous: V3** — [4.04 MB refinement model, 98 Gemini images, and floor-gradient tests](README_V3.md).

> **New: larger ecommerce model.** See [README_V2.md](README_V2.md) for the 132 KB model trained on real product cutouts, held-out comparisons, generated-image test, and usage. The material below documents the original v1 experiment.

A trained 3,833-parameter CNN that predicts soft transparency for JPEG artwork composited over RGB(0,255,0). Includes Python training, an 11,260-byte float16-weight TFLite model, native-resolution PNG conversion, generated examples, and a working browser demo. No paid APIs, external training assets, or cloud GPUs used.

## Use

From this project directory, the installed environment is ready:

```sh
.venv/bin/python remove.py /path/to/banner.jpg /path/to/transparent.png
.venv/bin/python serve.py
```

Open http://localhost:8766/web/?model=v1 for the original model, or http://localhost:8766/web/ for the selected banner model. The demo server binds to localhost. Browser inference and PNG creation were verified in the Codex in-app browser. Files remain local; the demo downloads pinned TensorFlow.js 3.21.0 and TFLite alpha.9 runtime assets from jsDelivr. Those runtime assets are much larger than the model; 11 KB describes model weights/graph only. Deployment requires serving the model and page over HTTP(S); use the isolation headers in `serve.py`. The older alpha runtime is pinned because alpha.10 failed to initialize in the tested browser. Other browsers/devices have not been tested. For offline use, vendor the pinned JS/WASM assets and update URLs.

Fresh installation uses Python 3.10 (TensorFlow 2.16.2 does not support Python 3.14):

```sh
uv venv --python python3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Results and scope

64 held-out procedural test composites, JPEG qualities 55–100, with text, shapes, strokes, gradients, soft and translucent edges. Training/validation/test seeds are disjoint. No real AI-generated banner dataset was supplied, so these results demonstrate synthetic-task performance, not proven superiority on production banners.

| Method | Mean absolute alpha error ↓ | Soft-edge error ↓ | Foreground IoU ↑ |
|---|---:|---:|---:|
| Trained TFLite model | 0.02425 | 0.10926 | 0.8510 |
| Continuous color-key baseline | 0.03137 | 0.13588 | 0.8198 |
| Supplied DeepLab v3 | 0.15090 | 0.44805 | 0.0561 |

Model size: 11,260 bytes versus 2,779,264 bytes for DeepLab (247× smaller). Native CPU median per 128×128 test image: 4.40 ms model, 12.20 ms DeepLab, 0.028 ms color key. These include wrapper preprocessing and differ in network input resolution; they are not browser latency benchmarks. The inexpensive color baseline is competitive and much faster.

DeepLab output has 21 classes. Evaluation assumes class zero is background and combines all other classes as foreground. With no model documentation supplied, normalization was selected from [-1,1], [0,1], and raw RGB using eight validation samples; [-1,1] won. DeepLab is a semantic-segmentation baseline and generally discards procedural artwork. We trained from scratch for this chroma task; the supplied model's weights are not used.

See `reports/metrics.json`, `reports/comparison.png`, and `reports/training.json` for evidence. Comparison columns are input, ground truth, trained model, color baseline, DeepLab.

## Model contract

- Input: float32 `[1,128,128,3]`, RGB values in [0,1], no mean subtraction.
- Output: float32 `[1,128,128,1]`, alpha in [0,1]; zero transparent, one opaque.
- Layers: 3×3 convolutions with 12/16/12 ReLU channels, then 1×1 sigmoid output. Receptive field 7×7.
- Float16 storage, float32 I/O; built-in TFLite operators only, no Flex/custom operators.
- Large images use 128×128 tiles with four-pixel context, keeping each central 120×120 region. Replicate padding at image edges preserves dimensions without globally resizing text.
- Exporters must apply the same tiling and RGB normalization. `web/index.html` contains a full JS implementation.
- PNG postprocessing snaps alpha below .01 to zero and above .99 to one, then estimates foreground RGB by inverting `C = alpha*F + (1-alpha)*green`. Low-alpha division is stabilized at .03. The model itself predicts alpha only.

## Reproduce

```sh
.venv/bin/python train.py --epochs 12 --count 768
.venv/bin/python evaluate.py --deeplab /Users/aakash/Downloads/deeplabv3.tflite
.venv/bin/python verify.py
```

768 generated training images and 96 validation images, 96×96, batch 16, Adam, seed 73. Weighted alpha loss emphasizes soft edges. Best checkpoint is selected by validation loss. Successful training/export took about 18 seconds locally, excluding environment setup and data generation. Data are generated in memory; stored examples use untouched test seeds. Keras weights are retained in `models/best.keras` for further training. Results may vary slightly across hardware/runtime versions.

`verify.py` checks TFLite against full-frame Keras inference (including tile boundaries), tiny/non-multiple image sizes, and fully transparent pure green. Maximum observed alpha difference was 0.000337. Browser smoke test loaded the actual `.tflite`, processed the sample and created a PNG download link.

## Limits and next data

Identical green foreground and background cannot be reliably separated from RGB alone. Training deliberately excludes strongly green-dominant foreground. A separate green-foreground stress set gives IoU 0.0021: genuine green artwork is usually removed. This model is for reserved green chroma backgrounds, not general background removal.

The network has local context only; it does not recognize objects. Heavy JPEG compression, green spill, glass, shadows, bright green/yellow artwork and unfamiliar foreground textures can produce errors. Recovered foreground colors are approximate, especially on translucent edges; alpha metrics do not measure that color error. Synthetic foregrounds are gradients/noise, not photographs. The next meaningful improvement is training and testing on representative real banner/RGBA pairs; hold out whole source designs before generating variants. No claim of production readiness is made without those examples.

Runtime API reference: https://js.tensorflow.org/api_tflite/0.0.1-alpha.9/
Conversion reference: https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
