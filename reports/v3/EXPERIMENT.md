# V3 experiment record

The user requested a replacement Gemini credential, up to 100 generated ecommerce product images with solid green chroma backgrounds, and a deeper browser model below 10 MB. They subsequently asked to retain floor-gradient images as useful tests.

## Generation and splits

`generate_v3.py` plans 50 subjects × 2 views. Both views of each subject share the same split: one fifth of subjects are visual-test only, the rest may supply weak training pixels. Request reservation is locked and saved before sending; the request cap is 100, and uncertain requests are never repeated. API credentials are read from the environment and are not saved.

Exactly 100 requests were reserved: 98 returned images were saved; two were interrupted when generation was paused to strengthen the background prompt. Those two outcomes/charges are unknown. They were not retried. All raw originals and prompts are in `datasets/generated_v3/` and its manifest.

The first prompt occasionally produced gradients or white floors. The second prompt explicitly asks for a product cutout pasted onto a flat green digital plate, including a uniform bottom third. Quality checking routed 95 images to the solid-green set and IDs 0, 9, 15 to the user-requested gradient/white-floor hard cases. No original image was discarded. Those hard-case images are excluded from weak training and receive their own visual comparison. Two hard cases share a product subject with an accepted training view, so this is not a subject-independent generalization test; it is a background-condition stress test.

For accepted images only, connected exterior pixels confidently matching the dominant border green are normalized to exact RGB(0,255,0). Mixed edge colors are retained. This is conservative background normalization, not a ground-truth product matte. The raw path, prepared path, normalization fraction and QA results are recorded. A border check is not exhaustive manual segmentation validation.

There are 76 generated training images and 19 generated solid-background visual-test images, plus the three background-deviation tests. No quantitative alpha accuracy is claimed for these Gemini images because the original alpha is unknown. Training uses only eroded high-confidence opaque/background pixels with 0.1 confidence; ambiguous green foreground and transition pixels are ignored.

## Larger architecture

The pretrained V2 branch is frozen. A trainable U-Net receives RGB plus its alpha prediction. Encoder widths are 32/64/128, the bottleneck is 256, and the decoder uses 128/64/32 channels with skip connections. Each block has two 3×3 convolutions. A zero-initialized tanh correction head starts from exactly the V2 alpha. Its correction is added and clipped to [0,1]. The complete network has 22 convolution layers counting the frozen V2 branch.

The export uses float16 weight storage, float32 RGB/alpha I/O, and built-in TFLite operators only. The exported file is asserted below 10,000,000 bytes. Input tiles are 256×256, with a 64-pixel halo and 128-pixel retained center. Tile origins and halo align to the encoder's factor-of-eight downsampling. Verification compares this tiled inference against a full-frame Keras run with matching padding/alignment.

## Training and evaluation

Source products retain the existing split: 157 training, 12 validation and 23 test products. Initial supervised refinement runs on 1,570 real-product composites plus 256 procedural images at 96×96. Once the generated batch was ready, the best checkpoint was retained and training transitioned to larger-context 128×128 composites and weak generated patches. Completed initial epochs and the transition are recorded in `training.json`; the mixed-data stage is recorded in `finetune.json`. Checkpoint selection uses validation only and preserves the initial checkpoint if fine-tuning does not improve validation loss.

The main benchmark repeats the V2 held-out setup (23 products × 3 variants, original downloaded alpha). The procedural test uses the same 32 untouched examples. The previously generated transparent product sheet is also retained as one known-alpha generated test. New Gemini test images, including gradient cases, get visual comparisons only. These tests do not establish universal performance on AI-generated banners.

Genuine green foreground remains ambiguous. Floor gradients add another distinct problem: the RGB uncompositing wrapper assumes a pure green backdrop, so pale floors and nonuniform lighting can remain opaque or produce color errors. The gradient test exposes this limitation rather than counting inferred masks as truth.
