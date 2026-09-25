# Larger model experiment

The source-level ecommerce split is fixed in `datasets/products/manifest.json`: 157 training products, 12 validation products and 23 test products. Each source has its original downloaded alpha. Augmentations are generated only after splitting. The main training run uses 1,884 product composites plus 256 procedural text/shape composites. This is not 2,140 independent products.

Architecture: six full-resolution dilated convolutions, with channel widths 24, 32, 48, 48, 32, 24 and dilation 1, 1, 2, 4, 2, 1, followed by a one-channel sigmoid. The receptive field grows from 7×7 in v1 to 23×23 in v2. No pretrained DeepLab weights are used. Native-resolution tiling uses a 160×160 input and 12-pixel halo, retaining 136×136 pixels per tile.

Loss weights alpha boundaries and soft edges. Checkpoint selection uses only validation loss. Training uses 96×96 crops/composites; evaluation uses larger images to check spatial transfer. The model is fully convolutional and does not globally resize banners.

Gemini image generation was attempted once with `gemini-3.1-flash-lite-image` and once with `gemini-2.5-flash-image`. Both returned HTTP 429 `RESOURCE_EXHAUSTED`. Authentication/model listing succeeded. No Gemini-generated images were returned, and no retries were made. Credentials are not saved in the project.

Fallback: one image was generated with the built-in image generator, containing six ecommerce products on genuine transparency. Its path and prompt are in `datasets/generated/builtin_manifest.json`. The entire source is held out of training. Evaluation composites the actual generated alpha over green and tests JPEG qualities 60, 80 and 95. This is one independent generated source, not six independently generated images or three independent source images. Its alpha includes any imperfections in the generated cutout.

`generate_products.py` is available for a future Gemini run after quota is enabled. It reads `GEMINI_API_KEY` from the environment, defaults to 24 planned images, logs attempts, skips completed images, and stops on errors. It never embeds the key. Automatically estimated masks from such generated green-background images are weak labels only; quantitative evaluation requires independently available alpha.
