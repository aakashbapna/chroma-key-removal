# Data, labels and splits

## Product banners

192 alpha-bearing product assets were originally downloaded from DummyJSON; 155 suitable product sources contribute to the one-product offer-banner dataset. The source split is 128 training / 9 validation / 18 test products. Source hashes and IDs do not overlap between these splits.

2,264 exact-compositing banners: 2,048 training / 72 validation / 144 test. Each contains one product asset, offer headline, supporting text and CTA. A source SKU photo may show several components of that product. Backgrounds comprise 1,698 solid green, 283 shaded green and 283 green-to-white floor gradients. Labels remove the full background, including the floor. Source-provided cutout alpha was not manually corrected.

Sizes are 640×320, 512×512 and 384×512, with varied fonts, placements, JPEG quality and subsampling. The manifest records combined alpha, RGBA target, product/text/decoration masks, background, source hashes and rendering seeds. Fonts are referenced by local path/hash and are not redistributed. `build_banner_dataset.py --skip-generated` supports rebuilding without Gemini assets.

## Generated product derivatives

The prior bounded Gemini run saved 98 images from 100 request reservations. Two interrupted requests had unknown output; they were not retried. No further generation is used by the current experiments.

Existing generated images supplied 348 offer-banner derivatives: 304 weak training examples, 38 solid-green visual tests and 6 original floor-gradient visual tests. Product masks are approximate; rendered text is exact. Uncertain product pixels have zero confidence, confidently estimated product regions reduced confidence. Visual-only tests have no independent product alpha. Some original gradient failures share a subject with a training view; these are qualitative cases, not independent subject benchmarks.

## Synthetic edge cases

1,240 stress banners: 1,024 training / 72 validation / 144 test. Eight cases: tiny text/thin lines, translucency, green spill, heavy JPEG compression, green foreground, pale floor, off-green background and soft shadow. Each source product contributes one per case. Source-level splits are inherited unchanged. Train/validation/test seeds are disjoint. The banner and stress tests share the same 18 products, so their measurements are correlated.

## Supplied real banners

Local files are `datasets/real_banners/makeup.jpg` and `watch.jpg`. The user requested that the makeup banner—including lavender background and both cubes—stay intact, giving an exact all-opaque target. The watch's pseudo-alpha is estimated by local chroma color projection, with confidence 0.15 away from uncertain boundaries; it is not a hand-labeled target.

64 makeup crops and 128 watch crops were added to the follow-up training run. Both originals are training/acceptance examples, never included in independent accuracy metrics. Watch acceptance checks verify selected transparent/opaque locations and remaining green excess at the visible boundary; they cannot establish true alpha accuracy.

## Storage and loading

Raw/downloaded/generated source images and optimizer checkpoints stay local and are ignored by Git. Manifests, hashes, generators, verification reports, selected result images and compact model checkpoints are tracked. All manifest paths are relative to the project root. `banner_data.py` returns RGB plus alpha/confidence/text-alpha labels and excludes weak examples from validation/test. Regenerating against a changed upstream catalog can change hashes, splits and results.

## Additional derivatives

`datasets/real_variants/manifest.jsonl` records 256 more full-banner derivatives from the same two user sources, all training-only. Makeup: 128 variants with complete artwork retained, slight rotations and varied surrounding colors; exact all-opaque targets. Watch: 128 recomposites across pure/dark/off-green backgrounds, floor gradients, green spill, compression, smaller placement and soft edges; alpha inherited from the estimated source cutout, confidence 0.15. None is an independent test example. No extra API calls or weight updates were made when generating these derivatives.
