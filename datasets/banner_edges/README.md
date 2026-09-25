# Synthetic banner stress set

1,240 exact-compositing examples: 1,024 train, 72 validation, 144 test. All inherit the banner dataset's source-product split; no source hash crosses splits. The eight cases are tiny text/thin lines, translucency, green spill, heavy JPEG compression, green foreground, pale floor, off-green background and soft shadow. Each source product appears once per case. Train/validation/test seeds are disjoint.

Run `build_edge_cases.py` after building the base banner dataset. Run `verify_banner_edges.py` to verify every image, alpha label and split. `manifest.jsonl` records source hashes, seeds and paths. Labels are exact for compositing but inherit imperfect source cutouts. Translucency and shadows are synthetic effects, not physically rendered glass or lighting.

Raw image files are ignored by Git; the manifest, generators and verification report are published. These stress examples share product sources with the corresponding banner split, so the two test suites are correlated.
