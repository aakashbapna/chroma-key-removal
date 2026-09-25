# Native-resolution diagnostics

Eleven diagnostic inputs compare the original V3 with selected Run 2 using the actual 256-pixel tile / 64-pixel halo / 128-pixel stride inference wrapper. These examples share a single source product and are qualitative checks, not a representative accuracy estimate. Main metrics use all 144 test banners and 144 stress images at standardized resolution.

The selected model passed finite-alpha/dimension checks for 1×1, 9×7, 129×127 and 133×261 inputs. Pure-green alpha was exactly zero in these cases. Maximum measured Keras/TFLite difference on a banner input was 0.000598.

Native floor-gradient alpha MAE improved from 0.2930 (original V3) to 0.0692 (Run 2), but Run 1 was better on this same example at 0.0123. The Run 2 off-green diagnostic regressed from 0.1233 to 0.2280 alpha MAE. Thus full-resolution gradient/off-green behavior remains weaker than the resized benchmark suggests. Check `verification.json` for every diagnostic, and inspect the images; neither new checkpoint is uniformly superior.

`comparison.jpg` shows solid green, shaded green and floor-gradient results. Each row is input, ground truth, original V3, selected Run 2. Individual transparent PNGs are included. Browser smoke reports are stored under the corresponding run directories.
