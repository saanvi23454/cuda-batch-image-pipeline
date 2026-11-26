# Input images

`run.sh` fills this folder automatically:

- `sipi_*.pgm` - USC-SIPI "Miscellaneous" volume, converted to 8-bit grayscale by `scripts/fetch_sipi.py`.
- `synthetic_*.pgm` - deterministic low-contrast test images from `scripts/generate_synthetic.py` (sizes 256x256 up to 2048x2048, including odd sizes).

Any binary `.pgm` (P5) or `.ppm` (P6) file dropped here is picked up by the pipeline.
