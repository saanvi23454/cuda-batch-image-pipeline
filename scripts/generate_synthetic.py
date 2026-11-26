#!/usr/bin/env python3
"""Generates deterministic synthetic grayscale test images as binary PGM.

The images are deliberately low-contrast (so histogram equalization has
visible work to do), contain sharp geometric edges (for the Sobel stage) and
noise (for the Gaussian blur). Sizes vary, including odd sizes, to exercise
the vectorised histogram tail and odd-width cuFFT R2C transforms.

Usage:
  python3 scripts/generate_synthetic.py --out data/input --count 24
"""

import argparse
import os

import numpy as np

SIZES = [(256, 256), (512, 512), (640, 480), (1023, 769), (1024, 1024),
         (1920, 1080), (2048, 2048)]


def make_image(rng, width, height, index):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    # Smooth background gradient with a random orientation.
    angle = rng.uniform(0, np.pi)
    img = 0.5 + 0.25 * np.sin((np.cos(angle) * xx + np.sin(angle) * yy) /
                              max(width, height) * np.pi * 2)
    # A handful of filled circles and rectangles with sharp edges.
    for _ in range(4 + index % 5):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        r = rng.uniform(0.05, 0.2) * min(width, height)
        level = rng.uniform(0.2, 0.8)
        if rng.random() < 0.5:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
        else:
            mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < 0.6 * r)
        img[mask] = level
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    # Squash into a narrow intensity band -> low contrast input.
    lo = rng.uniform(60, 110)
    span = rng.uniform(40, 70)
    img = lo + span * np.clip(img, 0, 1)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def write_pgm(path, pixels):
    height, width = pixels.shape
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (width, height))
        f.write(pixels.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/input")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.count):
        width, height = SIZES[i % len(SIZES)]
        pixels = make_image(rng, width, height, i)
        path = os.path.join(args.out, "synthetic_%02d_%dx%d.pgm" %
                            (i, width, height))
        write_pgm(path, pixels)
        print("wrote", path)


if __name__ == "__main__":
    main()
