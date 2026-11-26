#!/usr/bin/env python3
"""Packages a finished run into proof_of_execution.tar.gz.

Collects the run log, nvidia-smi output, metrics CSV and figures, and
converts every input/output image to PNG with names that make the
before/after relationship obvious:

  images/<name>_1_before_input.png
  images/<name>_2_after_equalized.png
  images/<name>_3_after_blurred.png
  images/<name>_4_after_edges.png

Images larger than --max_side pixels are downscaled for the archive (the
full-resolution PGMs stay in results/output). Requires Pillow.

Usage:
  python3 scripts/package_proof.py --results results --input data/input
"""

import argparse
import csv
import os
import shutil
import tarfile
import tempfile

from PIL import Image

TOP_LEVEL_FILES = ["run_log.txt", "nvidia_smi.txt", "timing_by_stage.png",
                   "speedup.png", "gallery.png"]
STAGES = [("equalized", "2_after_equalized"), ("blurred", "3_after_blurred"),
          ("edges", "4_after_edges")]


def to_png(src, dst, max_side):
    image = Image.open(src).convert("L")
    image.thumbnail((max_side, max_side))
    image.save(dst, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--input", default="data/input")
    parser.add_argument("--out", default="proof_of_execution.tar.gz")
    parser.add_argument("--max_side", type=int, default=1024)
    args = parser.parse_args()

    metrics = os.path.join(args.results, "output", "metrics.csv")
    if not os.path.exists(metrics):
        raise SystemExit("no %s - run ./run.sh first" % metrics)

    staging = tempfile.mkdtemp()
    root = os.path.join(staging, "proof_of_execution")
    os.makedirs(os.path.join(root, "images"))
    shutil.copy(metrics, root)
    for name in TOP_LEVEL_FILES:
        path = os.path.join(args.results, name)
        if os.path.exists(path):
            shutil.copy(path, root)

    with open(metrics, newline="") as f:
        names = [row["image"] for row in csv.DictReader(f)]
    for name in names:
        images_dir = os.path.join(root, "images")
        src = os.path.join(args.input, name + ".pgm")
        if os.path.exists(src):
            to_png(src, os.path.join(images_dir, name + "_1_before_input.png"),
                   args.max_side)
        for stage, label in STAGES:
            src = os.path.join(args.results, "output",
                               "%s_%s.pgm" % (name, stage))
            if os.path.exists(src):
                to_png(src, os.path.join(images_dir,
                                         "%s_%s.png" % (name, label)),
                       args.max_side)

    with open(os.path.join(root, "README.txt"), "w") as f:
        f.write(
            "Proof of execution for the CUDA batch image pipeline.\n\n"
            "run_log.txt     console output of ./run.sh (GPU model, every\n"
            "                image processed, summary, stream scaling)\n"
            "nvidia_smi.txt  GPU that executed the run\n"
            "metrics.csv     per-image GPU/CPU stage timings + verification\n"
            "*.png           timing, speed-up and before/after gallery\n"
            "images/         every image: 1_before_input -> 2_after_equalized"
            "\n                -> 3_after_blurred -> 4_after_edges\n"
            "                (downscaled to max %d px for upload size)\n"
            % args.max_side)

    with tarfile.open(args.out, "w:gz") as tar:
        tar.add(root, arcname="proof_of_execution")
    shutil.rmtree(staging)
    print("wrote %s (%.1f MB, %d images)" %
          (args.out, os.path.getsize(args.out) / 1e6, len(names)))


if __name__ == "__main__":
    main()
