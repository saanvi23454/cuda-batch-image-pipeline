#!/usr/bin/env python3
"""Turns a pipeline run into proof-of-execution figures.

Reads results/output/metrics.csv and the result PGMs and writes:
  results/timing_by_stage.png  GPU vs CPU time per stage, per image size
  results/speedup.png          CPU/GPU speed-up versus image size
  results/gallery.png          input -> equalized -> blurred -> edges strips

Requires numpy, matplotlib and Pillow.

Usage:
  python3 scripts/plot_results.py --results results --input data/input
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from PIL import Image  # pylint: disable=wrong-import-position


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key not in ("image", "status"):
                row[key] = float(value)
        row["mpix"] = row["width"] * row["height"] / 1e6
    return sorted(rows, key=lambda r: r["mpix"])


def plot_stages(rows, out_path):
    stages = [("equalize", "gpu_equalize_ms", "cpu_equalize_ms"),
              ("blur (cuFFT)", "gpu_blur_fft_ms", "cpu_blur_ms"),
              ("sobel", "gpu_sobel_ms", "cpu_sobel_ms")]
    sizes = sorted({(int(r["width"]), int(r["height"])) for r in rows},
                   key=lambda s: s[0] * s[1])
    fig, axes = plt.subplots(1, len(stages), figsize=(15, 4.5))
    for ax, (label, gpu_key, cpu_key) in zip(axes, stages):
        gpu = [np.mean([r[gpu_key] for r in rows
                        if (r["width"], r["height"]) == s]) for s in sizes]
        cpu = [np.mean([r[cpu_key] for r in rows
                        if (r["width"], r["height"]) == s]) for s in sizes]
        x = np.arange(len(sizes))
        ax.bar(x - 0.2, cpu, 0.4, label="CPU (1 thread)", color="#8c8c8c")
        ax.bar(x + 0.2, gpu, 0.4, label="GPU", color="#2a7ab9")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["%dx%d" % s for s in sizes], rotation=45,
                           ha="right", fontsize=8)
        ax.set_title(label)
        ax.set_ylabel("ms (log scale)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle("Per-stage time, mean per image size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_speedup(rows, out_path):
    mpix = [r["mpix"] for r in rows]
    kernel_gpu = [r["gpu_equalize_ms"] + r["gpu_blur_fft_ms"] +
                  r["gpu_sobel_ms"] for r in rows]
    speedup_kernels = [r["cpu_total_ms"] / g if g > 0 else 0
                       for r, g in zip(rows, kernel_gpu)]
    speedup_total = [r["cpu_total_ms"] / r["gpu_total_ms"]
                     if r["gpu_total_ms"] > 0 else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(mpix, speedup_kernels, label="compute only", color="#2a7ab9")
    ax.scatter(mpix, speedup_total, label="incl. host<->device copies",
               color="#e08a1e", marker="s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("image size (Mpixel)")
    ax.set_ylabel("speed-up vs single-thread CPU")
    ax.set_title("GPU speed-up versus image size")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def load_gray(path, size=256):
    image = Image.open(path).convert("L")
    image.thumbnail((size, size))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(image, ((size - image.width) // 2,
                         (size - image.height) // 2))
    return np.asarray(canvas)


def plot_gallery(rows, results_dir, input_dir, out_path, max_rows=4):
    output_dir = os.path.join(results_dir, "output")
    # Pick evenly spaced examples across the size range.
    picks = [rows[i] for i in np.linspace(0, len(rows) - 1,
                                          min(max_rows, len(rows))).astype(int)]
    titles = ["input", "equalized", "blurred (cuFFT)", "edges (Sobel)"]
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(picks):
        name = row["image"]
        paths = [os.path.join(input_dir, name + ".pgm")] + [
            os.path.join(output_dir, "%s_%s.pgm" % (name, suffix))
            for suffix in ("equalized", "blurred", "edges")]
        for c, path in enumerate(paths):
            ax = axes[r, c]
            ax.axis("off")
            if os.path.exists(path):
                ax.imshow(load_gray(path), cmap="gray", vmin=0, vmax=255)
            if r == 0:
                ax.set_title(titles[c])
        axes[r, 0].text(-10, 128, name, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--input", default="data/input")
    args = parser.parse_args()

    rows = load_rows(os.path.join(args.results, "output", "metrics.csv"))
    if not rows:
        raise SystemExit("metrics.csv is empty")
    plot_stages(rows, os.path.join(args.results, "timing_by_stage.png"))
    if rows[0]["cpu_total_ms"] > 0:
        plot_speedup(rows, os.path.join(args.results, "speedup.png"))
    plot_gallery(rows, args.results, args.input,
                 os.path.join(args.results, "gallery.png"))
    print("figures written to", args.results)


if __name__ == "__main__":
    main()
