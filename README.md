# CUDA Batch Image Pipeline: cuFFT + CUB + Custom Kernels on Many Images

This is the capstone project for the GPU Programming Specialization (course: CUDA Advanced Libraries).

The program takes a folder of grayscale images and runs a three-stage enhancement and edge-detection pipeline on every image. All stages run on the GPU, and several CUDA streams keep more than one image in flight at a time. A single-threaded CPU version of every stage runs alongside it. The CPU results check the GPU output (bit-exact where that is possible) and give a baseline for the speed-up.

```
            ┌──────────── Stage 1 ────────────┐  ┌──────── Stage 2 ────────┐  ┌─ Stage 3 ─┐
 input ──►  histogram ─► CUB inclusive ─► LUT  ─►  cuFFT R2C ─► × Gaussian ─►  Sobel 3x3   ─► edges
 (uint8)    (shared-mem   scan → CDF     apply     H(f) ─► cuFFT C2R          (shared-mem
            atomics)                     │                    │                tile + halo)
                                         ▼                    ▼                    ▼
                                     equalized.pgm        blurred.pgm          edges.pgm
```

## Why this project

Image pre-processing for computer vision usually looks like this: lots of images, cheap operations on each one, and a pipeline where the data should stay on the device between steps. I wanted a project that brings together the course's main topics in one realistic program:

| Topic from the course | Where it is used |
|---|---|
| **cuFFT** (Advanced Libraries) | Gaussian blur done as a point-wise product in the frequency domain (`R2C` → multiply → `C2R`) |
| **CUB / CCCL** (Thrust's backend) | `cub::DeviceScan::InclusiveSum` turns the histogram into a CDF |
| **Custom kernels and shared memory** | Per-block shared-memory histogram with `uchar4` vector loads; LUT kept in shared memory; Sobel with a 16×16 tile plus a 1-pixel halo |
| **Streams, events, pinned memory** | Each "lane" has its own stream, pinned staging buffers, cuFFT plans and timing events, so copies and kernels of different images overlap |
| **Verification and benchmarking** | CPU reference, per-stage CUDA-event timing, CSV metrics, plots |

## Repository layout

```
├── Makefile                  build (nvcc), `make cpu_test`, `make run`
├── run.sh                    one command: data → build → run → logs → plots
├── src/
│   ├── main.cc               CLI driver: load batch, run GPU + CPU, verify, write CSV
│   ├── gpu_pipeline.{h,cu}   GpuPipeline class: streams/lanes, buffers, cuFFT plans, CUB scan
│   ├── kernels.{h,cu}        custom CUDA kernels + launchers
│   ├── cpu_reference.{h,cc}  single-threaded reference of each stage, PSNR / max-diff
│   ├── image.{h,cc}          PGM/PPM reader and writer (no external dependencies)
│   ├── cli_options.{h,cc}    --flag=value parser
│   └── cuda_check.h          CUDA / cuFFT error checking
├── tests/cpu_reference_test.cc   unit tests that run without a GPU
├── scripts/
│   ├── fetch_sipi.py         downloads the USC-SIPI "misc" volume, converts it to PGM
│   ├── generate_synthetic.py deterministic low-contrast test images (256² … 2048²)
│   ├── plot_results.py       timing / speed-up / gallery figures from metrics.csv
│   └── package_proof.py      builds proof_of_execution.tar.gz (logs + before/after PNGs)
├── data/input/               input images (filled in by run.sh)
└── results/                  proof of execution (log, CSV, output images, figures)
```

## Requirements and installation

| Need | Version used / notes |
|---|---|
| NVIDIA GPU + driver | any CUDA-capable GPU (the Coursera lab GPU works) |
| CUDA Toolkit | 11.x, 12.x or 13.x (`nvcc`, cuFFT; CUB ships with the toolkit) |
| g++ | 9 or newer (C++17) |
| Python 3 + numpy | for data scripts; `pillow` for SIPI conversion; `matplotlib` for plots |

```bash
# Python helpers (only needed for data download and plots)
python3 -m pip install --user numpy pillow matplotlib
```

If CUDA is not in `/usr/local/cuda`, pass `CUDA_PATH=/path/to/cuda` to `make`. To build for one specific GPU, pass `ARCH=sm_75` (or `sm_86`, and so on).

## How to run

The quickest way to do everything:

```bash
./run.sh            # SIPI images if downloadable, plus 24 synthetic images
./run.sh --synthetic   # only synthetic images (no internet needed)
SIGMA=3 STREAMS=8 ./run.sh
```

Or step by step:

```bash
make                                   # builds bin/image_pipeline
make cpu_test                          # host-side unit tests (no GPU needed)
python3 scripts/generate_synthetic.py --out data/input --count 24
./bin/image_pipeline --input_dir=data/input --output_dir=results/output --sigma=2 --streams=4
```

### Running on Google Colab

1. In a new notebook, choose **Runtime → Change runtime type → T4 GPU**.
2. Run this in a cell:

```bash
!git clone https://github.com/saanvi23454/cuda-batch-image-pipeline.git
%cd cuda-batch-image-pipeline
!chmod +x run.sh && ./run.sh
```

3. Download `proof_of_execution.tar.gz` from the file browser on the left.

### Command-line flags

| Flag | Default | Meaning |
|---|---|---|
| `--input_dir=DIR` | `data/input` | folder of binary `.pgm` (P5) / `.ppm` (P6) images |
| `--output_dir=DIR` | `results/output` | where `<name>_equalized/_blurred/_edges.pgm` are written |
| `--csv=FILE` | `<output_dir>/metrics.csv` | per-image timing and verification metrics |
| `--sigma=F` | `2.0` | Gaussian standard deviation in pixels |
| `--streams=N` | `4` | concurrent CUDA streams (1–32) |
| `--max_images=N` | all | limit the batch size |
| `--no_cpu_check` | | skip the CPU reference (GPU timing only) |
| `--no_save` | | don't write output images |
| `--help` | | print usage |

The exit code is `0` when every image passes verification, `2` when any image fails, and `1` for usage or I/O errors.

## How the GPU is used (algorithms)

**Stage 1: Histogram equalization.** `HistogramKernel` builds a 256-bin histogram for each block in shared memory. Each thread reads four pixels at a time through `uchar4`, and each block adds its bins to the global histogram with one atomic per bin. This avoids millions of contended global atomics. `cub::DeviceScan::InclusiveSum` then produces the CDF. Its scratch buffer is allocated once per lane, so the scan never calls `cudaMalloc` (which would synchronize) inside the batch loop. `BuildLutKernel` maps `cdf → round(255·(cdf−cdf_min)/(N−cdf_min))` using 64-bit integer arithmetic. `ApplyLutKernel` keeps the LUT in shared memory and writes both a `uint8` copy and a `float` copy, which feeds cuFFT directly.

**Stage 2: Gaussian blur with cuFFT.** Blurring is a convolution, and in the frequency domain a convolution becomes a multiplication. The image goes through `cufftExecR2C`. Its half-spectrum (`h × (w/2+1)`) is multiplied by the analytic transfer function of a Gaussian, `H(f) = exp(−2π²σ²(fx²+fy²))`, with the `1/(w·h)` inverse normalization folded into the same multiply. `cufftExecC2R` then brings it back. The cost is `O(N log N)` whatever σ is, while direct convolution costs `O(N·σ)`. The plans for each image size are created once per lane in `Prepare()` and bound to the lane's stream.

**Stage 3: Sobel edge magnitude.** Each 16×16 thread block loads an 18×18 tile (tile plus halo) into shared memory, clamping at the image borders. Each thread then reads 8 neighbours from shared memory instead of global memory. The kernel uses `__fsqrt_rn` so the square root is IEEE round-to-nearest, which makes the result bit-exact with `std::sqrt` on the CPU.

**Batching and concurrency.** `GpuPipeline` owns `N` lanes. Each lane has its own non-blocking stream, pinned host buffers (needed for truly asynchronous `cudaMemcpyAsync`), device buffers, cuFFT plans and 6 CUDA events. Images are processed in waves of `N`: every lane's H2D copy, kernels and D2H copy are queued without any host synchronization, so the copy engines and SMs overlap work from different images. `run.sh` repeats the run with 1, 2, 4 and 8 streams to measure how much this helps.

## Verification strategy

Each image gets three independent checks, all recorded in `metrics.csv`:

1. **Equalization:** the GPU and CPU must match exactly (`equalize_max_diff == 0`). Both use the same integer formula.
2. **Blur:** the CPU uses a *spatial* separable Gaussian (truncated at 4σ) with wrap-around borders, which is a different algorithm from the GPU's FFT version. They should agree to within rounding, and the check requires a PSNR of at least 40 dB. Before the GPU run, I checked the frequency-domain formula against the spatial version with a NumPy model of the same math (`rfft2` → multiply → `irfft2`). For σ ∈ {1, 2, 4} and all seven test sizes (including 1023×769), the largest difference was 1 grey level and the PSNR was 66–88 dB.
3. **Sobel:** this stage is checked on its own. The CPU Sobel runs on the *GPU's* blurred image, so the edge maps must match exactly (`sobel_max_diff == 0`).

## Proof of execution

`run.sh` saves everything in `results/`. It also packs the logs and every before/after image (as PNG) into **`proof_of_execution.tar.gz`**, which is the file to upload to Coursera (`scripts/package_proof.py`). The logs, CSV and figures are committed here too. The full-resolution `.pgm` images are not, because of their size; `run.sh` regenerates them.

| File | What it shows |
|---|---|
| `results/run_log.txt` | full console output: GPU model, CUDA versions, every image processed, summary, stream-scaling experiment |
| `results/nvidia_smi.txt` | `nvidia-smi` output from the machine that ran the code |
| `results/output/metrics.csv` | per-image GPU stage times, CPU stage times, verification metrics and PASS/FAIL |
| `results/output/*_{equalized,blurred,edges}.pgm` | the output images (inside the tarball as `*_1_before_input.png` … `*_4_after_edges.png`) |
| `results/timing_by_stage.png`, `results/speedup.png`, `results/gallery.png` | figures generated from the CSV |

### Results

Run in the Coursera GPU lab on an **NVIDIA L4** (compute capability 8.9, 58 SMs, CUDA runtime 11.3, driver 12.6). The full log is in [`results/run_log.txt`](results/run_log.txt).

| Metric | Value |
|---|---|
| Images | 24 images, 256×256 to 2048×2048 (26.85 Mpixel total) |
| GPU batch wall time (incl. all host↔device copies) | **41.3 ms** (649.6 Mpixel/s) |
| CPU reference (single thread) | 1955.2 ms (13.7 Mpixel/s) |
| End-to-end speed-up | **47.3×** |
| Per-image speed-up (CPU time ÷ GPU latency) | up to ~22× for 256×256, up to ~180× for 1024×1024, ~185–195× for 2048×2048 |
| Verification | **24/24 PASS** (equalization and Sobel bit-exact, blur PSNR ≥ 40 dB) |
| One-time setup (CUDA context, buffers, cuFFT plans) | 1017.5 ms |

Stream scaling (same 24 images, GPU only):

| Streams | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Batch wall time (ms) | 49.38 | 43.60 | 41.11 | 39.79 |
| Throughput (Mpixel/s) | 543.8 | 615.8 | 653.1 | 674.8 |

What the numbers show:
* **Speed-up grows with image size.** A 256×256 image has too little work to fill 58 SMs, so fixed costs (kernel launches, small copies) dominate. At 2048×2048 the GPU is busy and the gap to the CPU reaches about 190×.
* **Streams help, but with diminishing returns.** Going from 1 to 8 streams cuts the batch time by 19%. The gain comes from overlapping one image's copies with another image's kernels. Beyond 4 streams there is little left to overlap.
* **Setup dominates a small batch.** Creating the CUDA context and the cuFFT plans for 7 image sizes × 4 lanes took about 25× longer than processing the whole batch. A real service would create them once and reuse them across batches, which is why they are timed separately here.
* **Per-image latency rises under concurrency.** Small images processed later in the batch show ~1.6 ms latencies instead of ~0.15 ms, because four lanes share the GPU. Overall throughput still improves, which is the trade-off streams make.
* **PTX JIT at work.** The lab's CUDA 11.3 compiler does not know the L4's sm_89 architecture. The default build embeds PTX, and the driver compiled it for the L4 at load time. This is part of the one-time setup cost.

## Lessons learned and challenges

* **FFT convolution means circular borders.** Multiplying spectra is a *circular* convolution, so the left edge of the image bleeds into the right edge. The CPU reference therefore uses wrap-around borders. Without that, the two would disagree at the edges and the comparison would be meaningless. Zero-padding to `(w+k)×(h+k)` would avoid the wrap if needed.
* **Bit-exact GPU/CPU agreement takes care.** Floating-point rounding is the usual reason GPU and CPU results differ slightly. Integer LUT arithmetic and `__fsqrt_rn` remove that for stages 1 and 3. For stage 2 a PSNR threshold is the honest test, because the two algorithms are different by design.
* **Hidden synchronization kills stream overlap.** A library call that allocates memory internally (for example a Thrust algorithm without a preallocated buffer) calls `cudaMalloc`/`cudaFree`, which serializes the streams. Querying CUB's scratch size once and reusing the buffer keeps the batch loop fully asynchronous. The same goes for pinned memory: async copies from pageable memory are silently synchronous.
* **Setup cost vs. steady state.** Creating a cuFFT plan costs far more than running it for one small image. Planning once per image size in `Prepare()` and timing it separately makes the benchmark reflect the steady state.
* **Small images don't fill the GPU.** For 256×256 images the kernels finish in microseconds and launch and copy overhead dominate. This is exactly the case where several streams help most.

## Next steps

* Replace the per-image wave with a persistent producer/consumer queue so a slow large image doesn't hold up its whole wave.
* Use batched cuFFT plans (`cufftPlanMany`) for many images of the same size.
* Compare the custom Sobel kernel with NPP's `nppiFilterSobel*` and the histogram with `nppiHistogramEven`.
* Extend to colour (equalize in YCbCr) and to video frames.

## Short description (for the Coursera submission)

I built a batch GPU image-processing pipeline in CUDA C++ that equalizes the histogram, blurs and edge-detects whole folders of images (USC-SIPI plus synthetic images up to 2048×2048). Histogram equalization uses a shared-memory histogram kernel with a CUB device-wide scan. The Gaussian blur runs in the frequency domain with cuFFT (R2C → analytic Gaussian transfer function → C2R). Sobel edges come from a shared-memory tiled kernel. Images go through several CUDA streams with pinned memory so transfers and kernels overlap. Every result is checked against a CPU implementation (bit-exact for equalization and Sobel, PSNR ≥ 40 dB for the FFT blur), and per-stage timings, speed-ups, logs and output images are committed to the repository as proof of execution.

## Attribution

* Images: USC-SIPI Image Database, Miscellaneous volume (https://sipi.usc.edu/database/), plus generated synthetic images.
* Project structure inspired by the course template: https://github.com/PascaleCourseraCourses/CUDAatScaleForTheEnterpriseCourseProjectTemplate
