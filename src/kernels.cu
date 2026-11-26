// Copyright 2026 The cuda-batch-image-pipeline Authors.

#include "src/kernels.h"

#include <algorithm>

#include "src/cuda_check.h"

namespace imgpipe {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxGridBlocks = 1024;
constexpr int kSobelTile = 16;
constexpr float kPi = 3.14159265358979f;

int GridSizeFor(size_t work_items) {
  const size_t blocks = (work_items + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(
      std::max<size_t>(1, std::min<size_t>(blocks, kMaxGridBlocks)));
}

__global__ void HistogramKernel(const uint8_t* __restrict__ image,
                                size_t num_pixels,
                                unsigned int* __restrict__ histogram) {
  __shared__ unsigned int local[kHistogramBins];
  for (int i = threadIdx.x; i < kHistogramBins; i += blockDim.x) local[i] = 0;
  __syncthreads();

  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  // Main body: one 32-bit load fetches four pixels (cudaMalloc guarantees
  // 256-byte alignment of `image`).
  const size_t num_words = num_pixels / 4;
  const uchar4* words = reinterpret_cast<const uchar4*>(image);
  for (size_t i = first; i < num_words; i += stride) {
    const uchar4 v = words[i];
    atomicAdd(&local[v.x], 1u);
    atomicAdd(&local[v.y], 1u);
    atomicAdd(&local[v.z], 1u);
    atomicAdd(&local[v.w], 1u);
  }
  // Tail: the last (num_pixels % 4) pixels.
  for (size_t i = num_words * 4 + first; i < num_pixels; i += stride) {
    atomicAdd(&local[image[i]], 1u);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kHistogramBins; i += blockDim.x) {
    if (local[i] != 0) atomicAdd(&histogram[i], local[i]);
  }
}

__global__ void BuildLutKernel(const unsigned int* __restrict__ cdf,
                               uint8_t* __restrict__ lut) {
  __shared__ unsigned int cdf_min;
  const int i = threadIdx.x;
  if (i == 0) cdf_min = 0;
  __syncthreads();
  // The CDF is non-decreasing, so exactly one bin is the first non-zero one.
  if (cdf[i] > 0 && (i == 0 || cdf[i - 1] == 0)) cdf_min = cdf[i];
  __syncthreads();

  const int64_t total = cdf[kHistogramBins - 1];
  const int64_t denom = total - static_cast<int64_t>(cdf_min);
  if (denom <= 0) {
    lut[i] = static_cast<uint8_t>(i);  // Constant image: identity mapping.
    return;
  }
  int64_t num = static_cast<int64_t>(cdf[i]) - cdf_min;
  num = (num < 0 ? 0 : num) * 255;
  lut[i] = static_cast<uint8_t>((num + denom / 2) / denom);
}

__global__ void ApplyLutKernel(const uint8_t* __restrict__ input,
                               const uint8_t* __restrict__ lut,
                               uint8_t* __restrict__ output,
                               float* __restrict__ output_float,
                               size_t num_pixels) {
  __shared__ uint8_t local_lut[kHistogramBins];
  for (int i = threadIdx.x; i < kHistogramBins; i += blockDim.x) {
    local_lut[i] = lut[i];
  }
  __syncthreads();
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_pixels; i += stride) {
    const uint8_t v = local_lut[input[i]];
    output[i] = v;
    output_float[i] = static_cast<float>(v);
  }
}

__global__ void GaussianSpectrumKernel(cufftComplex* spectrum, int width,
                                       int height, float sigma) {
  const int half_width = width / 2 + 1;
  const int kx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ky = blockIdx.y * blockDim.y + threadIdx.y;
  if (kx >= half_width || ky >= height) return;

  // Signed frequencies in cycles/pixel. The R2C layout stores only
  // non-negative x frequencies; y frequencies wrap past height/2.
  const float fx = static_cast<float>(kx) / width;
  const float fy = static_cast<float>(ky <= height / 2 ? ky : ky - height) /
                   height;
  // Fourier transform of a unit-area Gaussian with std-dev sigma.
  const float gain = expf(-2.0f * kPi * kPi * sigma * sigma *
                          (fx * fx + fy * fy)) /
                     (static_cast<float>(width) * height);
  cufftComplex& c = spectrum[static_cast<size_t>(ky) * half_width + kx];
  c.x *= gain;
  c.y *= gain;
}

__global__ void FloatToU8Kernel(const float* __restrict__ input,
                                uint8_t* __restrict__ output,
                                size_t num_pixels) {
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_pixels; i += stride) {
    const int v = static_cast<int>(input[i] + 0.5f);
    output[i] = static_cast<uint8_t>(min(max(v, 0), 255));
  }
}

__global__ void SobelKernel(const uint8_t* __restrict__ input,
                            uint8_t* __restrict__ output, int width,
                            int height) {
  constexpr int kTileWithHalo = kSobelTile + 2;
  __shared__ int tile[kTileWithHalo][kTileWithHalo];

  const int origin_x = blockIdx.x * kSobelTile - 1;
  const int origin_y = blockIdx.y * kSobelTile - 1;
  // Cooperative load of the (tile + halo) region, clamping at image borders.
  for (int ty = threadIdx.y; ty < kTileWithHalo; ty += kSobelTile) {
    for (int tx = threadIdx.x; tx < kTileWithHalo; tx += kSobelTile) {
      const int gx = min(max(origin_x + tx, 0), width - 1);
      const int gy = min(max(origin_y + ty, 0), height - 1);
      tile[ty][tx] = input[static_cast<size_t>(gy) * width + gx];
    }
  }
  __syncthreads();

  const int x = blockIdx.x * kSobelTile + threadIdx.x;
  const int y = blockIdx.y * kSobelTile + threadIdx.y;
  if (x >= width || y >= height) return;
  const int cx = threadIdx.x + 1;
  const int cy = threadIdx.y + 1;
  const int gx = (tile[cy - 1][cx + 1] + 2 * tile[cy][cx + 1] +
                  tile[cy + 1][cx + 1]) -
                 (tile[cy - 1][cx - 1] + 2 * tile[cy][cx - 1] +
                  tile[cy + 1][cx - 1]);
  const int gy = (tile[cy + 1][cx - 1] + 2 * tile[cy + 1][cx] +
                  tile[cy + 1][cx + 1]) -
                 (tile[cy - 1][cx - 1] + 2 * tile[cy - 1][cx] +
                  tile[cy - 1][cx + 1]);
  // __fsqrt_rn is IEEE round-to-nearest even under --use_fast_math, which
  // keeps the result bit-exact with std::sqrt on the CPU.
  const float mag = __fsqrt_rn(static_cast<float>(gx * gx + gy * gy));
  output[static_cast<size_t>(y) * width + x] =
      static_cast<uint8_t>(min(255, static_cast<int>(mag + 0.5f)));
}

}  // namespace

void LaunchHistogram(const uint8_t* image, size_t num_pixels,
                     unsigned int* histogram, cudaStream_t stream) {
  HistogramKernel<<<GridSizeFor(num_pixels / 4 + 1), kThreadsPerBlock, 0,
                    stream>>>(image, num_pixels, histogram);
  CUDA_CHECK_LAUNCH();
}

void LaunchBuildEqualizationLut(const unsigned int* cdf, uint8_t* lut,
                                cudaStream_t stream) {
  BuildLutKernel<<<1, kHistogramBins, 0, stream>>>(cdf, lut);
  CUDA_CHECK_LAUNCH();
}

void LaunchApplyLut(const uint8_t* input, const uint8_t* lut, uint8_t* output,
                    float* output_float, size_t num_pixels,
                    cudaStream_t stream) {
  ApplyLutKernel<<<GridSizeFor(num_pixels), kThreadsPerBlock, 0, stream>>>(
      input, lut, output, output_float, num_pixels);
  CUDA_CHECK_LAUNCH();
}

void LaunchGaussianSpectrumFilter(cufftComplex* spectrum, int width,
                                  int height, float sigma,
                                  cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((width / 2 + 1 + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  GaussianSpectrumKernel<<<grid, block, 0, stream>>>(spectrum, width, height,
                                                     sigma);
  CUDA_CHECK_LAUNCH();
}

void LaunchFloatToU8(const float* input, uint8_t* output, size_t num_pixels,
                     cudaStream_t stream) {
  FloatToU8Kernel<<<GridSizeFor(num_pixels), kThreadsPerBlock, 0, stream>>>(
      input, output, num_pixels);
  CUDA_CHECK_LAUNCH();
}

void LaunchSobelMagnitude(const uint8_t* input, uint8_t* output, int width,
                          int height, cudaStream_t stream) {
  const dim3 block(kSobelTile, kSobelTile);
  const dim3 grid((width + kSobelTile - 1) / kSobelTile,
                  (height + kSobelTile - 1) / kSobelTile);
  SobelKernel<<<grid, block, 0, stream>>>(input, output, width, height);
  CUDA_CHECK_LAUNCH();
}

}  // namespace imgpipe
