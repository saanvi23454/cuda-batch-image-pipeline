// Copyright 2026 The cuda-batch-image-pipeline Authors.
//
// Host-side launchers for the custom CUDA kernels used by the pipeline. Every
// launcher is asynchronous with respect to the host and runs on `stream`.

#ifndef SRC_KERNELS_H_
#define SRC_KERNELS_H_

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstddef>
#include <cstdint>

namespace imgpipe {

constexpr int kHistogramBins = 256;

// Accumulates a 256-bin histogram of `image` into `histogram`, which must be
// zeroed beforehand. Uses per-block shared-memory histograms and vectorised
// 4-pixel loads to reduce global atomic contention.
void LaunchHistogram(const uint8_t* image, size_t num_pixels,
                     unsigned int* histogram, cudaStream_t stream);

// Builds the equalization look-up table from the inclusive CDF.
void LaunchBuildEqualizationLut(const unsigned int* cdf, uint8_t* lut,
                                cudaStream_t stream);

// Applies `lut` to every pixel, writing both an 8-bit copy and a float copy
// (the float copy feeds the cuFFT stage directly).
void LaunchApplyLut(const uint8_t* input, const uint8_t* lut, uint8_t* output,
                    float* output_float, size_t num_pixels,
                    cudaStream_t stream);

// Multiplies the R2C spectrum (height x (width/2+1)) by the analytic
// Gaussian transfer function and by 1/(width*height) to normalise the inverse
// transform.
void LaunchGaussianSpectrumFilter(cufftComplex* spectrum, int width,
                                  int height, float sigma,
                                  cudaStream_t stream);

// Rounds and saturates floats to [0, 255].
void LaunchFloatToU8(const float* input, uint8_t* output, size_t num_pixels,
                     cudaStream_t stream);

// 3x3 Sobel gradient magnitude using a shared-memory tile with a one-pixel
// halo and clamp-to-edge borders.
void LaunchSobelMagnitude(const uint8_t* input, uint8_t* output, int width,
                          int height, cudaStream_t stream);

}  // namespace imgpipe

#endif  // SRC_KERNELS_H_
