// Copyright 2026 The cuda-batch-image-pipeline Authors.
//
// Single-threaded CPU implementations of every pipeline stage. They are used
// to (1) verify the GPU results and (2) provide a baseline for speed-ups.

#ifndef SRC_CPU_REFERENCE_H_
#define SRC_CPU_REFERENCE_H_

#include "src/image.h"

namespace imgpipe {

// Global histogram equalization. Uses the same integer rounding as the GPU
// LUT kernel, so the results must match bit-for-bit.
GrayImage CpuEqualizeHistogram(const GrayImage& input);

// Spatial separable Gaussian blur with circular (wrap-around) borders, the
// same boundary condition that FFT-based convolution implies. The kernel is
// truncated at 4 sigma and renormalised.
GrayImage CpuGaussianBlur(const GrayImage& input, float sigma);

// 3x3 Sobel gradient magnitude with clamp-to-edge borders, rounded and
// saturated to [0, 255]. Bit-exact with the GPU kernel for identical input.
GrayImage CpuSobelMagnitude(const GrayImage& input);

// Image comparison helpers.
int MaxAbsDiff(const GrayImage& a, const GrayImage& b);
// Peak signal-to-noise ratio in dB; returns 99.0 for identical images.
double Psnr(const GrayImage& a, const GrayImage& b);

}  // namespace imgpipe

#endif  // SRC_CPU_REFERENCE_H_
