// Copyright 2026 The cuda-batch-image-pipeline Authors.

#include "src/cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace imgpipe {
namespace {

constexpr int kHistogramBins = 256;

inline int Wrap(int i, int n) {
  i %= n;
  return i < 0 ? i + n : i;
}

inline int Clamp(int v, int lo, int hi) {
  return std::min(std::max(v, lo), hi);
}

}  // namespace

GrayImage CpuEqualizeHistogram(const GrayImage& input) {
  int64_t cdf[kHistogramBins] = {0};
  for (uint8_t p : input.pixels) ++cdf[p];
  for (int i = 1; i < kHistogramBins; ++i) cdf[i] += cdf[i - 1];

  const int64_t total = cdf[kHistogramBins - 1];
  int64_t cdf_min = 0;
  for (int i = 0; i < kHistogramBins; ++i) {
    if (cdf[i] > 0) {
      cdf_min = cdf[i];
      break;
    }
  }
  const int64_t denom = total - cdf_min;

  uint8_t lut[kHistogramBins];
  for (int i = 0; i < kHistogramBins; ++i) {
    if (denom <= 0) {
      lut[i] = static_cast<uint8_t>(i);  // Constant image: identity mapping.
    } else {
      const int64_t num = std::max<int64_t>(cdf[i] - cdf_min, 0) * 255;
      lut[i] = static_cast<uint8_t>((num + denom / 2) / denom);
    }
  }

  GrayImage out(input.width, input.height);
  for (size_t i = 0; i < input.Size(); ++i) {
    out.pixels[i] = lut[input.pixels[i]];
  }
  return out;
}

GrayImage CpuGaussianBlur(const GrayImage& input, float sigma) {
  const int w = input.width, h = input.height;
  const int radius = std::max(1, static_cast<int>(std::ceil(4.0f * sigma)));
  std::vector<float> taps(2 * radius + 1);
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    taps[k + radius] = std::exp(-(k * k) / (2.0f * sigma * sigma));
    sum += taps[k + radius];
  }
  for (float& t : taps) t /= sum;

  std::vector<float> tmp(input.Size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float acc = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        acc += taps[k + radius] * input.At(Wrap(x + k, w), y);
      }
      tmp[y * w + x] = acc;
    }
  }
  GrayImage out(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float acc = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        acc += taps[k + radius] * tmp[Wrap(y + k, h) * w + x];
      }
      out.pixels[y * w + x] =
          static_cast<uint8_t>(Clamp(static_cast<int>(acc + 0.5f), 0, 255));
    }
  }
  return out;
}

GrayImage CpuSobelMagnitude(const GrayImage& input) {
  const int w = input.width, h = input.height;
  GrayImage out(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      auto p = [&](int dx, int dy) {
        return static_cast<int>(
            input.At(Clamp(x + dx, 0, w - 1), Clamp(y + dy, 0, h - 1)));
      };
      const int gx = (p(1, -1) + 2 * p(1, 0) + p(1, 1)) -
                     (p(-1, -1) + 2 * p(-1, 0) + p(-1, 1));
      const int gy = (p(-1, 1) + 2 * p(0, 1) + p(1, 1)) -
                     (p(-1, -1) + 2 * p(0, -1) + p(1, -1));
      const float mag = std::sqrt(static_cast<float>(gx * gx + gy * gy));
      out.pixels[y * w + x] =
          static_cast<uint8_t>(std::min(255, static_cast<int>(mag + 0.5f)));
    }
  }
  return out;
}

int MaxAbsDiff(const GrayImage& a, const GrayImage& b) {
  int max_diff = 0;
  for (size_t i = 0; i < a.Size(); ++i) {
    max_diff = std::max(max_diff, std::abs(a.pixels[i] - b.pixels[i]));
  }
  return max_diff;
}

double Psnr(const GrayImage& a, const GrayImage& b) {
  double sse = 0.0;
  for (size_t i = 0; i < a.Size(); ++i) {
    const double d = static_cast<double>(a.pixels[i]) - b.pixels[i];
    sse += d * d;
  }
  if (sse == 0.0) return 99.0;
  const double mse = sse / static_cast<double>(a.Size());
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

}  // namespace imgpipe
