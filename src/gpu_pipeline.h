// Copyright 2026 The cuda-batch-image-pipeline Authors.
//
// GPU batch pipeline:  input -> histogram equalization (custom kernels + CUB
// scan) -> Gaussian blur in the frequency domain (cuFFT) -> Sobel edge
// magnitude (shared-memory tiled kernel).
//
// Images are spread over several CUDA streams ("lanes"). Each lane owns its
// device buffers, pinned host staging buffers, cuFFT plans and timing events,
// so copies and kernels of different images can overlap on the device.

#ifndef SRC_GPU_PIPELINE_H_
#define SRC_GPU_PIPELINE_H_

#include <memory>
#include <string>
#include <vector>

#include "src/image.h"

namespace imgpipe {

// Per-image GPU timings in milliseconds, measured with CUDA events. Because
// lanes run concurrently these are latencies, not exclusive device time.
struct GpuStageTimings {
  float upload_ms = 0.0f;
  float equalize_ms = 0.0f;
  float blur_fft_ms = 0.0f;
  float sobel_ms = 0.0f;
  float download_ms = 0.0f;
  float total_ms = 0.0f;
};

struct GpuResult {
  GrayImage equalized;
  GrayImage blurred;
  GrayImage edges;
  GpuStageTimings timings;
};

// Returns a one-line description of the active CUDA device.
std::string DescribeCudaDevice();

class GpuPipeline {
 public:
  GpuPipeline(int num_streams, float sigma);
  ~GpuPipeline();

  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;

  // Allocates buffers and creates cuFFT plans for every image size in
  // `images`, so that one-time setup cost is excluded from benchmarks.
  void Prepare(const std::vector<GrayImage>& images);

  // Runs the full pipeline on every image. Results are returned in input
  // order. Up to `num_streams` images are in flight at once.
  std::vector<GpuResult> ProcessBatch(const std::vector<GrayImage>& images);

  int num_streams() const { return static_cast<int>(lanes_.size()); }

 private:
  struct Lane;

  void Enqueue(Lane* lane, const GrayImage& image);
  void Collect(Lane* lane, GpuResult* result);

  float sigma_;
  std::vector<std::unique_ptr<Lane>> lanes_;
};

}  // namespace imgpipe

#endif  // SRC_GPU_PIPELINE_H_
