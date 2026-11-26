// Copyright 2026 The cuda-batch-image-pipeline Authors.

#include "src/gpu_pipeline.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cufft.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/cuda_check.h"
#include "src/kernels.h"

namespace imgpipe {
namespace {

// Event indices marking the boundaries between pipeline stages.
enum EventIndex {
  kEventStart = 0,
  kEventUploaded,
  kEventEqualized,
  kEventBlurred,
  kEventEdges,
  kEventDownloaded,
  kNumEvents
};

template <typename T>
void FreeDevice(T** ptr) {
  if (*ptr != nullptr) CUDA_CHECK(cudaFree(*ptr));
  *ptr = nullptr;
}

template <typename T>
void FreeHost(T** ptr) {
  if (*ptr != nullptr) CUDA_CHECK(cudaFreeHost(*ptr));
  *ptr = nullptr;
}

float ElapsedMs(cudaEvent_t from, cudaEvent_t to) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, from, to));
  return ms;
}

}  // namespace

std::string DescribeCudaDevice() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int runtime_version = 0, driver_version = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  std::ostringstream out;
  out << "GPU " << device << ": " << prop.name << " (compute capability "
      << prop.major << "." << prop.minor << ", "
      << prop.multiProcessorCount << " SMs, "
      << (prop.totalGlobalMem >> 20) << " MiB), CUDA runtime "
      << runtime_version / 1000 << "." << (runtime_version % 1000) / 10
      << ", driver " << driver_version / 1000 << "."
      << (driver_version % 1000) / 10;
  return out.str();
}

// Everything one stream needs to process one image at a time.
struct GpuPipeline::Lane {
  cudaStream_t stream = nullptr;
  cudaEvent_t events[kNumEvents] = {};

  // Size of the image currently in flight on this lane.
  int width = 0;
  int height = 0;
  size_t capacity = 0;  // Pixels the buffers below can hold.

  // Device buffers.
  uint8_t* d_input = nullptr;
  uint8_t* d_equalized = nullptr;
  uint8_t* d_blurred = nullptr;
  uint8_t* d_edges = nullptr;
  float* d_real = nullptr;
  cufftComplex* d_spectrum = nullptr;
  unsigned int* d_histogram = nullptr;
  unsigned int* d_cdf = nullptr;
  uint8_t* d_lut = nullptr;
  void* d_scan_temp = nullptr;
  size_t scan_temp_bytes = 0;

  // Pinned host staging buffers (required for truly asynchronous copies).
  uint8_t* h_input = nullptr;
  uint8_t* h_equalized = nullptr;
  uint8_t* h_blurred = nullptr;
  uint8_t* h_edges = nullptr;

  // cuFFT plans keyed by (width, height): {forward R2C, inverse C2R}.
  std::map<std::pair<int, int>, std::pair<cufftHandle, cufftHandle>> plans;

  Lane() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (cudaEvent_t& e : events) CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaMalloc(&d_histogram, kHistogramBins * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_cdf, kHistogramBins * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_lut, kHistogramBins));
    // Query and allocate CUB's scratch space once, so the scan never
    // allocates (and therefore never synchronises) inside the hot loop.
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, scan_temp_bytes,
                                             d_histogram, d_cdf,
                                             kHistogramBins, stream));
    CUDA_CHECK(cudaMalloc(&d_scan_temp, scan_temp_bytes));
  }

  ~Lane() {
    ReleaseImageBuffers();
    for (auto& entry : plans) {
      cufftDestroy(entry.second.first);
      cufftDestroy(entry.second.second);
    }
    FreeDevice(&d_histogram);
    FreeDevice(&d_cdf);
    FreeDevice(&d_lut);
    FreeDevice(&d_scan_temp);
    for (cudaEvent_t e : events) cudaEventDestroy(e);
    cudaStreamDestroy(stream);
  }

  void ReleaseImageBuffers() {
    FreeDevice(&d_input);
    FreeDevice(&d_equalized);
    FreeDevice(&d_blurred);
    FreeDevice(&d_edges);
    FreeDevice(&d_real);
    FreeDevice(&d_spectrum);
    FreeHost(&h_input);
    FreeHost(&h_equalized);
    FreeHost(&h_blurred);
    FreeHost(&h_edges);
    capacity = 0;
  }

  // Grows the buffers so that a w x h image fits. A real image of width w
  // has an R2C spectrum of h * (w/2 + 1) complex values, which is at most
  // h * w complex values for any w >= 1.
  void EnsureCapacity(int w, int h) {
    const size_t pixels = static_cast<size_t>(w) * h;
    if (pixels <= capacity) return;
    ReleaseImageBuffers();
    CUDA_CHECK(cudaMalloc(&d_input, pixels));
    CUDA_CHECK(cudaMalloc(&d_equalized, pixels));
    CUDA_CHECK(cudaMalloc(&d_blurred, pixels));
    CUDA_CHECK(cudaMalloc(&d_edges, pixels));
    CUDA_CHECK(cudaMalloc(&d_real, pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum, pixels * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMallocHost(&h_input, pixels));
    CUDA_CHECK(cudaMallocHost(&h_equalized, pixels));
    CUDA_CHECK(cudaMallocHost(&h_blurred, pixels));
    CUDA_CHECK(cudaMallocHost(&h_edges, pixels));
    capacity = pixels;
  }

  // Returns (creating if necessary) the plans for a w x h image, bound to
  // this lane's stream.
  std::pair<cufftHandle, cufftHandle> PlansFor(int w, int h) {
    const auto key = std::make_pair(w, h);
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;
    cufftHandle forward, inverse;
    // cuFFT uses row-major (slowest dimension first) ordering: nx = rows.
    CUFFT_CHECK(cufftPlan2d(&forward, h, w, CUFFT_R2C));
    CUFFT_CHECK(cufftPlan2d(&inverse, h, w, CUFFT_C2R));
    CUFFT_CHECK(cufftSetStream(forward, stream));
    CUFFT_CHECK(cufftSetStream(inverse, stream));
    plans[key] = std::make_pair(forward, inverse);
    return plans[key];
  }
};

GpuPipeline::GpuPipeline(int num_streams, float sigma) : sigma_(sigma) {
  // Touch the context up front so its creation is not billed to image 0.
  CUDA_CHECK(cudaFree(nullptr));
  for (int i = 0; i < num_streams; ++i) {
    lanes_.push_back(std::make_unique<Lane>());
  }
}

GpuPipeline::~GpuPipeline() = default;

void GpuPipeline::Prepare(const std::vector<GrayImage>& images) {
  int max_w = 1, max_h = 1;
  size_t max_pixels = 0;
  for (const GrayImage& image : images) {
    if (image.Size() > max_pixels) {
      max_pixels = image.Size();
      max_w = image.width;
      max_h = image.height;
    }
  }
  for (auto& lane : lanes_) {
    lane->EnsureCapacity(max_w, max_h);
    for (const GrayImage& image : images) {
      lane->PlansFor(image.width, image.height);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

void GpuPipeline::Enqueue(Lane* lane, const GrayImage& image) {
  const int w = image.width, h = image.height;
  const size_t n = image.Size();
  lane->EnsureCapacity(w, h);
  const auto plans = lane->PlansFor(w, h);
  lane->width = w;
  lane->height = h;
  cudaStream_t s = lane->stream;

  std::memcpy(lane->h_input, image.pixels.data(), n);

  CUDA_CHECK(cudaEventRecord(lane->events[kEventStart], s));
  CUDA_CHECK(cudaMemcpyAsync(lane->d_input, lane->h_input, n,
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaEventRecord(lane->events[kEventUploaded], s));

  // Stage 1: histogram equalization.
  CUDA_CHECK(cudaMemsetAsync(lane->d_histogram, 0,
                             kHistogramBins * sizeof(unsigned), s));
  LaunchHistogram(lane->d_input, n, lane->d_histogram, s);
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(lane->d_scan_temp,
                                           lane->scan_temp_bytes,
                                           lane->d_histogram, lane->d_cdf,
                                           kHistogramBins, s));
  LaunchBuildEqualizationLut(lane->d_cdf, lane->d_lut, s);
  LaunchApplyLut(lane->d_input, lane->d_lut, lane->d_equalized, lane->d_real,
                 n, s);
  CUDA_CHECK(cudaEventRecord(lane->events[kEventEqualized], s));

  // Stage 2: Gaussian blur as a point-wise product in the frequency domain.
  CUFFT_CHECK(cufftExecR2C(plans.first, lane->d_real, lane->d_spectrum));
  LaunchGaussianSpectrumFilter(lane->d_spectrum, w, h, sigma_, s);
  CUFFT_CHECK(cufftExecC2R(plans.second, lane->d_spectrum, lane->d_real));
  LaunchFloatToU8(lane->d_real, lane->d_blurred, n, s);
  CUDA_CHECK(cudaEventRecord(lane->events[kEventBlurred], s));

  // Stage 3: Sobel edge magnitude of the blurred image.
  LaunchSobelMagnitude(lane->d_blurred, lane->d_edges, w, h, s);
  CUDA_CHECK(cudaEventRecord(lane->events[kEventEdges], s));

  CUDA_CHECK(cudaMemcpyAsync(lane->h_equalized, lane->d_equalized, n,
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(lane->h_blurred, lane->d_blurred, n,
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaMemcpyAsync(lane->h_edges, lane->d_edges, n,
                             cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaEventRecord(lane->events[kEventDownloaded], s));
}

void GpuPipeline::Collect(Lane* lane, GpuResult* result) {
  CUDA_CHECK(cudaStreamSynchronize(lane->stream));
  const int w = lane->width, h = lane->height;
  const size_t n = static_cast<size_t>(w) * h;
  result->equalized = GrayImage(w, h);
  result->blurred = GrayImage(w, h);
  result->edges = GrayImage(w, h);
  std::memcpy(result->equalized.pixels.data(), lane->h_equalized, n);
  std::memcpy(result->blurred.pixels.data(), lane->h_blurred, n);
  std::memcpy(result->edges.pixels.data(), lane->h_edges, n);

  const cudaEvent_t* e = lane->events;
  GpuStageTimings& t = result->timings;
  t.upload_ms = ElapsedMs(e[kEventStart], e[kEventUploaded]);
  t.equalize_ms = ElapsedMs(e[kEventUploaded], e[kEventEqualized]);
  t.blur_fft_ms = ElapsedMs(e[kEventEqualized], e[kEventBlurred]);
  t.sobel_ms = ElapsedMs(e[kEventBlurred], e[kEventEdges]);
  t.download_ms = ElapsedMs(e[kEventEdges], e[kEventDownloaded]);
  t.total_ms = ElapsedMs(e[kEventStart], e[kEventDownloaded]);
}

std::vector<GpuResult> GpuPipeline::ProcessBatch(
    const std::vector<GrayImage>& images) {
  std::vector<GpuResult> results(images.size());
  const size_t lanes = lanes_.size();
  // Process in waves: enqueue one image per lane, then drain the wave. All
  // lanes of a wave execute concurrently on the device.
  for (size_t wave_start = 0; wave_start < images.size();
       wave_start += lanes) {
    const size_t wave_end = std::min(images.size(), wave_start + lanes);
    for (size_t i = wave_start; i < wave_end; ++i) {
      Enqueue(lanes_[i - wave_start].get(), images[i]);
    }
    for (size_t i = wave_start; i < wave_end; ++i) {
      Collect(lanes_[i - wave_start].get(), &results[i]);
    }
  }
  return results;
}

}  // namespace imgpipe
