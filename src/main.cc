// Copyright 2026 The cuda-batch-image-pipeline Authors.
//
// Entry point: loads a directory of images, runs the GPU pipeline on all of
// them, optionally runs the CPU reference to verify and benchmark, and writes
// result images, a per-image CSV and a summary.

#include <algorithm>
#include <cctype>
#include <chrono>  // NOLINT(build/c++11)
#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "src/cli_options.h"
#include "src/cpu_reference.h"
#include "src/gpu_pipeline.h"
#include "src/image.h"

namespace fs = std::filesystem;

namespace imgpipe {
namespace {

// Minimum acceptable agreement between the cuFFT blur and the spatial CPU
// blur. They use different (but equivalent up to rounding) formulations.
constexpr double kMinBlurPsnrDb = 40.0;

struct CpuTimings {
  double equalize_ms = 0.0;
  double blur_ms = 0.0;
  double sobel_ms = 0.0;
  double total_ms = 0.0;
};

struct Verification {
  int equalize_max_diff = 0;
  double blur_psnr_db = 0.0;
  int sobel_max_diff = 0;
  bool passed = true;
};

struct LoadedImage {
  std::string name;  // File name without extension.
  GrayImage image;
};

double MsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

std::vector<LoadedImage> LoadImages(const std::string& dir, int max_images) {
  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".pgm" || ext == ".ppm") paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());

  std::vector<LoadedImage> images;
  for (const fs::path& path : paths) {
    if (max_images > 0 && static_cast<int>(images.size()) >= max_images) break;
    LoadedImage loaded;
    loaded.name = path.stem().string();
    std::string error;
    if (!ReadPnm(path.string(), &loaded.image, &error)) {
      std::cerr << "warning: skipping " << path << ": " << error << "\n";
      continue;
    }
    images.push_back(std::move(loaded));
  }
  return images;
}

CpuTimings RunCpuPipeline(const GrayImage& input, float sigma,
                          GrayImage* equalized, GrayImage* blurred,
                          GrayImage* edges) {
  CpuTimings t;
  auto start = std::chrono::steady_clock::now();
  *equalized = CpuEqualizeHistogram(input);
  t.equalize_ms = MsSince(start);
  start = std::chrono::steady_clock::now();
  *blurred = CpuGaussianBlur(*equalized, sigma);
  t.blur_ms = MsSince(start);
  start = std::chrono::steady_clock::now();
  *edges = CpuSobelMagnitude(*blurred);
  t.sobel_ms = MsSince(start);
  t.total_ms = t.equalize_ms + t.blur_ms + t.sobel_ms;
  return t;
}

Verification Verify(const GpuResult& gpu, const GrayImage& cpu_equalized,
                    const GrayImage& cpu_blurred) {
  Verification v;
  v.equalize_max_diff = MaxAbsDiff(gpu.equalized, cpu_equalized);
  v.blur_psnr_db = Psnr(gpu.blurred, cpu_blurred);
  // Check the Sobel kernel in isolation by feeding the CPU the exact same
  // (GPU-blurred) input, so blur rounding differences cannot leak in.
  v.sobel_max_diff = MaxAbsDiff(gpu.edges, CpuSobelMagnitude(gpu.blurred));
  v.passed = v.equalize_max_diff == 0 && v.sobel_max_diff == 0 &&
             v.blur_psnr_db >= kMinBlurPsnrDb;
  return v;
}

int Run(const CliOptions& options) {
  std::vector<LoadedImage> loaded =
      LoadImages(options.input_dir, options.max_images);
  if (loaded.empty()) {
    std::cerr << "error: no .pgm/.ppm images found in " << options.input_dir
              << "\n";
    return 1;
  }
  std::vector<GrayImage> images;
  size_t total_pixels = 0;
  for (const LoadedImage& l : loaded) {
    images.push_back(l.image);
    total_pixels += l.image.Size();
  }
  fs::create_directories(options.output_dir);
  const fs::path csv_parent = fs::path(options.csv_path).parent_path();
  if (!csv_parent.empty()) fs::create_directories(csv_parent);

  std::printf("%s\n", DescribeCudaDevice().c_str());
  std::printf("Images: %zu (%.2f Mpixel total) from %s\n", images.size(),
              total_pixels / 1e6, options.input_dir.c_str());
  std::printf("Config: sigma=%.2f streams=%d cpu_check=%s\n", options.sigma,
              options.num_streams, options.cpu_check ? "on" : "off");

  // GPU run. Setup (allocation, cuFFT planning) is timed separately.
  auto start = std::chrono::steady_clock::now();
  GpuPipeline pipeline(options.num_streams, options.sigma);
  pipeline.Prepare(images);
  const double gpu_setup_ms = MsSince(start);
  start = std::chrono::steady_clock::now();
  std::vector<GpuResult> gpu_results = pipeline.ProcessBatch(images);
  const double gpu_wall_ms = MsSince(start);

  std::ofstream csv(options.csv_path);
  if (!csv) {
    std::cerr << "error: cannot write " << options.csv_path << "\n";
    return 1;
  }
  csv << "image,width,height,gpu_upload_ms,gpu_equalize_ms,gpu_blur_fft_ms,"
         "gpu_sobel_ms,gpu_download_ms,gpu_total_ms,cpu_equalize_ms,"
         "cpu_blur_ms,cpu_sobel_ms,cpu_total_ms,equalize_max_diff,"
         "blur_psnr_db,sobel_max_diff,status\n";

  double cpu_wall_ms = 0.0;
  int failures = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    const GpuResult& gpu = gpu_results[i];
    const GpuStageTimings& g = gpu.timings;
    CpuTimings c;
    Verification v;
    std::string status = "skipped";
    if (options.cpu_check) {
      GrayImage cpu_eq, cpu_blur, cpu_edges;
      c = RunCpuPipeline(images[i], options.sigma, &cpu_eq, &cpu_blur,
                         &cpu_edges);
      cpu_wall_ms += c.total_ms;
      v = Verify(gpu, cpu_eq, cpu_blur);
      status = v.passed ? "PASS" : "FAIL";
      if (!v.passed) ++failures;
    }
    if (options.save_images) {
      const std::string base = options.output_dir + "/" + loaded[i].name;
      if (!WritePgm(base + "_equalized.pgm", gpu.equalized) ||
          !WritePgm(base + "_blurred.pgm", gpu.blurred) ||
          !WritePgm(base + "_edges.pgm", gpu.edges)) {
        std::cerr << "error: failed to write outputs for " << base << "\n";
        return 1;
      }
    }
    char line[512];
    std::snprintf(line, sizeof(line),
                  "%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,"
                  "%.3f,%d,%.2f,%d,%s\n",
                  loaded[i].name.c_str(), images[i].width, images[i].height,
                  g.upload_ms, g.equalize_ms, g.blur_fft_ms, g.sobel_ms,
                  g.download_ms, g.total_ms, c.equalize_ms, c.blur_ms,
                  c.sobel_ms, c.total_ms, v.equalize_max_diff,
                  v.blur_psnr_db, v.sobel_max_diff, status.c_str());
    csv << line;
    std::printf("[%3zu/%zu] %-28s %5dx%-5d gpu %8.3f ms  cpu %9.3f ms  %s\n",
                i + 1, images.size(), loaded[i].name.c_str(),
                images[i].width, images[i].height, g.total_ms, c.total_ms,
                status.c_str());
  }

  const double mpix = total_pixels / 1e6;
  std::printf("\n=== Summary ===\n");
  std::printf("GPU setup (alloc + cuFFT plans): %10.2f ms\n", gpu_setup_ms);
  std::printf("GPU batch wall time (incl. copies): %7.2f ms  (%.1f Mpixel/s)"
              "\n", gpu_wall_ms, mpix / (gpu_wall_ms / 1e3));
  if (options.cpu_check) {
    std::printf("CPU reference wall time:       %10.2f ms  (%.1f Mpixel/s)\n",
                cpu_wall_ms, mpix / (cpu_wall_ms / 1e3));
    std::printf("End-to-end speed-up:           %10.1fx\n",
                cpu_wall_ms / gpu_wall_ms);
    std::printf("Verification: %zu/%zu images PASS\n",
                images.size() - failures, images.size());
  }
  std::printf("Per-image metrics written to %s\n", options.csv_path.c_str());
  return failures == 0 ? 0 : 2;
}

}  // namespace
}  // namespace imgpipe

int main(int argc, char** argv) {
  imgpipe::CliOptions options;
  std::string error;
  if (!imgpipe::ParseCliOptions(argc, argv, &options, &error)) {
    std::cerr << "error: " << error << "\n\n" << imgpipe::CliUsage(argv[0]);
    return 1;
  }
  if (options.show_help) {
    std::cout << imgpipe::CliUsage(argv[0]);
    return 0;
  }
  return imgpipe::Run(options);
}
