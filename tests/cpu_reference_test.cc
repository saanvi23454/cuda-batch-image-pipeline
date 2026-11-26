// Copyright 2026 The cuda-batch-image-pipeline Authors.
//
// Self-contained unit tests for the host-side code (CPU reference stages,
// PGM I/O and CLI parsing). Runs without a GPU: `make cpu_test`.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "src/cli_options.h"
#include "src/cpu_reference.h"
#include "src/image.h"

namespace imgpipe {
namespace {

int g_failures = 0;

void Expect(bool condition, const char* what) {
  std::printf("  [%s] %s\n", condition ? " OK " : "FAIL", what);
  if (!condition) ++g_failures;
}

GrayImage MakeRamp(int w, int h, int lo, int hi) {
  GrayImage img(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      img.pixels[y * w + x] =
          static_cast<uint8_t>(lo + (hi - lo) * x / std::max(1, w - 1));
    }
  }
  return img;
}

void TestEqualization() {
  std::printf("Histogram equalization\n");
  const GrayImage ramp = MakeRamp(64, 8, 100, 140);
  const GrayImage eq = CpuEqualizeHistogram(ramp);
  int lo = 255, hi = 0;
  for (uint8_t p : eq.pixels) {
    lo = std::min<int>(lo, p);
    hi = std::max<int>(hi, p);
  }
  Expect(lo == 0 && hi == 255, "low-contrast ramp is stretched to [0,255]");
  bool monotonic = true;
  for (int x = 1; x < ramp.width; ++x) {
    if (eq.At(x, 0) < eq.At(x - 1, 0)) monotonic = false;
  }
  Expect(monotonic, "mapping is monotonic");

  GrayImage flat(10, 10);
  std::fill(flat.pixels.begin(), flat.pixels.end(), 77);
  Expect(MaxAbsDiff(CpuEqualizeHistogram(flat), flat) == 0,
         "constant image is left unchanged");
}

void TestBlur() {
  std::printf("Gaussian blur\n");
  GrayImage flat(33, 17);
  std::fill(flat.pixels.begin(), flat.pixels.end(), 123);
  Expect(MaxAbsDiff(CpuGaussianBlur(flat, 2.0f), flat) == 0,
         "constant image is a fixed point");

  GrayImage impulse(41, 41);
  impulse.pixels[20 * 41 + 20] = 255;
  const GrayImage blurred = CpuGaussianBlur(impulse, 1.5f);
  Expect(blurred.At(20, 20) < 255 && blurred.At(20, 20) > 0,
         "impulse is spread out");
  Expect(blurred.At(19, 20) == blurred.At(21, 20) &&
             blurred.At(20, 19) == blurred.At(20, 21),
         "response is symmetric");
  // Circular boundary: a bright column at x=0 leaks into x=w-1.
  GrayImage edge(16, 4);
  for (int y = 0; y < 4; ++y) edge.pixels[y * 16] = 255;
  Expect(CpuGaussianBlur(edge, 1.0f).At(15, 0) > 0,
         "borders wrap around (matches FFT convolution)");
}

void TestSobel() {
  std::printf("Sobel magnitude\n");
  GrayImage step(8, 8);
  for (int y = 0; y < 8; ++y) {
    for (int x = 4; x < 8; ++x) step.pixels[y * 8 + x] = 50;
  }
  const GrayImage edges = CpuSobelMagnitude(step);
  Expect(edges.At(3, 4) == 200 && edges.At(4, 4) == 200,
         "vertical step of 50 gives magnitude 4*50");
  Expect(edges.At(0, 4) == 0 && edges.At(7, 4) == 0,
         "flat regions give zero");
  GrayImage big_step(8, 8);
  for (int y = 0; y < 8; ++y) {
    for (int x = 4; x < 8; ++x) big_step.pixels[y * 8 + x] = 255;
  }
  Expect(CpuSobelMagnitude(big_step).At(4, 4) == 255, "output saturates");
}

void TestMetrics() {
  std::printf("Metrics\n");
  const GrayImage a = MakeRamp(16, 16, 0, 255);
  GrayImage b = a;
  Expect(Psnr(a, b) == 99.0 && MaxAbsDiff(a, b) == 0, "identical images");
  b.pixels[5] = static_cast<uint8_t>(b.pixels[5] + 3);
  Expect(MaxAbsDiff(a, b) == 3, "max diff detects a single pixel");
  Expect(std::fabs(Psnr(a, b) - 10.0 * std::log10(255.0 * 255.0 /
                                                   (9.0 / 256.0))) < 1e-9,
         "PSNR formula");
}

void TestPgmRoundTrip() {
  std::printf("PGM I/O\n");
  const GrayImage img = MakeRamp(13, 7, 3, 250);
  const std::string path = "bin/roundtrip_test.pgm";
  Expect(WritePgm(path, img), "write succeeds");
  GrayImage back;
  std::string error;
  Expect(ReadPnm(path, &back, &error), "read succeeds");
  Expect(back.width == 13 && back.height == 7 && MaxAbsDiff(img, back) == 0,
         "pixels survive a round trip");
  Expect(!ReadPnm("bin/does_not_exist.pgm", &back, &error),
         "missing file is reported");
}

void TestCli() {
  std::printf("CLI parsing\n");
  const char* argv[] = {"prog", "--input_dir=in", "--sigma", "3.5",
                        "--streams=8", "--no_save"};
  CliOptions opts;
  std::string error;
  Expect(ParseCliOptions(6, const_cast<char**>(argv), &opts, &error),
         "valid flags parse");
  Expect(opts.input_dir == "in" && opts.sigma == 3.5f &&
             opts.num_streams == 8 && !opts.save_images && opts.cpu_check,
         "values are applied");
  Expect(opts.csv_path == "results/output/metrics.csv", "csv path default");

  const char* bad[] = {"prog", "--streams=0"};
  CliOptions opts2;
  Expect(!ParseCliOptions(2, const_cast<char**>(bad), &opts2, &error),
         "out-of-range value is rejected");
  const char* unknown[] = {"prog", "--bogus=1"};
  Expect(!ParseCliOptions(2, const_cast<char**>(unknown), &opts2, &error),
         "unknown flag is rejected");
}

}  // namespace
}  // namespace imgpipe

int main() {
  imgpipe::TestEqualization();
  imgpipe::TestBlur();
  imgpipe::TestSobel();
  imgpipe::TestMetrics();
  imgpipe::TestPgmRoundTrip();
  imgpipe::TestCli();
  std::printf("\n%s (%d failure%s)\n",
              imgpipe::g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
              imgpipe::g_failures, imgpipe::g_failures == 1 ? "" : "s");
  return imgpipe::g_failures == 0 ? 0 : 1;
}
