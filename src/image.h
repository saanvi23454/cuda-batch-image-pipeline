// Copyright 2026 The cuda-batch-image-pipeline Authors.
//
// Minimal 8-bit grayscale image container plus PGM/PPM file I/O.

#ifndef SRC_IMAGE_H_
#define SRC_IMAGE_H_

#include <cstdint>
#include <string>
#include <vector>

namespace imgpipe {

// A dense, row-major, single-channel 8-bit image.
struct GrayImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;

  GrayImage() = default;
  GrayImage(int w, int h) : width(w), height(h), pixels(Size()) {}

  size_t Size() const {
    return static_cast<size_t>(width) * static_cast<size_t>(height);
  }
  uint8_t At(int x, int y) const { return pixels[y * width + x]; }
};

// Reads a binary PGM (P5) or PPM (P6) file. Colour images are converted to
// grayscale with ITU-R BT.601 luma weights. Returns false and fills `error` on
// failure.
bool ReadPnm(const std::string& path, GrayImage* image, std::string* error);

// Writes `image` as a binary PGM (P5) file. Returns false on failure.
bool WritePgm(const std::string& path, const GrayImage& image);

}  // namespace imgpipe

#endif  // SRC_IMAGE_H_
