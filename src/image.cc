// Copyright 2026 The cuda-batch-image-pipeline Authors.

#include "src/image.h"

#include <cctype>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace imgpipe {
namespace {

// Reads the next whitespace-delimited header token, skipping '#' comments.
bool ReadHeaderToken(std::istream* in, std::string* token) {
  token->clear();
  int c = in->get();
  while (c != EOF) {
    if (c == '#') {
      while (c != EOF && c != '\n') c = in->get();
    } else if (std::isspace(c)) {
      if (!token->empty()) return true;
    } else {
      token->push_back(static_cast<char>(c));
    }
    c = in->get();
  }
  return !token->empty();
}

bool ParsePositiveInt(const std::string& token, int* value) {
  if (token.empty() || token.size() > 9) return false;
  for (char c : token) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  }
  *value = std::stoi(token);
  return *value > 0;
}

}  // namespace

bool ReadPnm(const std::string& path, GrayImage* image, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open file";
    return false;
  }
  std::string magic, w_token, h_token, max_token;
  if (!ReadHeaderToken(&in, &magic) || !ReadHeaderToken(&in, &w_token) ||
      !ReadHeaderToken(&in, &h_token) || !ReadHeaderToken(&in, &max_token)) {
    *error = "truncated header";
    return false;
  }
  if (magic != "P5" && magic != "P6") {
    *error = "unsupported format " + magic + " (need binary P5 or P6)";
    return false;
  }
  int width = 0, height = 0, max_value = 0;
  if (!ParsePositiveInt(w_token, &width) ||
      !ParsePositiveInt(h_token, &height) ||
      !ParsePositiveInt(max_token, &max_value) || max_value > 255) {
    *error = "invalid header values (only 8-bit images are supported)";
    return false;
  }

  const int channels = (magic == "P6") ? 3 : 1;
  std::vector<uint8_t> raw(static_cast<size_t>(width) * height * channels);
  in.read(reinterpret_cast<char*>(raw.data()),
          static_cast<std::streamsize>(raw.size()));
  if (in.gcount() != static_cast<std::streamsize>(raw.size())) {
    *error = "truncated pixel data";
    return false;
  }

  *image = GrayImage(width, height);
  if (channels == 1) {
    image->pixels = std::move(raw);
  } else {
    for (size_t i = 0; i < image->Size(); ++i) {
      const int r = raw[3 * i], g = raw[3 * i + 1], b = raw[3 * i + 2];
      image->pixels[i] =
          static_cast<uint8_t>((299 * r + 587 * g + 114 * b + 500) / 1000);
    }
  }
  if (max_value != 255) {
    for (uint8_t& p : image->pixels) {
      p = static_cast<uint8_t>((p * 255 + max_value / 2) / max_value);
    }
  }
  return true;
}

bool WritePgm(const std::string& path, const GrayImage& image) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "P5\n" << image.width << " " << image.height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.Size()));
  return static_cast<bool>(out);
}

}  // namespace imgpipe
