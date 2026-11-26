// Copyright 2026 The cuda-batch-image-pipeline Authors.

#include "src/cli_options.h"

#include <cstdlib>
#include <sstream>
#include <string>

namespace imgpipe {
namespace {

bool ParseFloat(const std::string& text, float* value) {
  char* end = nullptr;
  *value = std::strtof(text.c_str(), &end);
  return !text.empty() && end != nullptr && *end == '\0';
}

bool ParseInt(const std::string& text, int* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text.c_str(), &end, 10);  // NOLINT
  if (text.empty() || end == nullptr || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

}  // namespace

std::string CliUsage(const char* program_name) {
  std::ostringstream out;
  out << "Usage: " << program_name << " [flags]\n\n"
      << "Batch GPU image pipeline: histogram equalization (CUB) -> Gaussian\n"
      << "blur via cuFFT -> Sobel edge magnitude, with CPU verification.\n\n"
      << "Flags:\n"
      << "  --input_dir=DIR    Directory of .pgm/.ppm images "
         "(default data/input)\n"
      << "  --output_dir=DIR   Where result images go (default "
         "results/output)\n"
      << "  --csv=FILE         Per-image metrics CSV (default "
         "<output_dir>/metrics.csv)\n"
      << "  --sigma=FLOAT      Gaussian standard deviation in pixels "
         "(default 2.0)\n"
      << "  --streams=INT      Number of concurrent CUDA streams "
         "(default 4)\n"
      << "  --max_images=INT   Process at most this many images (default "
         "all)\n"
      << "  --no_cpu_check     Skip the CPU reference and verification\n"
      << "  --no_save          Do not write result images\n"
      << "  --help             Show this message\n";
  return out.str();
}

bool ParseCliOptions(int argc, char** argv, CliOptions* options,
                     std::string* error) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      *error = "unexpected argument: " + arg;
      return false;
    }
    std::string name = arg.substr(2);
    std::string value;
    bool has_value = false;
    const size_t eq = name.find('=');
    if (eq != std::string::npos) {
      value = name.substr(eq + 1);
      name = name.substr(0, eq);
      has_value = true;
    }

    // Boolean flags take no value.
    if (name == "help" || name == "no_cpu_check" || name == "no_save") {
      if (has_value) {
        *error = "--" + name + " takes no value";
        return false;
      }
      if (name == "help") options->show_help = true;
      if (name == "no_cpu_check") options->cpu_check = false;
      if (name == "no_save") options->save_images = false;
      continue;
    }

    if (!has_value) {
      if (i + 1 >= argc) {
        *error = "missing value for --" + name;
        return false;
      }
      value = argv[++i];
    }

    if (name == "input_dir") {
      options->input_dir = value;
    } else if (name == "output_dir") {
      options->output_dir = value;
    } else if (name == "csv") {
      options->csv_path = value;
    } else if (name == "sigma") {
      if (!ParseFloat(value, &options->sigma) || options->sigma <= 0.0f ||
          options->sigma > 64.0f) {
        *error = "--sigma must be a number in (0, 64]";
        return false;
      }
    } else if (name == "streams") {
      if (!ParseInt(value, &options->num_streams) ||
          options->num_streams < 1 || options->num_streams > 32) {
        *error = "--streams must be an integer in [1, 32]";
        return false;
      }
    } else if (name == "max_images") {
      if (!ParseInt(value, &options->max_images) || options->max_images < 0) {
        *error = "--max_images must be a non-negative integer";
        return false;
      }
    } else {
      *error = "unknown flag --" + name;
      return false;
    }
  }
  if (options->csv_path.empty()) {
    options->csv_path = options->output_dir + "/metrics.csv";
  }
  return true;
}

}  // namespace imgpipe
