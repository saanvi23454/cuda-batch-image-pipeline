// Copyright 2026 The cuda-batch-image-pipeline Authors.
//
// Command-line options for the batch pipeline.

#ifndef SRC_CLI_OPTIONS_H_
#define SRC_CLI_OPTIONS_H_

#include <string>

namespace imgpipe {

struct CliOptions {
  std::string input_dir = "data/input";
  std::string output_dir = "results/output";
  std::string csv_path;  // Defaults to <output_dir>/metrics.csv.
  float sigma = 2.0f;
  int num_streams = 4;
  int max_images = 0;  // 0 means "all images in input_dir".
  bool cpu_check = true;
  bool save_images = true;
  bool show_help = false;
};

// Parses flags of the form --name=value or --name value. Returns false and
// sets `error` on invalid input.
bool ParseCliOptions(int argc, char** argv, CliOptions* options,
                     std::string* error);

// Returns the --help text.
std::string CliUsage(const char* program_name);

}  // namespace imgpipe

#endif  // SRC_CLI_OPTIONS_H_
