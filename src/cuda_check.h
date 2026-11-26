// Copyright 2026 The cuda-batch-image-pipeline Authors.
//
// Error-checking helpers for the CUDA runtime and cuFFT APIs. Any failure is
// fatal: the message names the failing call, file and line, then exits.

#ifndef SRC_CUDA_CHECK_H_
#define SRC_CUDA_CHECK_H_

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>

namespace imgpipe {

inline void CheckCuda(cudaError_t status, const char* call, const char* file,
                      int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s at %s:%d in %s\n",
                 cudaGetErrorString(status), file, line, call);
    std::exit(EXIT_FAILURE);
  }
}

inline void CheckCufft(cufftResult status, const char* call, const char* file,
                       int line) {
  if (status != CUFFT_SUCCESS) {
    std::fprintf(stderr, "cuFFT error %d at %s:%d in %s\n",
                 static_cast<int>(status), file, line, call);
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace imgpipe

#define CUDA_CHECK(call) ::imgpipe::CheckCuda((call), #call, __FILE__, __LINE__)
#define CUFFT_CHECK(call) \
  ::imgpipe::CheckCufft((call), #call, __FILE__, __LINE__)
// Checks for errors from the most recent kernel launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

#endif  // SRC_CUDA_CHECK_H_
