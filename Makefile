# Build file for the CUDA batch image pipeline.
#
#   make                 # build bin/image_pipeline
#   make ARCH=sm_75      # build for a specific GPU (default: nvcc's default)
#   make cpu_test        # build + run the CPU-only unit tests (no GPU needed)
#   make run             # run the pipeline on data/input
#   make clean
#
# The CUDA toolkit is located automatically (nvcc on PATH, /usr/local/cuda,
# or the newest /usr/local/cuda-*). Override with CUDA_PATH=/path/to/cuda.

CUDA_CANDIDATES := $(patsubst %/bin/nvcc,%,$(realpath $(shell command -v nvcc 2>/dev/null))) \
                   $(patsubst %/bin/nvcc,%,$(wildcard /usr/local/cuda/bin/nvcc)) \
                   $(patsubst %/bin/nvcc,%,$(lastword $(sort $(wildcard /usr/local/cuda-*/bin/nvcc))))
CUDA_PATH ?= $(firstword $(CUDA_CANDIDATES) /usr/local/cuda)
NVCC      ?= $(CUDA_PATH)/bin/nvcc
CXX       ?= g++

BIN_DIR   := bin
BUILD_DIR := build
TARGET    := $(BIN_DIR)/image_pipeline

ARCH ?=
ifneq ($(ARCH),)
  ARCH_FLAGS := -arch=$(ARCH)
endif

CXXSTD     := -std=c++17
INCLUDES   := -I. -I$(CUDA_PATH)/include
NVCCFLAGS  := $(CXXSTD) -O3 $(ARCH_FLAGS) $(INCLUDES) -Xcompiler -Wall,-Wextra
CXXFLAGS   := $(CXXSTD) -O3 -Wall -Wextra -I.
LDFLAGS    := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib
LDLIBS     := -lcufft -lcudart -lstdc++fs

CU_SRCS := src/kernels.cu src/gpu_pipeline.cu
CC_SRCS := src/main.cc src/image.cc src/cpu_reference.cc src/cli_options.cc
OBJS    := $(CU_SRCS:src/%.cu=$(BUILD_DIR)/%.o) \
           $(CC_SRCS:src/%.cc=$(BUILD_DIR)/%.o)

.PHONY: all clean run cpu_test

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(ARCH_FLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.cu src/*.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cc src/*.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET) --input_dir=data/input --output_dir=results/output

# CPU-only tests: compiled with the host compiler, so they run anywhere.
cpu_test: | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tests/cpu_reference_test.cc src/cpu_reference.cc \
	    src/image.cc src/cli_options.cc -o $(BIN_DIR)/cpu_reference_test
	./$(BIN_DIR)/cpu_reference_test

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)
