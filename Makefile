# Compilers
NVCC := nvcc
CXX  := g++

# Target Architecture Flags (Ada, Ampere, Hopper)
# sm_89: RTX 4070 Ti Super (Development)
# sm_80: A100 (Benchmarking)
# sm_90: H100 (Benchmarking)
ARCH_FLAGS := -gencode arch=compute_89,code=sm_89 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_90,code=sm_90

# Project Files
TARGET   := attention_proj
SRCS     := wk1.cpp cpu_reference.cpp gpu_kernels.cu
HDRS     := attention.h
OBJS     := $(SRCS:.cpp=.o)
OBJS     := $(OBJS:.cu=.o)
# This automatically handles the object file mapping
OBJS     := $(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SRCS)))

# Base Flags
CXXFLAGS := -std=c++17
NVCCFLAGS := -std=c++17 $(ARCH_FLAGS) --use_fast_math

# Build Modes
ifeq ($(MODE),release)
    # Optimization for Week 3-4 (Benchmarking/Roofline)
    CXXFLAGS += -O3
    NVCCFLAGS += -O3 -lineinfo
else
    # Debug for Week 1-2 (Accuracy/Logic)
    MODE = debug
    CXXFLAGS += -g -O0
    NVCCFLAGS += -g -G
endif

# Rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean