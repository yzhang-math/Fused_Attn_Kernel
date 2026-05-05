# Mixed-Precision Fused Attention Kernel

This repository contains an implementation of a fused attention kernel using NVIDIA CUDA and Tensor Cores. It progresses from a standard CPU reference to an SRAM-optimized fused kernel utilizing the Online Softmax algorithm.

## File Structure
- `gpu_kernels.cu`: Contains Naive, Fused (SRAM), and WMMA (Tensor Core) kernels.
- `cpu_reference.cpp`: Standard and Online Softmax CPU implementations for verification.
- `final_test.cpp`: Main benchmarking harness for accuracy and end-to-end performance.
- `kernel_timing_test.cpp`: Isolated timing harness using CUDA events to exclude H2D/D2H transfers.
- `generate_roofline.py`: Python script for generating Roofline analysis and memory traffic plots.
- `Makefile`: Configured with optimization flags and architecture support (`sm_80`, `sm_89`, `sm_90`).

## Build Instructions
To build the project in **Release Mode** (optimized for benchmarking):
```bash
make MODE=release