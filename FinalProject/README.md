# Mixed-Precision Fused Attention Kernel

This project implements and benchmarks high-performance attention kernels in CUDA, progressing from a CPU reference to:

- a naive GPU kernel,
- an SRAM-optimized fused kernel (Online Softmax),
- and a mixed-precision Tensor Core kernel using `nvcuda::wmma`.

## Prerequisites

- **Hardware**: NVIDIA GPU with Tensor Core support (Ampere/Ada/Hopper recommended)
- **Software**:
  - CUDA Toolkit 11.0+
  - C++ compiler with C++17 support
  - Python 3 with `numpy` and `matplotlib` (for Roofline plots)

## Build

The project is built with the provided `Makefile`.

### Release mode (benchmarking)

```bash
make MODE=release
```

### Debug mode (verification)

```bash
make MODE=debug
```

## Run

### Standard benchmark + correctness check

Runs CPU reference plus naive/fused/WMMA GPU kernels and reports accuracy + end-to-end performance.

```bash
./attention_proj
```

### Kernel-only timing (for Roofline data)

Build the timing harness that excludes host-device transfer overhead:

```bash
make clean
make MODE=release MAIN_SRC=kernel_timing_test.cpp
./attention_proj
```

## Reproduce Roofline Analysis

After collecting timing results, generate performance plots:

```bash
python3 generate_roofline.py
```

This produces:

- `roofline_analysis.png`
- `speedup_comparison.png`
- `memory_analysis.png`

## Cluster Execution (Slurm)

To run in a Slurm-managed environment:

```bash
sbatch job_submit.sh
```

## Numerical Tolerance

Verification uses an absolute tolerance of `1e-4`, suitable for mixed-precision FP16 Tensor Core arithmetic.

## Project File Map

- `gpu_kernels.cu`: Naive, fused (SRAM), and WMMA kernels
- `cpu_reference.cpp`: Standard and Online Softmax CPU references
- `final_test.cpp`: Main benchmark and accuracy harness
- `kernel_timing_test.cpp`: Kernel-only microbenchmark harness
- `generate_roofline.py`: Roofline/speedup/memory analysis plots
- `attention.h`: Shared constants and kernel declarations
- `Makefile`: Build configuration (`sm_80`, `sm_89`, `sm_90`, modes, entry source)