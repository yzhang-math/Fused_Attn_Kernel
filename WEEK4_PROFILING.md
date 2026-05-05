# Week 4: Benchmarking, Roofline Analysis, and Reporting

## Task 4.1: Profile with NVIDIA Nsight Compute (ncu)

### Profiling Command Examples

For the Naive Kernel:
```bash
ncu --set full -o naive_profile.ncu-rep \
  ./attention_proj -kernel naive
```

For the Fused Kernel:
```bash
ncu --set full -o fused_profile.ncu-rep \
  ./attention_proj -kernel fused
```

For the WMMA Kernel:
```bash
ncu --set full -o wmma_profile.ncu-rep \
  ./attention_proj -kernel wmma
```

### Metrics to Extract

Key metrics for roofline analysis:
- **Memory Throughput**: Global Memory Throughput (GB/s)
- **Compute Throughput**: Floating Point Operations (TFLOPS)
- **L1 Cache Hit Rate**: Data locality indicator
- **L2 Cache Hit Rate**: Memory hierarchy effectiveness
- **Shared Memory Efficiency**: Bank conflict indicators
- **Warp Efficiency**: Instruction-level parallelism
- **SM Utilization**: GPU occupancy

## Task 4.2: Extract Throughput Metrics

### Naive Kernel Arithmetic

**Operation Count:**
- QK^T computation: N × N × D multiply-add = 2 × N × N × D FLOPS
- Softmax + PV: N × (N + N×D) = N×(N + N×D) FLOPS
- Total: ~N² × D × 3 FLOPS (dominant term)

For N=1024, D=64:
- Total FLOPS: 1024² × 64 × 3 ≈ 201.3 GFLOPS (theoretical)

**Memory Bandwidth:**
- Input: 3 × N × D × sizeof(float) = 3 × 1024 × 64 × 4 ≈ 786 KB
- Intermediate S: N × N × sizeof(float) = 1024² × 4 ≈ 4 MB
- Output: N × D × sizeof(float) = 1024 × 64 × 4 ≈ 256 KB
- Total: ~4.25 MB per execution

**Arithmetic Intensity (AI):**
- AI = FLOPS / Memory (in bytes) = 201.3G / (4.25M) ≈ 47 FLOP/byte
- This is quite high (compute-bound for modern GPUs)

### Fused Kernel Arithmetic

**Operation Count (Same as Naive):**
- ~N² × D × 3 FLOPS

**Memory Bandwidth (Reduced):**
- Input: 3 × N × D × sizeof(float) ≈ 786 KB
- Intermediate S: N × N × sizeof(float) ≈ 4 MB (kept in shared memory for reuse)
- Output: N × D × sizeof(float) ≈ 256 KB
- Global memory: ~5.04 MB

**Arithmetic Intensity:**
- Higher due to better register and shared memory usage
- Expected: 50-60 FLOP/byte

### WMMA Kernel Arithmetic

**Operation Count:**
- Same FLOPS as naive (201.3 GFLOPS)

**Memory Bandwidth:**
- Same as fused: ~5.04 MB

**Arithmetic Intensity:**
- Comparable to fused with FP16 benefits
- FP16 uses 2x less memory: ~2.5 MB
- Expected AI: 100-120 FLOP/byte

## Task 4.3: Roofline Model Construction

### GPU Hardware Specs (for reference)

**NVIDIA RTX 4090 (Ada Architecture - Development GPU):**
- Peak FP32 Throughput: 82.6 TFLOPS
- Peak Memory Bandwidth: 936 GB/s
- Peak FP32 Roofline Knee: 82.6T / 936G ≈ 88 FLOP/byte

**NVIDIA A100 (Ampere - Benchmarking):**
- Peak FP32 Throughput: 312 TFLOPS
- Peak Memory Bandwidth: 2.039 TB/s
- Peak FP32 Roofline Knee: 312T / 2039G ≈ 153 FLOP/byte

**NVIDIA H100 (Hopper - Benchmarking):**
- Peak FP32 Throughput: 756 TFLOPS
- Peak FP32 Memory Bandwidth: 3.35 TB/s
- Peak FP32 Roofline Knee: 756T / 3350G ≈ 226 FLOP/byte

### Roofline Analysis Expected Results

For our attention kernels with AI ≈ 50 FLOP/byte:

**RTX 4090:**
- Roofline Knee: 88 FLOP/byte
- Our AI: 50 FLOP/byte → Memory-Bound
- Expected Throughput: Min(82.6T, 50 × 936G) ≈ 46.8 TFLOPS

**A100:**
- Roofline Knee: 153 FLOP/byte
- Our AI: 50 FLOP/byte → Memory-Bound
- Expected Throughput: Min(312T, 50 × 2039G) ≈ 101.9 TFLOPS

**H100:**
- Roofline Knee: 226 FLOP/byte
- Our AI: 50 FLOP/byte → Memory-Bound
- Expected Throughput: Min(756T, 50 × 3350G) ≈ 167.5 TFLOPS

### Impact of Fused Kernel

By improving shared memory reuse and reducing global memory traffic:
- Reduces memory requirement by ~15-20%
- Increases arithmetic intensity slightly
- Shifts from memory-bound towards compute-bound on higher-end GPUs

## Task 4.4: Final Report Sections

### Section 1: Online Softmax Mathematics

**Standard Softmax:**
```
Attention(Q,K,V) = softmax(QK^T/√d)V
```

**Online Softmax (Incremental Computation):**
For each row i of output:
```
m_j = max(m_{j-1}, qk_j)
s_j = s_{j-1} × exp(m_{j-1} - m_j) + exp(qk_j - m_j)
o_j = o_{j-1} × exp(m_{j-1} - m_j) + exp(qk_j - m_j) × v_j

Final: O_i = o_n / s_n
```

**Benefits:**
- Numerical stability through rescaling
- Single pass through K,V eliminating intermediate N×N matrix storage
- Natural fusion with tiled computation

### Section 2: Shared Memory Bank Conflict Resolution

**Problem:**
- Half-warp (16 threads) accesses 16 consecutive floats
- Causes bank conflicts if stride = 1 (all threads access different banks but at same cycle)

**Solution:**
- Padding: Add extra column to shared memory
- BR×(D+4) instead of BR×D
- Eliminates stride-1 conflicts in dot product

**Impact:**
- ~5-10% performance improvement on high-occupancy kernels
- Critical for 64-wide D dimension (matches warp size)

### Section 3: Hardware Profiling Results

(To be filled with actual ncu output)

### Section 4: Performance Summary

Speedup comparisons and architectural insights.

