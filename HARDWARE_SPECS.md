# Hardware Specifications & Performance Analysis

## Actual GPU Hardware

### RTX 3090 (Ampere)
- **CUDA Cores**: 10,496
- **Boost Clock**: 1.86 GHz
- **FP32 Peak**: 39.5 TFLOPS
- **Memory**: 24GB GDDR6X
- **Native Bandwidth**: 936 GB/s (PCIe 4.0 x16)
- **Actual Bandwidth** (PCIe 4.0 x8): **468 GB/s** ⚡

### RTX 4070 Ti Super (Ada)
- **CUDA Cores**: 12,800
- **Boost Clock**: 2.61 GHz
- **FP32 Peak**: 66.6 TFLOPS
- **Memory**: 16GB GDDR6X
- **Native Bandwidth**: 576 GB/s (PCIe 4.0 x16)
- **Actual Bandwidth** (PCIe 4.0 x8): **288 GB/s** ⚡

## PCIe 4.0 x8 Bandwidth Limitation

**PCIe Gen 4.0 Specifications:**
- Gen 4.0 x16: 32 GB/s per direction (64 GB/s bidirectional)
- Gen 4.0 x8: **16 GB/s per direction** (32 GB/s bidirectional)

**Impact on GPU Bandwidth:**
- Both GPUs have native bandwidth > PCIe 4.0 x8 limit
- Memory bandwidth = min(GPU native, PCIe bandwidth)
- PCIe 4.0 x8 becomes the bottleneck for both GPUs
- Affects H2D/D2H transfers but NOT kernel computation

## Roofline Analysis (Updated)

### RTX 3090 (PCIe 4.0 x8)

**Knee Point Calculation:**
- Peak TFLOPS: 39.5
- Memory Bandwidth: 468 GB/s
- Arithmetic Intensity at Knee: 39.5 × 10¹² / (468 × 10⁹) = **84.4 FLOP/byte**

**Kernel Performance Predictions:**
| Kernel | AI (FLOP/byte) | Status | Expected TFLOPS |
|--------|----------------|--------|-----------------|
| Naive  | 47 | Memory-bound | 47 × 468/1000 = **22 TFLOPS** |
| Fused  | 200 | Compute-bound | **39.5 TFLOPS** (limited by peak) |
| WMMA   | 270 | Compute-bound | **39.5 TFLOPS** (limited by peak) |

**Speedup Prediction:**
- Naive: ~20 ms
- Fused: ~13 ms
- Expected Speedup: **~1.5x** (over naive GPU)

### RTX 4070 Ti Super (PCIe 4.0 x8)

**Knee Point Calculation:**
- Peak TFLOPS: 66.6
- Memory Bandwidth: 288 GB/s
- Arithmetic Intensity at Knee: 66.6 × 10¹² / (288 × 10⁹) = **231 FLOP/byte**

**Kernel Performance Predictions:**
| Kernel | AI (FLOP/byte) | Status | Expected TFLOPS |
|--------|----------------|--------|-----------------|
| Naive  | 47 | Memory-bound | 47 × 288/1000 = **13.5 TFLOPS** |
| Fused  | 200 | Memory-bound | 200 × 288/1000 = **57.6 TFLOPS** |
| WMMA   | 270 | Compute-bound | **66.6 TFLOPS** (limited by peak) |

**Speedup Prediction:**
- Naive: ~32 ms
- Fused: ~4 ms
- Expected Speedup: **~8x** (over naive GPU)

## Key Observations

### PCIe 4.0 x8 Impact

1. **Significantly Lower Bandwidth**
   - RTX 3090: 468 GB/s → 50% loss from PCIe limitation
   - RTX 4070 Ti Super: 288 GB/s → 50% loss from PCIe limitation

2. **RTX 3090 Characteristics**
   - Lower peak compute (39.5 TFLOPS)
   - Easier to saturate (low knee point: 84 FLOP/byte)
   - All three kernels reach compute-bound state

3. **RTX 4070 Ti Super Characteristics**
   - Higher peak compute (66.6 TFLOPS)
   - Harder to saturate (high knee point: 231 FLOP/byte)
   - Only WMMA kernel reaches compute-bound
   - Fused kernel remains memory-bound

### Comparison to Original Analysis

| Metric | RTX 4090 | RTX 3090 | 4070 Ti Super |
|--------|----------|----------|---------------|
| Peak TFLOPS | 82.6 | 39.5 | 66.6 |
| Memory BW (GB/s) | 936 | 468 | 288 |
| Fused Expected TFLOPS | 82.6 | 39.5 | 57.6 |
| Speedup Potential | 24.9x | ~1.5x | ~8x |

## Timing Measurements

### Expected Runtime (N=1024, D=64)

**RTX 3090 Estimate:**
```
CPU Standard:  2,635 ms (baseline)
GPU Naive:     ~180-200 ms (15x speedup)
GPU Fused:     ~130-150 ms (18-20x speedup)
GPU WMMA:      ~150-170 ms (15-18x speedup)
```

**RTX 4070 Ti Super Estimate:**
```
CPU Standard:  2,635 ms (baseline)
GPU Naive:     ~300-350 ms (7-9x speedup)
GPU Fused:     ~100-120 ms (22-26x speedup)
GPU WMMA:      ~80-100 ms (26-33x speedup)
```

## Profiling Recommendations

### Using NVIDIA Nsight Compute

For accurate measurements on actual hardware:

```bash
# RTX 3090 profiling
ncu -k "qk_kernel|softmax_v_kernel" --set full ./attention_proj

# RTX 4070 Ti Super profiling
ncu -k "fused_attention_v1" --set full ./attention_proj

# WMMA kernel profiling
ncu -k "fused_attention_wmma" --set full ./attention_proj
```

### Metrics to Extract

- **sm__throughput.avg.pct_of_peak_sustained_active** (% of peak)
- **dram__throughput.avg.pct_of_peak_sustained_active** (memory %)
- **dram__bytes_read.sum + dram__bytes_write.sum** (total traffic)
- **smsp__inst_executed.sum** (total instructions)

### PCIe Bandwidth Monitoring

```bash
# Monitor PCIe Gen 4.0 x8 usage
nvidia-smi dmon -s pcm  # Shows PCIe throughput

# Expected ceiling: 32 GB/s bidirectional (16 GB/s per direction)
```

## Conclusions

1. **PCIe 4.0 x8 is limiting memory bandwidth**, but kernel compute efficiency still matters
2. **RTX 3090** can achieve compute-bound state more easily (lower knee point)
3. **RTX 4070 Ti Super** requires higher AI to reach compute-bound (harder to achieve)
4. **Fused kernel** is critical for 4070 Ti Super to maintain speedup despite bandwidth limits
5. **WMMA kernel** can overcome PCIe limitation on 4070 Ti Super through higher AI (270)

