#include "attention.h"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// Helper to check CUDA errors
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

int main() {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Host memory
    std::vector<float> Q_host(N * D), K_host(N * D), V_host(N * D);
    std::vector<float> out_cpu(N * D), out_gpu(N * D);

    for (int i = 0; i < N * D; ++i) {
        Q_host[i] = dist(gen);
        K_host[i] = dist(gen);
        V_host[i] = dist(gen);
    }

    // Device memory
    float *Q_dev, *K_dev, *V_dev, *out_dev;
    CUDA_CHECK(cudaMalloc(&Q_dev, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&K_dev, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&V_dev, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out_dev, N * D * sizeof(float)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     KERNEL-ONLY TIMING (Excluding H2D/D2H Transfers)        ║" << std::endl;
    std::cout << "║     (N=" << std::setw(4) << N << ", D=" << std::setw(2) << D << ")                              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

    // CPU baseline (host-side timing includes everything)
    std::cout << "\n--- CPU Baseline (Full Timing) ---" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_attention_standard(Q_host.data(), K_host.data(), V_host.data(), out_cpu.data());
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_diff = cpu_end - cpu_start;
    std::cout << std::setw(25) << std::left << "CPU Standard" 
              << " | " << std::fixed << std::setprecision(3) << std::setw(10) << cpu_diff.count() << " ms" << std::endl;

    std::cout << "\n--- GPU Kernels (Kernel-Only Timing via CUDA Events) ---" << std::endl;
    std::cout << "Note: H2D and D2H transfers not included in timing\n" << std::endl;

    // GPU Naive - with H2D, kernel, D2H separation
    std::cout << "GPU Naive (Unfused):" << std::endl;
    CUDA_CHECK(cudaMemcpy(Q_dev, Q_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K_dev, K_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V_dev, V_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    gpu_attention_naive(Q_dev, K_dev, V_dev, out_dev);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float naive_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    CUDA_CHECK(cudaMemcpy(out_gpu.data(), out_dev, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "  Kernel Time:    " << std::fixed << std::setprecision(3) << std::setw(10) << naive_time << " ms" << std::endl;
    verify_results("GPU Naive", out_cpu.data(), out_gpu.data(), N * D);

    // GPU Fused
    std::cout << "\nGPU Fused (SRAM):" << std::endl;
    CUDA_CHECK(cudaMemcpy(Q_dev, Q_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K_dev, K_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V_dev, V_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    gpu_attention_fused(Q_dev, K_dev, V_dev, out_dev);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float fused_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&fused_time, start, stop));
    CUDA_CHECK(cudaMemcpy(out_gpu.data(), out_dev, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "  Kernel Time:    " << std::fixed << std::setprecision(3) << std::setw(10) << fused_time << " ms" << std::endl;
    verify_results("GPU Fused", out_cpu.data(), out_gpu.data(), N * D);

    // GPU WMMA
    std::cout << "\nGPU WMMA (FP16):" << std::endl;
    CUDA_CHECK(cudaMemcpy(Q_dev, Q_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K_dev, K_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V_dev, V_host.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    gpu_attention_wmma(Q_dev, K_dev, V_dev, out_dev);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float wmma_time = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&wmma_time, start, stop));
    CUDA_CHECK(cudaMemcpy(out_gpu.data(), out_dev, N * D * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "  Kernel Time:    " << std::fixed << std::setprecision(3) << std::setw(10) << wmma_time << " ms" << std::endl;
    verify_results("GPU WMMA", out_cpu.data(), out_gpu.data(), N * D);

    // Summary
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                   Kernel-Only Performance Summary              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::setw(25) << std::left << "CPU Standard" 
              << " | " << std::fixed << std::setprecision(3) << std::setw(10) << cpu_diff.count() << " ms | baseline" << std::endl;
    std::cout << std::setw(25) << std::left << "GPU Naive" 
              << " | " << std::fixed << std::setprecision(3) << std::setw(10) << naive_time << " ms | " 
              << std::fixed << std::setprecision(1) << cpu_diff.count()/naive_time << "x speedup" << std::endl;
    std::cout << std::setw(25) << std::left << "GPU Fused" 
              << " | " << std::fixed << std::setprecision(3) << std::setw(10) << fused_time << " ms | "
              << std::fixed << std::setprecision(1) << cpu_diff.count()/fused_time << "x speedup" << std::endl;
    std::cout << std::setw(25) << std::left << "GPU WMMA" 
              << " | " << std::fixed << std::setprecision(3) << std::setw(10) << wmma_time << " ms | "
              << std::fixed << std::setprecision(1) << cpu_diff.count()/wmma_time << "x speedup" << std::endl;

    // TFLOPS calculation
    float total_flops = 4.0f * N * N * D / 1e12;  // in TFLOPS
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    Actual Kernel TFLOPS                       ║" << std::endl;
    std::cout << "║        (Kernel execution only, excluding transfers)            ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::setw(25) << std::left << "GPU Naive" 
              << " | " << std::fixed << std::setprecision(3) << (total_flops/(naive_time/1000.0)) << " TFLOPS" << std::endl;
    std::cout << std::setw(25) << std::left << "GPU Fused" 
              << " | " << std::fixed << std::setprecision(3) << (total_flops/(fused_time/1000.0)) << " TFLOPS" << std::endl;
    std::cout << std::setw(25) << std::left << "GPU WMMA" 
              << " | " << std::fixed << std::setprecision(3) << (total_flops/(wmma_time/1000.0)) << " TFLOPS" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(Q_dev));
    CUDA_CHECK(cudaFree(K_dev));
    CUDA_CHECK(cudaFree(V_dev));
    CUDA_CHECK(cudaFree(out_dev));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
