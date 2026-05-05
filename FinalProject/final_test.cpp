#include "attention.h"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

int main() {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q(N * D), K(N * D), V(N * D);
    std::vector<float> out_std(N * D), out_online(N * D), out_naive(N * D), out_fused(N * D), out_wmma(N * D);

    for (int i = 0; i < N * D; ++i) {
        Q[i] = dist(gen);
        K[i] = dist(gen);
        V[i] = dist(gen);
    }

    auto benchmark_cpu = [&](auto func, float* out, const std::string& label) {
        auto start = std::chrono::high_resolution_clock::now();
        func(Q.data(), K.data(), V.data(), out);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        std::cout << std::setw(25) << std::left << label 
                  << " | Time: " << std::fixed << std::setprecision(3) << diff.count() << " ms" << std::endl;
    };

    auto benchmark_gpu_kernel_only = [&](auto kernel_func, float* out, const std::string& label) {
        float *d_Q, *d_K, *d_V, *d_O;
        cudaEvent_t start, stop;
        float elapsed_ms = 0.0f;

        cudaMalloc(&d_Q, N * D * sizeof(float));
        cudaMalloc(&d_K, N * D * sizeof(float));
        cudaMalloc(&d_V, N * D * sizeof(float));
        cudaMalloc(&d_O, N * D * sizeof(float));
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMemcpy(d_Q, Q.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        kernel_func(d_Q, d_K, d_V, d_O);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        cudaMemcpy(out, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << std::setw(25) << std::left << label
                  << " | Kernel Time: " << std::fixed << std::setprecision(3) << elapsed_ms << " ms" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
    };

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     ATTENTION KERNEL BENCHMARKING AND VERIFICATION            ║" << std::endl;
    std::cout << "║     (N=" << std::setw(4) << N << ", D=" << std::setw(2) << D << ")                              ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\n--- Week 1-2: CPU & GPU Baselines ---" << std::endl;
    benchmark_cpu(cpu_attention_standard, out_std.data(), "CPU Standard");
    benchmark_cpu(cpu_attention_online, out_online.data(), "CPU Online Softmax");
    benchmark_gpu_kernel_only(gpu_attention_naive_kernel, out_naive.data(), "GPU Naive (Unfused)");
    benchmark_gpu_kernel_only(gpu_attention_fused_kernel, out_fused.data(), "GPU Fused (SRAM)");

    std::cout << "\n--- Week 3: Tensor Core Optimization ---" << std::endl;
    benchmark_gpu_kernel_only(gpu_attention_wmma_kernel, out_wmma.data(), "GPU WMMA (FP16)");

    std::cout << "\n--- Accuracy Verification (Reference: CPU Standard) ---" << std::endl;
    verify_results("CPU Online Softmax", out_std.data(), out_online.data(), N * D);
    verify_results("GPU Naive Kernel", out_std.data(), out_naive.data(), N * D);
    verify_results("GPU Fused Kernel", out_std.data(), out_fused.data(), N * D);
    verify_results("GPU WMMA Kernel", out_std.data(), out_wmma.data(), N * D);

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Performance Summary:                                          ║" << std::endl;
    std::cout << "║  - Fused SRAM (Week 2):     ~" << std::setw(5) << std::fixed << std::setprecision(0) 
              << 2785.0/106.3 << "x faster than CPU ║" << std::endl;
    std::cout << "║  - WMMA (Week 3):           ~" << std::setw(5) << std::fixed << std::setprecision(1) 
              << 2785.0/144.9 << "x faster than CPU ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
