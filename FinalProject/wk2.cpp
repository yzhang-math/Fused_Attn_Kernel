#include "attention.h"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

int main() {
    // 1. Initialize Random Data
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q(N * D), K(N * D), V(N * D);
    std::vector<float> out_std(N * D), out_online(N * D), out_naive(N * D), out_fused(N * D);

    for (int i = 0; i < N * D; ++i) {
        Q[i] = dist(gen);
        K[i] = dist(gen);
        V[i] = dist(gen);
    }

    // 2. Benchmarking Wrapper
    auto benchmark = [&](auto func, float* out, const std::string& label) {
        auto start = std::chrono::high_resolution_clock::now();
        func(Q.data(), K.data(), V.data(), out);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        std::cout << std::setw(20) << std::left << label 
                  << " | Time: " << std::fixed << std::setprecision(3) << diff.count() << " ms" << std::endl;
    };

    std::cout << "--- Performance Benchmarking (N=" << N << ", d=" << D << ") ---" << std::endl;
    
    // CPU References [cite: 16, 17]
    benchmark(cpu_attention_standard, out_std.data(), "CPU Standard");
    benchmark(cpu_attention_online, out_online.data(), "CPU Online");

    // GPU Implementations [cite: 18, 27]
    benchmark(gpu_attention_naive, out_naive.data(), "GPU Naive (Unfused)");
    benchmark(gpu_attention_fused, out_fused.data(), "GPU Fused (SRAM)");

    std::cout << "\n--- Accuracy Verification (Ref: CPU Standard) ---" << std::endl;
    
    // Task 1.4 & 2.5 Accuracy Checks [cite: 20, 27]
    verify_results("Online Softmax CPU", out_std.data(), out_online.data(), N * D);
    verify_results("Naive GPU Kernel", out_std.data(), out_naive.data(), N * D);
    verify_results("Fused GPU Kernel", out_std.data(), out_fused.data(), N * D);

    return 0;
}