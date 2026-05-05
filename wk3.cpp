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
    std::vector<float> out_std(N * D), out_wmma(N * D);

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

    std::cout << "--- Week 3: Tensor Core Integration (N=" << N << ", d=" << D << ") ---" << std::endl;

    // CPU Reference [cite: 16]
    benchmark(cpu_attention_standard, out_std.data(), "CPU Standard");

    // GPU WMMA Implementation [cite: Week 3]
    benchmark(gpu_attention_wmma, out_wmma.data(), "GPU WMMA Tensor Core");

    std::cout << "\n--- Accuracy Verification (Ref: CPU Standard) ---" << std::endl;

    // Task 3.5 Accuracy Check [cite: Week 3]
    verify_results("WMMA Tensor Core", out_std.data(), out_wmma.data(), N * D);

    return 0;
}
