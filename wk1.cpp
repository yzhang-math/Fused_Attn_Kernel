#include "attention.h"
#include <chrono>
#include <iostream>

bool verify_results(const float* ref, const float* test, int size, float tolerance) {
    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_err = std::max(max_err, std::abs(ref[i] - test[i]));
    }
    std::cout << "Max Error: " << max_err << " | ";
    return max_err < tolerance;
}

int main() {
    std::vector<float> Q(N * D, 0.1f), K(N * D, 0.2f), V(N * D, 0.3f);
    std::vector<float> out_std(N * D), out_online(N * D), out_gpu(N * D);

    // Timing Wrapper
    auto benchmark = [&](auto func, float* out, const std::string& label) {
        auto start = std::chrono::high_resolution_clock::now();
        func(Q.data(), K.data(), V.data(), out);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        std::cout << label << " Time: " << diff.count() << " ms" << std::endl;
    };

    benchmark(cpu_attention_standard, out_std.data(), "CPU Standard");
    benchmark(cpu_attention_online, out_online.data(), "CPU Online");
    benchmark(gpu_attention_naive, out_gpu.data(), "GPU Naive");

    // Task 1.4: Accuracy Verification 
    std::cout << "Online Softmax Accuracy: " << (verify_results(out_std.data(), out_online.data(), N * D) ? "PASS" : "FAIL") << std::endl;
    std::cout << "Naive GPU Accuracy: " << (verify_results(out_std.data(), out_gpu.data(), N * D) ? "PASS" : "FAIL") << std::endl;

    return 0;
}