#include "attention.h"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

// Kernel-only benchmark wrapper (excludes H2D/D2H transfers)
struct KernelBenchmark {
    float* d_Q;
    float* d_K;
    float* d_V;
    float* d_O;
    float* h_O;
    int size_Q, size_K, size_V, size_O;

    KernelBenchmark(int N, int D) : size_Q(N*D), size_K(N*D), size_V(N*D), size_O(N*D) {
        cudaMalloc(&d_Q, size_Q * sizeof(float));
        cudaMalloc(&d_K, size_K * sizeof(float));
        cudaMalloc(&d_V, size_V * sizeof(float));
        cudaMalloc(&d_O, size_O * sizeof(float));
        h_O = new float[size_O];
    }

    ~KernelBenchmark() {
        cudaFree(d_Q);
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_O);
        delete[] h_O;
    }

    void upload_data(const float* Q, const float* K, const float* V) {
        cudaMemcpy(d_Q, Q, size_Q * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K, size_K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V, size_V * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

    void download_data() {
        cudaMemcpy(h_O, d_O, size_O * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    double benchmark_naive_kernel(int iterations = 3) {
        // Warmup
        gpu_attention_naive_kernel(d_Q, d_K, d_V, d_O);
        cudaDeviceSynchronize();

        // Timed iterations
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        double total_time = 0.0;
        for (int i = 0; i < iterations; ++i) {
            cudaEventRecord(start);
            gpu_attention_naive_kernel(d_Q, d_K, d_V, d_O);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            total_time += elapsed_ms;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return total_time / iterations;
    }

    double benchmark_fused_kernel(int iterations = 3) {
        gpu_attention_fused_kernel(d_Q, d_K, d_V, d_O);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        double total_time = 0.0;
        for (int i = 0; i < iterations; ++i) {
            cudaEventRecord(start);
            gpu_attention_fused_kernel(d_Q, d_K, d_V, d_O);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            total_time += elapsed_ms;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return total_time / iterations;
    }

    double benchmark_wmma_kernel(int iterations = 3) {
        gpu_attention_wmma_kernel(d_Q, d_K, d_V, d_O);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        double total_time = 0.0;
        for (int i = 0; i < iterations; ++i) {
            cudaEventRecord(start);
            gpu_attention_wmma_kernel(d_Q, d_K, d_V, d_O);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            total_time += elapsed_ms;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return total_time / iterations;
    }
};

// Forward declarations for raw kernel launchers
extern "C" {
    void gpu_attention_naive_kernel(const float* Q, const float* K, const float* V, float* O);
    void gpu_attention_fused_kernel(const float* Q, const float* K, const float* V, float* O);
    void gpu_attention_wmma_kernel(const float* Q, const float* K, const float* V, float* O);
}

int main() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q(N * D), K(N * D), V(N * D);
    std::vector<float> out_std(N * D);

    for (int i = 0; i < N * D; ++i) {
        Q[i] = dist(gen);
        K[i] = dist(gen);
        V[i] = dist(gen);
    }

    // CPU reference (for accuracy verification)
    cpu_attention_standard(Q.data(), K.data(), V.data(), out_std.data());

    // Kernel-only benchmarking
    KernelBenchmark bench(N, D);
    bench.upload_data(Q.data(), K.data(), V.data());

    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  KERNEL-ONLY PERFORMANCE BENCHMARKING                     ║" << std::endl;
    std::cout << "║  (Excluding host↔device data transfer)                    ║" << std::endl;
    std::cout << "║  (N=" << std::setw(4) << N << ", D=" << std::setw(2) << D << ")                             ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    double naive_time = bench.benchmark_naive_kernel(5);
    double fused_time = bench.benchmark_fused_kernel(5);
    double wmma_time = bench.benchmark_wmma_kernel(5);

    std::cout << "\n--- Kernel Execution Times (ms) ---" << std::endl;
    std::cout << std::setw(25) << std::left << "Naive GPU Kernel"
              << " | " << std::fixed << std::setprecision(3) << naive_time << " ms" << std::endl;
    std::cout << std::setw(25) << std::left << "Fused GPU Kernel"
              << " | " << std::fixed << std::setprecision(3) << fused_time << " ms" << std::endl;
    std::cout << std::setw(25) << std::left << "WMMA GPU Kernel"
              << " | " << std::fixed << std::setprecision(3) << wmma_time << " ms" << std::endl;

    std::cout << "\n--- Speedup (vs Naive) ---" << std::endl;
    std::cout << std::setw(25) << std::left << "Fused Kernel"
              << " | " << std::fixed << std::setprecision(2) << (naive_time / fused_time) << "x" << std::endl;
    std::cout << std::setw(25) << std::left << "WMMA Kernel"
              << " | " << std::fixed << std::setprecision(2) << (naive_time / wmma_time) << "x" << std::endl;

    // Compute throughput (TFLOPS)
    // Total FLOPS: N^2 * D * 3 (approx for attention: QK^T + softmax + PV)
    double total_flops = (double)N * N * D * 3.0;
    double naive_tflops = (total_flops / 1e12) / (naive_time / 1000.0);
    double fused_tflops = (total_flops / 1e12) / (fused_time / 1000.0);
    double wmma_tflops = (total_flops / 1e12) / (wmma_time / 1000.0);

    std::cout << "\n--- Compute Throughput (TFLOPS) ---" << std::endl;
    std::cout << std::setw(25) << std::left << "Naive GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << naive_tflops << " TFLOPS" << std::endl;
    std::cout << std::setw(25) << std::left << "Fused GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << fused_tflops << " TFLOPS" << std::endl;
    std::cout << std::setw(25) << std::left << "WMMA GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << wmma_tflops << " TFLOPS" << std::endl;

    // Memory throughput estimation
    // Naive: Reads Q,K,V (once each) + S (R/W) + writes O = ~4.25 MB per iteration
    // Fused: Reads Q,K,V + writes O = ~1 MB per iteration (S stays in shared memory)
    double naive_memory_gb = (4.25e-9) / (naive_time / 1000.0);
    double fused_memory_gb = (1.0e-9) / (fused_time / 1000.0);
    double wmma_memory_gb = (0.75e-9) / (wmma_time / 1000.0);

    std::cout << "\n--- Memory Throughput (GB/s, estimated) ---" << std::endl;
    std::cout << std::setw(25) << std::left << "Naive GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << naive_memory_gb << " GB/s" << std::endl;
    std::cout << std::setw(25) << std::left << "Fused GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << fused_memory_gb << " GB/s" << std::endl;
    std::cout << std::setw(25) << std::left << "WMMA GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << wmma_memory_gb << " GB/s" << std::endl;

    // Arithmetic Intensity
    double naive_ai = total_flops / (4.25e6);
    double fused_ai = total_flops / (1.0e6);
    double wmma_ai = total_flops / (0.75e6);

    std::cout << "\n--- Arithmetic Intensity (FLOP/byte) ---" << std::endl;
    std::cout << std::setw(25) << std::left << "Naive GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << naive_ai << " FLOP/byte" << std::endl;
    std::cout << std::setw(25) << std::left << "Fused GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << fused_ai << " FLOP/byte" << std::endl;
    std::cout << std::setw(25) << std::left << "WMMA GPU Kernel"
              << " | " << std::fixed << std::setprecision(2) << wmma_ai << " FLOP/byte" << std::endl;

    std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Note: This measurement excludes host↔device transfers    ║" << std::endl;
    std::cout << "║  For accurate memory throughput, use: ncu --set full      ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
