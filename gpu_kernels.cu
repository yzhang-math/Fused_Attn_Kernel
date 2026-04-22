#include "attention.h"

// Naive QK^T Kernel
__global__ void qk_kernel(const float* Q, const float* K, float* S) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float tmp = 0.0f;
        for (int k = 0; k < D; ++k) {
            tmp += Q[row * D + k] * K[col * D + k];
        }
        S[row * N + col] = tmp * SCALE;
    }
}

// Naive Softmax + PV Kernel
__global__ void softmax_v_kernel(float* S, const float* V, float* O) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float m = -1e20f;
        for (int j = 0; j < N; ++j) m = fmaxf(m, S[row * N + j]);
        float s = 0.0f;
        for (int j = 0; j < N; ++j) {
            S[row * N + j] = expf(S[row * N + j] - m);
            s += S[row * N + j];
        }
        for (int j = 0; j < D; ++j) {
            float out = 0.0f;
            for (int k = 0; k < N; ++k) {
                out += (S[row * N + k] / s) * V[k * D + j];
            }
            O[row * D + j] = out;
        }
    }
}

void gpu_attention_naive(const float* Q, const float* K, const float* V, float* O) {
    float *d_Q, *d_K, *d_V, *d_S, *d_O;
    cudaMalloc(&d_Q, N * D * sizeof(float));
    cudaMalloc(&d_K, N * D * sizeof(float));
    cudaMalloc(&d_V, N * D * sizeof(float));
    cudaMalloc(&d_S, N * N * sizeof(float)); // Intermediate [cite: 19]
    cudaMalloc(&d_O, N * D * sizeof(float));

    cudaMemcpy(d_Q, Q, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * D * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridS((N + 15) / 16, (N + 15) / 16);
    dim3 blockS(16, 16);
    qk_kernel<<<gridS, blockS>>>(d_Q, d_K, d_S);

    softmax_v_kernel<<<(N + 255) / 256, 256>>>(d_S, d_V, d_O);

    cudaMemcpy(O, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_S); cudaFree(d_O);
}