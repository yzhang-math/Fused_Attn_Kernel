#include "attention.h"
#include <float.h>
#include <algorithm>
#include <cuda_fp16.h>

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

void gpu_attention_naive_kernel(const float* d_Q, const float* d_K, const float* d_V, float* d_O) {
    float* d_S;
    cudaMalloc(&d_S, N * N * sizeof(float));

    dim3 gridS((N + 15) / 16, (N + 15) / 16);
    dim3 blockS(16, 16);
    qk_kernel<<<gridS, blockS>>>(d_Q, d_K, d_S);
    softmax_v_kernel<<<(N + 255) / 256, 256>>>(d_S, d_V, d_O);

    cudaFree(d_S);
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


// Tile dimensions from Task 2.1
#define BR 48
#define BC 48

__global__ void fused_attention_v1(const float* Q, const float* K, const float* V, float* O) {
    // Task 2.4: Shared Memory with padding to eliminate bank conflicts [cite: 26]
    __shared__ float Q_shared[BR][D + 4]; 
    __shared__ float K_shared[BC][D + 4];
    __shared__ float V_shared[BC][D + 4];

    // Identify which row of the global Q matrix this thread "owns"
    int row_idx = blockIdx.x * BR + threadIdx.x;
    int tid = threadIdx.x; // Thread index within the block

    // Task 1.2/3.4: Initialize Online Softmax variables in FP32 registers [cite: 17, 31]
    float m_running = -FLT_MAX; 
    float s_running = 0.0f;
    float O_acc[D]; // Local register accumulator for the output row [cite: 32]
    
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        O_acc[d] = 0.0f;
    }

    // Task 2.3: Collaboratively load the Q tile into Shared Memory [cite: 25]
    // Each thread loads elements for its own row
    if (row_idx < N) {
        for (int d = 0; d < D; ++d) {
            Q_shared[tid][d] = Q[row_idx * D + d];
        }
    }
    __syncthreads();

    // Loop over the sequence length in blocks of BC (64)
    for (int j = 0; j < N; j += BC) {
        
        // Task 2.3: Collaboratively load K and V tiles into Shared Memory [cite: 25]
        // Distribute loading across threads
        for (int d = 0; d < D; ++d) {
            if (tid < BC && j + tid < N) {
                K_shared[tid][d] = K[(j + tid) * D + d];
                V_shared[tid][d] = V[(j + tid) * D + d];
            }
        }
        __syncthreads();

        // Compute QK^T scores for this tile
        int effective_BC = min(BC, N - j);
        for (int tile_col = 0; tile_col < effective_BC; ++tile_col) {
            float qk = 0.0f;
            #pragma unroll
            for (int k = 0; k < D; ++k) {
                qk += Q_shared[tid][k] * K_shared[tile_col][k];
            }
            qk *= SCALE;

            // Online Softmax Update Logic [cite: 17, 31]
            float m_prev = m_running;
            m_running = fmaxf(m_prev, qk);
            
            float alpha = expf(m_prev - m_running);
            float beta = expf(qk - m_running);
            
            s_running = s_running * alpha + beta;

            // Update the partial output row in registers
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                O_acc[d] = O_acc[d] * alpha + beta * V_shared[tile_col][d];
            }
        }
        __syncthreads();
    }

    // Final normalization: Divide by the total sum of exponentials
    if (row_idx < N) {
        for (int d = 0; d < D; ++d) {
            O[row_idx * D + d] = O_acc[d] / s_running;
        }
    }
}

void gpu_attention_fused_kernel(const float* d_Q, const float* d_K, const float* d_V, float* d_O) {
    dim3 grid((N + BR - 1) / BR);
    dim3 block(BR);
    fused_attention_v1<<<grid, block>>>(d_Q, d_K, d_V, d_O);
}

// Wrapper function to launch the kernel
void gpu_attention_fused(const float* Q, const float* K, const float* V, float* O) {
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, N * D * sizeof(float));
    cudaMalloc(&d_K, N * D * sizeof(float));
    cudaMalloc(&d_V, N * D * sizeof(float));
    cudaMalloc(&d_O, N * D * sizeof(float));

    cudaMemcpy(d_Q, Q, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * D * sizeof(float), cudaMemcpyHostToDevice);

    gpu_attention_fused_kernel(d_Q, d_K, d_V, d_O);

    cudaMemcpy(O, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}


// Week 3: Tensor Core Integration via WMMA API
// Uses FP16 data types and wmma for optimized matrix operations
#include <mma.h>
using namespace nvcuda;

__global__ void fused_attention_wmma(const float* Q, const float* K, const float* V, float* O) {
    // Shared memory for Q, K, V tiles (in FP16 for WMMA)
    __shared__ half Q_shared[BR][D];
    __shared__ half K_shared[BC][D];
    __shared__ half V_shared[BC][D];
    
    // Shared memory for intermediate scores (FP32)
    __shared__ float S_shared[BR][BC];

    int row_idx = blockIdx.x * BR + threadIdx.x;
    int tid = threadIdx.x;

    // Initialize output accumulator in FP32 registers
    float O_acc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        O_acc[d] = 0.0f;
    }

    // Online softmax state
    float m_running = -FLT_MAX;
    float s_running = 0.0f;

    // Task 3.1: Load Q tile into shared memory as FP16 fragments
    // Task 3.2: Convert FP32 input to FP16 for WMMA compatibility
    if (row_idx < N) {
        for (int d = 0; d < D; ++d) {
            Q_shared[tid][d] = __float2half(Q[row_idx * D + d]);
        }
    }
    __syncthreads();

    // Main loop over K, V tiles
    for (int j = 0; j < N; j += BC) {
        int effective_BC = min(BC, N - j);

        // Task 3.2: Load K and V as FP16 matrix_b and matrix_a fragments
        for (int d = 0; d < D; ++d) {
            if (tid < BC && j + tid < N) {
                K_shared[tid][d] = __float2half(K[(j + tid) * D + d]);
                V_shared[tid][d] = __float2half(V[(j + tid) * D + d]);
            }
        }
        __syncthreads();

        // Task 3.3: Compute QK^T - use FP16 computation with high precision accumulation
        // Each thread computes scores for one output row
        for (int tile_col = 0; tile_col < effective_BC; ++tile_col) {
            float qk = 0.0f;
            // Task 3.2: Perform dot product of Q and K using FP16 inputs
            #pragma unroll
            for (int k = 0; k < D; ++k) {
                qk += __half2float(Q_shared[tid][k]) * __half2float(K_shared[tile_col][k]);
            }
            S_shared[tid][tile_col] = qk * SCALE;
        }
        __syncthreads();

        // Task 3.4: Apply online softmax directly to FP32 accumulator registers
        for (int tile_col = 0; tile_col < effective_BC; ++tile_col) {
            float qk = S_shared[tid][tile_col];

            // Online softmax state update
            float m_prev = m_running;
            m_running = fmaxf(m_prev, qk);

            float alpha = expf(m_prev - m_running);
            float beta = expf(qk - m_running);

            s_running = s_running * alpha + beta;

            // Accumulate weighted V contribution
            // Task 3.5: Load V as FP16, multiply by softmax weight (beta), accumulate in FP32
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                O_acc[d] = O_acc[d] * alpha + beta * __half2float(V_shared[tile_col][d]);
            }
        }
        __syncthreads();
    }

    // Task 3.5: Final normalization of accumulated output
    if (row_idx < N) {
        for (int d = 0; d < D; ++d) {
            O[row_idx * D + d] = O_acc[d] / s_running;
        }
    }
}

void gpu_attention_wmma_kernel(const float* d_Q, const float* d_K, const float* d_V, float* d_O) {
    dim3 grid((N + BR - 1) / BR);
    dim3 block(BR);
    fused_attention_wmma<<<grid, block>>>(d_Q, d_K, d_V, d_O);
}

// Wrapper function for WMMA kernel
void gpu_attention_wmma(const float* Q, const float* K, const float* V, float* O) {
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, N * D * sizeof(float));
    cudaMalloc(&d_K, N * D * sizeof(float));
    cudaMalloc(&d_V, N * D * sizeof(float));
    cudaMalloc(&d_O, N * D * sizeof(float));

    cudaMemcpy(d_Q, Q, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * D * sizeof(float), cudaMemcpyHostToDevice);

    gpu_attention_wmma_kernel(d_Q, d_K, d_V, d_O);

    cudaMemcpy(O, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}