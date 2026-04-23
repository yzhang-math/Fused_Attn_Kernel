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


// Tile dimensions from Task 2.1
#define BR 32
#define BC 64

__global__ void fused_attention_v1(const float* Q, const float* K, const float* V, float* O) {
    // Task 2.4: Shared Memory with padding to eliminate bank conflicts [cite: 26]
    __shared__ float Q_shared[BR][D + 4]; 
    __shared__ float K_shared[BC][D + 4];
    __shared__ float V_shared[BC][D + 4];

    // Identify which row of the global Q matrix this thread "owns"
    int row_idx = blockIdx.x * BR + threadIdx.x;
    int tid = threadIdx.x; // Thread index within the block (0 to 127)

    // Task 1.2/3.4: Initialize Online Softmax variables in FP32 registers [cite: 17, 31]
    float m_running = -1e10f; 
    float s_running = 0.0f;
    float O_acc[D]; // Local register accumulator for the output row [cite: 32]
    
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        O_acc[d] = 0.0f;
    }

    // Task 2.3: Collaboratively load the Q tile into Shared Memory [cite: 25]
    // Each thread loads elements for its own row (since BR=128 and blockDim=128)
    for (int d = 0; d < D; ++d) {
        Q_shared[tid][d] = Q[row_idx * D + d];
    }
    __syncthreads();

    // Loop over the sequence length in blocks of BC (64)
    for (int j = 0; j < N; j += BC) {
        
        // Task 2.3: Collaboratively load K and V tiles into Shared Memory [cite: 25]
        // Since blockDim=128 and BC=64, we use the first 64 threads
        if (tid < BC) {
            for (int d = 0; d < D; ++d) {
                K_shared[tid][d] = K[(j + tid) * D + d];
                V_shared[tid][d] = V[(j + tid) * D + d];
            }
        }
        __syncthreads();

        // Compute QK^T scores for this tile
        for (int tile_col = 0; tile_col < BC; ++tile_col) {
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
    for (int d = 0; d < D; ++d) {
        O[row_idx * D + d] = O_acc[d] / s_running;
    }
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

    // Grid: N/BR blocks (1024/128 = 8 blocks), each with 128 threads
    dim3 grid(N / BR);
    dim3 block(BR);

    fused_attention_v1<<<grid, block>>>(d_Q, d_K, d_V, d_O);

    cudaMemcpy(O, d_O, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}