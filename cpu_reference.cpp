#include "attention.h"
#include <algorithm>
#include <float.h>

// Task 1.1: Standard CPU Reference [cite: 16]
void cpu_attention_standard(const float* Q, const float* K, const float* V, float* O) {
    std::vector<float> S(N * N);
    
    // 1. Compute Q*K^T and Scale
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k) {
                sum += Q[i * D + k] * K[j * D + k];
            }
            S[i * N + j] = sum * SCALE;
        }
    }

    // 2. Standard Softmax + Multiply by V
    for (int i = 0; i < N; ++i) {
        float row_max = -FLT_MAX;
        for (int j = 0; j < N; ++j) row_max = std::max(row_max, S[i * N + j]);
        
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            S[i * N + j] = std::exp(S[i * N + j] - row_max);
            row_sum += S[i * N + j];
        }

        for (int j = 0; j < N; ++j) {
            float res = 0.0f;
            for (int k = 0; k < N; ++k) {
                res += (S[i * N + k] / row_sum) * V[k * D + j];
            }
            O[i * D + j] = res;
        }
    }
}

// Task 1.2: Online Softmax Algorithm [cite: 17]
void cpu_attention_online(const float* Q, const float* K, const float* V, float* O) {
    for (int i = 0; i < N; ++i) {
        std::vector<float> row_out(D, 0.0f);
        float m = -FLT_MAX; // running max
        float s = 0.0f;     // running sum
        
        for (int j = 0; j < N; ++j) {
            float qk = 0.0f;
            for (int k = 0; k < D; ++k) qk += Q[i * D + k] * K[j * D + k];
            qk *= SCALE;

            float m_prev = m;
            m = std::max(m_prev, qk);
            
            float exp_val = std::exp(qk - m);
            float scale_prev = std::exp(m_prev - m);
            s = s * scale_prev + exp_val;

            for (int k = 0; k < D; ++k) {
                row_out[k] = row_out[k] * scale_prev + exp_val * V[j * D + k];
            }
        }
        for (int k = 0; k < D; ++k) O[i * D + k] = row_out[k] / s;
    }
}