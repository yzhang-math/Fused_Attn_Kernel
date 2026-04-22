#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

// Project Constraints 
const int N = 1024; // Sequence length
const int D = 64;   // Head dimension
const float SCALE = 0.125f; // 1/sqrt(D)

// CPU Implementations
void cpu_attention_standard(const float* Q, const float* K, const float* V, float* O);
void cpu_attention_online(const float* Q, const float* K, const float* V, float* O);

// GPU Implementations
void gpu_attention_naive(const float* Q, const float* K, const float* V, float* O);

// Utility for accuracy check 
bool verify_results(const float* ref, const float* test, int size, float tolerance = 1e-4f);

#endif