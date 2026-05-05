#include "attention.h"
#include <iostream>
#include <iomanip>
#include <cmath>

// Task 1.4/2.5 Verification Protocol [cite: 20, 27]
bool verify_results(const char* label, const float* ref, const float* test, int size, float tolerance) {
    float max_err = 0.0f;
    for (int i = 0; i < size; ++i) {
        float err = std::abs(ref[i] - test[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << std::setw(20) << std::left << label 
              << " | Max Error: " << std::scientific << std::setprecision(5) << max_err;
    
    if (max_err < tolerance) {
        std::cout << " | [PASS]" << std::endl;
        return true;
    } else {
        std::cout << " | [FAIL]" << std::endl;
        return false;
    }
}
