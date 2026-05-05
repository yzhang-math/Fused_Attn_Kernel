#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:15:00
#SBATCH -J Attention_Bench
#SBATCH -o attention_bench-%j.out
#SBATCH -e attention_bench-%j.err 
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

# Load CUDA module
module load nvidia/cuda/13.0.0

# Compile the project in release mode for benchmarking
make clean
make MODE=release

echo "Starting Attention Kernel Benchmarks..."
echo "Sequence Length (N): 1024, Head Dim (D): 64"
echo "-------------------------------------------"

# Run the standard benchmark and verification
./attention_proj

echo "-------------------------------------------"
echo "Running Kernel-Only Timing (excluding H2D/D2H)..."

# Recompile for kernel-only timing
make clean
make MODE=release MAIN_SRC=kernel_timing_test.cpp
./attention_proj

echo "Job Complete."