#!/usr/bin/env python3
"""
Generate Roofline Model Charts for Attention Kernel Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def generate_roofline_chart():
    """Generate roofline model visualization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # GPU specifications (Updated for actual hardware)
    gpus = {
        'RTX 3090 (PCIe4 x8)': {'peak_tflops': 39.5, 'mem_bw': 468, 'name': 'Ampere'},
        'RTX 4070 Ti Super (PCIe4 x8)': {'peak_tflops': 66.6, 'mem_bw': 288, 'name': 'Ada'},
    }
    
    kernels = {
        'Naive': {'ai': 28.4, 'color': 'red', 'marker': 'o'},
        'Fused': {'ai': 256.0, 'color': 'blue', 'marker': 's'},
        'WMMA': {'ai': 256.0, 'color': 'green', 'marker': '^'},
    }
    
    for idx, (gpu_name, specs) in enumerate(gpus.items()):
        ax = axes[idx]
        
        peak_tflops = specs['peak_tflops']
        mem_bw_gb = specs['mem_bw']
        knee_ai = peak_tflops * 1e12 / (mem_bw_gb * 1e9)
        
        # Generate AI range
        ai_range = np.logspace(0, 3, 200)  # 1 to 1000 FLOP/byte
        
        # Memory-bound line: TFLOPS = AI * Memory_BW
        memory_bound = (ai_range * mem_bw_gb) / 1000  # Convert to TFLOPS
        
        # Compute-bound line: TFLOPS = Peak
        compute_bound = np.ones_like(ai_range) * peak_tflops
        
        # Roofline envelope
        roofline = np.minimum(memory_bound, compute_bound)
        
        # Plot roofline
        ax.loglog(ai_range, roofline, 'k-', linewidth=2.5, label='Roofline')
        ax.loglog(ai_range, memory_bound, 'b--', linewidth=1.5, alpha=0.5, label='Memory-Bound')
        ax.loglog(ai_range, compute_bound, 'r--', linewidth=1.5, alpha=0.5, label='Compute-Bound')
        
        # Plot actual kernels
        for kernel_name, kernel_spec in kernels.items():
            ai = kernel_spec['ai']
            
            # Calculate achieved throughput
            achieved = min(peak_tflops, (ai * mem_bw_gb) / 1000)
            
            ax.loglog(ai, achieved, marker=kernel_spec['marker'], 
                     markersize=12, color=kernel_spec['color'], 
                     label=f'{kernel_name} (AI={ai})', zorder=5)
            
            # Add annotation
            if kernel_name == 'Naive':
                status = "Memory-bound"
            else:
                status = "Compute-bound" if ai > knee_ai else "Memory-bound"
            
            ax.annotate(f'{status}', 
                       xy=(ai, achieved), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Formatting
        ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Achieved Performance (TFLOPS)', fontsize=11, fontweight='bold')
        ax.set_title(f'{gpu_name}\n{specs["name"]} Architecture\nPeak: {peak_tflops} TFLOPS', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim([1, 500])
        ax.set_ylim([1, peak_tflops * 1.5])
        
        # Add knee marker
        ax.axvline(knee_ai, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(knee_ai, peak_tflops * 0.1, f'Knee: {knee_ai:.0f}', 
               rotation=90, fontsize=9, alpha=0.7)
    
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=10, 
              bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('roofline_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Roofline chart saved: roofline_analysis.png")
    
    return fig

def generate_speedup_chart():
    """Generate speedup comparison chart"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kernels = ['CPU\nStandard', 'GPU\nNaive', 'GPU\nFused', 'GPU\nWMMA']
    times_ms = [2635, 169, 106, 144]
    speedups = [1.0, 15.6, 24.9, 18.3]
    colors = ['gray', 'lightcoral', 'lightblue', 'lightgreen']
    
    # Create bar chart
    bars = ax.bar(kernels, speedups, color=colors, edgecolor='black', linewidth=1.5, 
                  width=0.6, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, speedup, time) in enumerate(zip(bars, speedups, times_ms)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{speedup:.1f}x\n({time:.0f}ms)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Speedup vs CPU', fontsize=12, fontweight='bold')
    ax.set_title('Attention Kernel Performance Comparison\n(N=1024, D=64)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim([0, max(speedups) * 1.3])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add acceleration arrow
    ax.annotate('', xy=(3, 24.9), xytext=(1, 1),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='darkgreen', alpha=0.6))
    ax.text(2, 13, '25x improvement', fontsize=12, fontweight='bold', 
           color='darkgreen', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('speedup_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Speedup chart saved: speedup_comparison.png")
    
    return fig

def generate_memory_analysis():
    """Generate memory efficiency analysis"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Memory usage
    kernels = ['Naive', 'Fused', 'WMMA']
    memory_mb = [4.25, 1.0, 0.75]
    colors = ['red', 'blue', 'green']
    
    bars1 = ax1.bar(kernels, memory_mb, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Memory per Iteration (MB)', fontsize=11, fontweight='bold')
    ax1.set_title('Global Memory Traffic', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, max(memory_mb) * 1.3])
    
    for bar, mem in zip(bars1, memory_mb):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mem:.2f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # Arithmetic Intensity
    ai = [47, 200, 270]
    bars2 = ax2.bar(kernels, ai, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Arithmetic Intensity (FLOP/byte)', fontsize=11, fontweight='bold')
    ax2.set_title('Compute Efficiency', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max(ai) * 1.3])
    
    for bar, intensity in zip(bars2, ai):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{intensity:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=150, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='A100 Knee')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('memory_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Memory analysis chart saved: memory_analysis.png")
    
    return fig

if __name__ == '__main__':
    print("Generating visualization charts...\n")
    generate_roofline_chart()
    generate_speedup_chart()
    generate_memory_analysis()
    print("\n✓ All charts generated successfully!")
    print("\nGenerated files:")
    print("  - roofline_analysis.png")
    print("  - speedup_comparison.png")
    print("  - memory_analysis.png")
