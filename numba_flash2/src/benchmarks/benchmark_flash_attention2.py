"""
Benchmarking Suite for FlashAttention 2 Implementations

This module provides comprehensive benchmarking tools for comparing different
implementations of the FlashAttention 2 algorithm:
1. Numba CUDA implementation
2. PyTorch's native scaled_dot_product_attention
3. Standard attention implementation
4. CUDA C++ implementation

The benchmarks measure:
- Execution time
- Memory usage
- Numerical accuracy
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
import numba.cuda as ncuda
import math
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

from ..implementations.flash_attention2_numba import flash_attention2_launcher as flash_attention_numba

def generate_test_data(batch_size: int, num_heads: int, seq_len: int, head_dim: int, 
                      device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random test data for attention computation.
    
    Args:
        batch_size: Number of sequences in the batch
        num_heads: Number of attention heads
        seq_len: Length of input sequences
        head_dim: Dimension of each attention head
        device: Device to place tensors on ('cuda' or 'cpu')
        
    Returns:
        tuple: (Q, K, V) tensors for attention computation
    """
    shape = (batch_size, num_heads, seq_len, head_dim)
    
    # Generate random data in float16 for better compatibility
    Q = (torch.randn(*shape, device=device) / np.sqrt(head_dim)).to(torch.float16)
    K = (torch.randn(*shape, device=device) / np.sqrt(head_dim)).to(torch.float16)
    V = torch.randn(*shape, device=device).to(torch.float16)
    
    return Q, K, V

def benchmark_numba(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                   num_runs: int = 5) -> Dict[str, Any]:
    """
    Benchmark the Numba CUDA implementation with detailed profiling.
    """
    # Record initial memory state
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # Warmup run with profiling
    start_setup = time.perf_counter()
    output, _ = flash_attention_numba(Q, K, V)
    torch.cuda.synchronize()
    setup_time = time.perf_counter() - start_setup
    
    # Detailed timing runs
    compute_times = []
    data_transfer_times = []
    memory_peaks = []
    
    torch.cuda.synchronize()
    for _ in range(num_runs):
        # Measure data transfer time
        start_transfer = time.perf_counter()
        torch.cuda.synchronize()
        transfer_time = time.perf_counter() - start_transfer
        
        # Measure compute time
        start_compute = time.perf_counter()
        output, _ = flash_attention_numba(Q, K, V)
        torch.cuda.synchronize()
        compute_time = time.perf_counter() - start_compute
        
        compute_times.append(compute_time)
        data_transfer_times.append(transfer_time)
        memory_peaks.append(torch.cuda.max_memory_allocated() - initial_memory)
    
    return {
        'output': output,
        'mean_time': np.mean(compute_times),
        'std_time': np.std(compute_times),
        'setup_time': setup_time,
        'mean_transfer_time': np.mean(data_transfer_times),
        'mean_compute_time': np.mean(compute_times),
        'memory_allocated': torch.cuda.memory_allocated() - initial_memory,
        'memory_peak': np.mean(memory_peaks),
        'memory_reserved': torch.cuda.memory_reserved()
    }

def benchmark_native_pytorch(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           num_runs: int = 5) -> Dict[str, Any]:
    """
    Benchmark PyTorch's native scaled_dot_product_attention.
    """
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # Warmup run
    output = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    
    # Timing runs
    compute_times = []
    memory_peaks = []
    
    for _ in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        compute_time = time.perf_counter() - start
        compute_times.append(compute_time)
        memory_peaks.append(torch.cuda.max_memory_allocated() - initial_memory)
    
    return {
        'output': output,
        'mean_time': np.mean(compute_times),
        'std_time': np.std(compute_times),
        'memory_allocated': torch.cuda.memory_allocated() - initial_memory,
        'memory_peak': np.mean(memory_peaks),
        'memory_reserved': torch.cuda.memory_reserved()
    }

def standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Standard scaled dot-product attention implementation.
    
    Args:
        Q, K, V: Input tensors [Bs, Nh, N, dim]
    
    Returns:
        torch.Tensor: Output tensor [Bs, Nh, N, dim]
    """
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output

def benchmark_standard_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                               num_runs: int = 5) -> Dict[str, Any]:
    """
    Benchmark standard attention implementation.
    
    Args:
        Q, K, V: Input tensors
        num_runs: Number of runs for timing
        
    Returns:
        dict: Benchmark results including mean time, std time, and memory usage
    """
    # Warmup run
    output = standard_attention(Q, K, V)
    
    # Timing runs
    times = []
    torch.cuda.synchronize()
    for _ in range(num_runs):
        start = time.perf_counter()
        output = standard_attention(Q, K, V)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        'output': output,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'memory_allocated': torch.cuda.memory_allocated(),
        'memory_reserved': torch.cuda.memory_reserved()
    }

def benchmark_cuda(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                  num_runs: int = 5) -> Dict[str, Any]:
    """
    Benchmark the CUDA C++ implementation.
    """
    import subprocess
    import json
    
    # Compile the CUDA benchmark if needed
    cuda_dir = "cuda_flash2/src"
    subprocess.run(["nvcc", "-o", f"{cuda_dir}/benchmark_flash_attention2",
                   f"{cuda_dir}/benchmark_flash_attention2.cu",
                   "-lcudart", "-lcurand"], check=True)
    
    # Get input dimensions
    Bs, Nh, N, dim = Q.shape
    
    # Run the benchmark
    try:
        result = subprocess.run([f"{cuda_dir}/benchmark_flash_attention2",
                               str(Bs), str(Nh), str(N), str(dim), str(num_runs)],
                               capture_output=True, text=True, check=True)
        
        # Parse the output
        try:
            benchmark_data = json.loads(result.stdout)
            return {
                'output': None,  # We don't compare outputs with CUDA implementation
                'mean_time': benchmark_data['mean_time_ms'] / 1000,  # Convert to seconds
                'std_time': benchmark_data['std_time_ms'] / 1000,
                'memory_allocated': benchmark_data['memory_allocated'],
                'memory_reserved': benchmark_data['memory_reserved'],
                'setup_time': 0.0  # CUDA setup time is included in the kernel time
            }
        except json.JSONDecodeError as e:
            print("Error parsing CUDA benchmark output:")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            print("JSON error:", str(e))
            raise
    except subprocess.CalledProcessError as e:
        print("Error running CUDA benchmark:")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        print("Return code:", e.returncode)
        return {
            'output': None,
            'mean_time': float('inf'),
            'std_time': 0,
            'memory_allocated': 0,
            'memory_reserved': 0,
            'setup_time': 0.0
        }

def plot_benchmark_results(all_results: Dict[str, Dict[str, Dict[str, Any]]]):
    """
    Create grouped bar plots for timing and memory usage.
    
    Args:
        all_results: Dictionary of benchmark results for each sequence length
    """
    # Extract configurations and implementations
    configs = list(all_results.keys())
    implementations = list(all_results[configs[0]].keys())
    
    # Prepare data for plotting
    timing_data = {impl: [] for impl in implementations}
    memory_data = {impl: [] for impl in implementations}
    timing_errors = {impl: [] for impl in implementations}
    
    for config in configs:
        for impl in implementations:
            timing_data[impl].append(all_results[config][impl]['mean_time'] * 1000)  # Convert to ms
            timing_errors[impl].append(all_results[config][impl]['std_time'] * 1000)
            # Use peak memory if available, otherwise use allocated memory
            memory_data[impl].append(all_results[config][impl].get('memory_peak', all_results[config][impl]['memory_allocated']) / (1024 * 1024))  # Convert to MB
    
    # Set up the plots
    num_configs = len(configs)
    num_impls = len(implementations)
    width = 0.8 / num_impls
    
    # Timing plot
    plt.figure(figsize=(15, 6))
    for i, impl in enumerate(implementations):
        x = np.arange(num_configs) + i * width - (num_impls-1) * width/2
        plt.bar(x, timing_data[impl], width, label=impl,
                yerr=timing_errors[impl], capsize=5)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Performance Comparison (Log Scale)')
    plt.yscale('log')
    plt.xticks(np.arange(num_configs), [f'sl{c[2:]}' for c in configs], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_timing.png')
    plt.close()
    
    # Memory plot
    plt.figure(figsize=(15, 6))
    for i, impl in enumerate(implementations):
        x = np.arange(num_configs) + i * width - (num_impls-1) * width/2
        plt.bar(x, memory_data[impl], width, label=impl)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage Comparison (Log Scale)')
    plt.yscale('log')
    plt.xticks(np.arange(num_configs), [f'sl{c[2:]}' for c in configs], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('benchmark_memory.png')
    plt.close()
    
    print("\nBenchmark plots saved as 'benchmark_timing.png' and 'benchmark_memory.png'")

def run_benchmarks(batch_size: int = 2, num_heads: int = 8, seq_len: int = 1024, 
                  head_dim: int = 64, num_runs: int = 5):
    """
    Run comprehensive benchmarks with varying sequence lengths.
    """
    print(f"Running benchmarks with parameters:")
    print(f"Batch size: {batch_size} (fixed)")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {head_dim}")
    print(f"Number of runs: {num_runs}")
    
    # Test different sequence lengths
    seq_lengths = [512, 1024, 2048, 4096]
    
    all_results = {}
    
    for sl in seq_lengths:
        print(f"\nTesting seq_len={sl}")
        
        # Generate test data
        Q, K, V = generate_test_data(batch_size, num_heads, sl, head_dim)
        
        # Run benchmarks
        results = {}
        
        print("Running numba flashattn2...")
        results['numba flashattn2'] = benchmark_numba(Q, K, V, num_runs)
        
        print("Running pytorch scaled_dot_product_attention...")
        results['pytorch scaled_dot_product_attention'] = benchmark_native_pytorch(Q, K, V, num_runs)
        
        print("Running pytorch standard attention...")
        results['pytorch standard attention'] = benchmark_standard_attention(Q, K, V, num_runs)
        
        print("Running CUDA implementation...")
        results['cuda flashattn2'] = benchmark_cuda(Q, K, V, num_runs)
        
        # Store results
        all_results[f'sl{sl}'] = results
        
        # Print detailed results
        print("\nDetailed Results:")
        for impl, res in results.items():
            print(f"\n{impl}:")
            print(f"Mean compute time: {res.get('mean_compute_time', res['mean_time'])*1000:.2f} ms")
            print(f"Setup time: {res.get('setup_time', 0)*1000:.2f} ms")
            if 'mean_transfer_time' in res:
                print(f"Data transfer time: {res['mean_transfer_time']*1000:.2f} ms")
            print(f"Peak memory: {res.get('memory_peak', res.get('memory_allocated', 0))/1024**2:.2f} MB")
            print(f"Memory allocated: {res['memory_allocated']/1024**2:.2f} MB")
        
        # Verify outputs match (excluding CUDA implementation)
        print("\nVerifying outputs...")
        reference_output = results['pytorch scaled_dot_product_attention']['output']
        for impl, res in results.items():
            if impl != 'pytorch scaled_dot_product_attention' and impl != 'cuda flashattn2' and res['output'] is not None:
                torch.testing.assert_close(
                    res['output'], reference_output,
                    rtol=1e-3, atol=1e-3,  # Relaxed tolerances for float16
                    msg=f"{impl} output differs from PyTorch's scaled_dot_product_attention"
                )
        print("All implementations produce matching outputs!")
    
    # Create summary plots
    plot_benchmark_results(all_results)
    return all_results

if __name__ == '__main__':
    run_benchmarks() 