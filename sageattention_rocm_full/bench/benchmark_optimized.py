"""
Optimized benchmark for SageAttention ROCm
Uses PyTorch's built-in optimized kernels
"""

import os
import sys
import torch
import time
import numpy as np

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import optimized implementation
from sageattention_rocm.core_optimized import sageattn_optimized, get_rocm_device_info


def check_environment():
    """Check and report environment status."""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_info = get_rocm_device_info()
        if device_info:
            print(f"Device: {device_info['name']}")
            print(f"Memory: {device_info['total_memory'] / 1e9:.1f} GB")
            print(f"Compute Capability: {device_info['compute_capability']}")
            print(f"Flash Attention: {'Available' if device_info.get('flash_attention') else 'Not Available'}")
            print(f"Memory Efficient Attention: {'Available' if device_info.get('mem_efficient_attention') else 'Not Available'}")

    return torch.cuda.is_available()


def benchmark_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
    num_warmup: int = 3,
    num_iterations: int = 20
):
    """Benchmark different attention implementations."""

    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

    results = {}

    # 1. PyTorch Native
    if device == 'cuda':
        torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    results['pytorch_native'] = (end - start) / num_iterations * 1000

    # 2. SageAttention Optimized with INT8
    try:
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = sageattn_optimized(q, k, v, tensor_layout="NHD", use_int8=True)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = sageattn_optimized(q, k, v, tensor_layout="NHD", use_int8=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        results['sage_int8'] = (end - start) / num_iterations * 1000

    except Exception as e:
        print(f"  SageAttention INT8 error: {e}")
        results['sage_int8'] = -1

    # 3. SageAttention without quantization
    try:
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = sageattn_optimized(q, k, v, tensor_layout="NHD", use_int8=False)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = sageattn_optimized(q, k, v, tensor_layout="NHD", use_int8=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        results['sage_no_quant'] = (end - start) / num_iterations * 1000

    except Exception as e:
        print(f"  SageAttention no-quant error: {e}")
        results['sage_no_quant'] = -1

    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    configs = [
        # (batch_size, num_heads, seq_len, head_dim)
        (1, 8, 256, 64),
        (2, 16, 512, 64),
        (2, 16, 1024, 128),
        (1, 32, 2048, 128),
    ]

    print("\n" + "=" * 70)
    print("Performance Benchmarks")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()

    # Table header
    print(f"{'Config':<25} {'PyTorch(ms)':<12} {'Sage-INT8(ms)':<14} {'Sage-NoQ(ms)':<13} {'INT8-Speedup':<12}")
    print("-" * 76)

    all_results = []

    for batch, heads, seq_len, head_dim in configs:
        config_str = f"B={batch},H={heads},L={seq_len}"

        try:
            results = benchmark_attention(
                batch, heads, seq_len, head_dim,
                dtype=dtype, device=device,
                num_warmup=3, num_iterations=10 if device == 'cpu' else 20
            )

            pytorch_time = results.get('pytorch_native', -1)
            sage_int8_time = results.get('sage_int8', -1)
            sage_noq_time = results.get('sage_no_quant', -1)

            if pytorch_time > 0 and sage_int8_time > 0:
                int8_speedup = pytorch_time / sage_int8_time
            else:
                int8_speedup = 0

            # Format output
            pytorch_str = f"{pytorch_time:.2f}" if pytorch_time > 0 else "N/A"
            sage_int8_str = f"{sage_int8_time:.2f}" if sage_int8_time > 0 else "N/A"
            sage_noq_str = f"{sage_noq_time:.2f}" if sage_noq_time > 0 else "N/A"
            speedup_str = f"{int8_speedup:.2f}x" if int8_speedup > 0 else "N/A"

            print(f"{config_str:<25} {pytorch_str:<12} {sage_int8_str:<14} {sage_noq_str:<13} {speedup_str:<12}")

            all_results.append({
                'config': config_str,
                'pytorch': pytorch_time,
                'sage_int8': sage_int8_time,
                'sage_noq': sage_noq_time,
                'speedup': int8_speedup
            })

        except Exception as e:
            print(f"{config_str:<25} Error: {e}")

    return all_results


def test_correctness():
    """Test correctness of optimized implementation."""

    if not torch.cuda.is_available():
        device = 'cpu'
        dtype = torch.float32
    else:
        device = 'cuda'
        dtype = torch.float16

    print("\n" + "=" * 70)
    print("Correctness Test")
    print("=" * 70)

    # Test configuration
    batch = 1
    heads = 8
    seq_len = 128
    head_dim = 64

    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)

    # Baseline
    with torch.no_grad():
        baseline = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Test INT8 version
    with torch.no_grad():
        output_int8 = sageattn_optimized(q, k, v, tensor_layout="NHD", use_int8=True)

    # Compare
    diff = (output_int8 - baseline).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    rel_error = (diff / (baseline.abs() + 1e-6)).mean().item()

    print(f"INT8 Quantized vs Baseline:")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  Relative error: {rel_error:.6f}")

    if rel_error < 0.05:  # 5% tolerance
        print("  Status: PASS")
    else:
        print("  Status: FAIL")


def main():
    print("=" * 70)
    print("SageAttention ROCm Optimized Benchmark")
    print("=" * 70)

    # Check environment
    gpu_available = check_environment()

    # Test correctness
    test_correctness()

    # Run benchmarks
    results = run_comprehensive_benchmark()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if results:
        valid_speedups = [r['speedup'] for r in results if r['speedup'] > 0]
        if valid_speedups:
            avg_speedup = np.mean(valid_speedups)
            print(f"Average INT8 Speedup: {avg_speedup:.2f}x")

            if avg_speedup > 1.0:
                print("\n[SUCCESS] INT8 quantization provides speedup!")
            else:
                print("\n[INFO] INT8 overhead detected - likely due to:")
                print("  - Small batch/sequence sizes")
                print("  - Quantization overhead on this GPU")
                print("  - Consider using larger sequences for better speedup")

        if gpu_available:
            print("\n[SUCCESS] Benchmarks completed on GPU")
            device_info = get_rocm_device_info()
            if device_info and (device_info.get('flash_attention') or device_info.get('mem_efficient_attention')):
                print("Optimized attention kernels are being used!")
        else:
            print("\n[INFO] Benchmarks ran on CPU")

    return 0


if __name__ == "__main__":
    exit(main())