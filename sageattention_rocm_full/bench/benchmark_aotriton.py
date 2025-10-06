"""
Benchmark SageAttention with AOTriton acceleration
"""

import os
import sys
import torch
import time
import numpy as np
from typing import Dict, List, Tuple

# Set environment variable for AOTriton
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_environment():
    """Check and report environment status."""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")

    # Check for AOTriton
    has_aotriton = hasattr(torch, '_triton_scaled_dot_attention')
    print(f"AOTriton Functions: {'AVAILABLE' if has_aotriton else 'NOT FOUND'}")

    print(f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: {os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'Not set')}")

    if not torch.cuda.is_available():
        print("\n[WARNING] No GPU detected. Benchmarks will run on CPU (slow).")
        print("For GPU benchmarks, ensure:")
        print("  1. ROCm drivers are installed")
        print("  2. PyTorch with ROCm support is installed")
        print("  3. AMD GPU is properly configured")
        return False

    return True


def benchmark_attention_kernel(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
    num_warmup: int = 3,
    num_iterations: int = 20
) -> Dict[str, float]:
    """Benchmark different attention implementations."""

    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

    results = {}

    # 1. Benchmark PyTorch native
    if device == 'cuda':
        torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    results['pytorch_native'] = (end - start) / num_iterations * 1000  # ms

    # 2. Benchmark SageAttention
    try:
        from sageattention_rocm import sageattn

        # Warmup
        for _ in range(num_warmup):
            _ = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)

        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        results['sageattention'] = (end - start) / num_iterations * 1000

    except Exception as e:
        print(f"  SageAttention error: {e}")
        results['sageattention'] = -1

    # 3. Try AOTriton directly if available
    if hasattr(torch, '_triton_scaled_dot_attention') and device == 'cuda':
        try:
            # Warmup
            for _ in range(num_warmup):
                _ = torch._triton_scaled_dot_attention(q, k, v)

            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = torch._triton_scaled_dot_attention(q, k, v)
            torch.cuda.synchronize()
            end = time.perf_counter()

            results['aotriton_direct'] = (end - start) / num_iterations * 1000

        except Exception as e:
            print(f"  AOTriton direct error: {e}")
            results['aotriton_direct'] = -1

    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks across different configurations."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    # Test configurations
    configs = [
        # (batch_size, num_heads, seq_len, head_dim)
        (1, 8, 256, 64),    # Small
        (2, 16, 512, 64),   # Medium
        (2, 16, 1024, 128), # Large
        (1, 32, 2048, 128), # Extra large
    ]

    print("\n" + "=" * 70)
    print("Running Comprehensive Benchmarks")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()

    # Table header
    print(f"{'Config':<30} {'PyTorch(ms)':<12} {'SageAttn(ms)':<12} {'AOTriton(ms)':<12} {'Speedup':<10}")
    print("-" * 76)

    all_results = []

    for batch, heads, seq_len, head_dim in configs:
        config_str = f"B={batch},H={heads},L={seq_len},D={head_dim}"

        try:
            results = benchmark_attention_kernel(
                batch, heads, seq_len, head_dim,
                dtype=dtype, device=device,
                num_warmup=3, num_iterations=10 if device == 'cpu' else 20
            )

            pytorch_time = results.get('pytorch_native', -1)
            sage_time = results.get('sageattention', -1)
            aotriton_time = results.get('aotriton_direct', -1)

            if pytorch_time > 0 and sage_time > 0:
                speedup = pytorch_time / sage_time
            else:
                speedup = 0

            # Format output
            pytorch_str = f"{pytorch_time:.2f}" if pytorch_time > 0 else "N/A"
            sage_str = f"{sage_time:.2f}" if sage_time > 0 else "N/A"
            aotriton_str = f"{aotriton_time:.2f}" if aotriton_time > 0 else "N/A"
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"

            print(f"{config_str:<30} {pytorch_str:<12} {sage_str:<12} {aotriton_str:<12} {speedup_str:<10}")

            all_results.append({
                'config': config_str,
                'pytorch': pytorch_time,
                'sageattention': sage_time,
                'aotriton': aotriton_time,
                'speedup': speedup
            })

        except Exception as e:
            print(f"{config_str:<30} Error: {e}")

    return all_results


def benchmark_memory_usage():
    """Benchmark memory usage of different implementations."""

    if not torch.cuda.is_available():
        print("\n[INFO] Memory benchmarks require GPU")
        return

    print("\n" + "=" * 70)
    print("Memory Usage Comparison")
    print("=" * 70)

    device = 'cuda'
    dtype = torch.float16

    configs = [
        (1, 8, 512, 64),
        (1, 16, 1024, 128),
        (2, 16, 2048, 128),
    ]

    print(f"{'Config':<30} {'PyTorch(MB)':<15} {'SageAttn(MB)':<15} {'Savings':<10}")
    print("-" * 70)

    for batch, heads, seq_len, head_dim in configs:
        config_str = f"B={batch},H={heads},L={seq_len},D={head_dim}"

        try:
            # Reset memory stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Measure PyTorch memory
            q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)

            torch.cuda.synchronize()
            start_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

            pytorch_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 - start_mem

            # Reset for SageAttention
            del q, k, v
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Measure SageAttention memory
            from sageattention_rocm import sageattn

            q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
            k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
            v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)

            torch.cuda.synchronize()
            start_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

            _ = sageattn(q, k, v, tensor_layout="NHD")
            torch.cuda.synchronize()

            sage_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 - start_mem

            # Calculate savings
            savings = (1 - sage_mem / pytorch_mem) * 100 if pytorch_mem > 0 else 0

            print(f"{config_str:<30} {pytorch_mem:<15.2f} {sage_mem:<15.2f} {savings:>8.1f}%")

        except Exception as e:
            print(f"{config_str:<30} Error: {e}")


def main():
    """Main benchmark function."""

    print("=" * 70)
    print("SageAttention ROCm Benchmark with AOTriton")
    print("=" * 70)

    # Check environment
    gpu_available = check_environment()

    # Import and check SageAttention
    print("\n" + "=" * 70)
    print("SageAttention Status")
    print("=" * 70)

    try:
        from sageattention_rocm import sageattn, get_rocm_device_info, AOTRITON_AVAILABLE

        print("[SUCCESS] SageAttention imported")

        device_info = get_rocm_device_info()
        if device_info:
            print(f"Device: {device_info.get('name', 'Unknown')}")
            print(f"AOTriton Available: {device_info.get('aotriton_available', False)}")
            print(f"Triton Available: {device_info.get('triton_available', False)}")

            if 'aotriton_status' in device_info:
                print("\nAOTriton Status:")
                for key, value in device_info['aotriton_status'].items():
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"[ERROR] Failed to import SageAttention: {e}")
        return 1

    # Run benchmarks
    results = run_comprehensive_benchmark()

    # Run memory benchmarks if GPU available
    if gpu_available:
        benchmark_memory_usage()

    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)

    if results:
        valid_speedups = [r['speedup'] for r in results if r['speedup'] > 0]
        if valid_speedups:
            avg_speedup = np.mean(valid_speedups)
            print(f"Average Speedup: {avg_speedup:.2f}x")

        if gpu_available:
            print("\n[SUCCESS] Benchmarks completed successfully!")
            if device_info.get('aotriton_available'):
                print("AOTriton acceleration is active!")
            else:
                print("Note: AOTriton not detected, using fallback implementations")
        else:
            print("\n[INFO] Benchmarks ran on CPU (limited performance)")
            print("Install PyTorch with ROCm for GPU acceleration")

    return 0


if __name__ == "__main__":
    exit(main())