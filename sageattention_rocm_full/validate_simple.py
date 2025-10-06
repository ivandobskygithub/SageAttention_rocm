"""
Simple validation script for SageAttention ROCm that works with or without Triton
"""

import torch
import torch.nn.functional as F
import time
import numpy as np

# Import our implementation
from sageattention_rocm import sageattn, get_rocm_device_info

def print_system_info():
    """Print system information."""
    print("=" * 60)
    print("SageAttention ROCm Validation (Simplified)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: True")

        device_info = get_rocm_device_info()
        if device_info:
            print(f"Architecture: {device_info.get('arch', 'unknown')}")
            print(f"Memory: {device_info['total_memory'] / 1e9:.1f} GB")
            print(f"Triton Available: {device_info.get('triton_available', False)}")
    else:
        print("No CUDA/ROCm device available!")
        return False

    print("=" * 60)
    return True


def validate_correctness():
    """Run basic correctness validation."""
    print("\nCorrectness Validation")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16

    test_configs = [
        (1, 8, 128, 64),   # Small
        (2, 16, 256, 128), # Medium
        (1, 8, 512, 64),   # Longer sequence
    ]

    all_passed = True

    for batch, heads, seq_len, head_dim in test_configs:
        print(f"\nTesting B={batch}, H={heads}, L={seq_len}, D={head_dim}")

        # Create test tensors
        q = torch.randn(batch, heads, seq_len, head_dim,
                      dtype=dtype, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim,
                      dtype=dtype, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim,
                      dtype=dtype, device=device)

        # Test non-causal
        try:
            # PyTorch baseline
            with torch.no_grad():
                baseline = F.scaled_dot_product_attention(q, k, v, is_causal=False)

            # SageAttention ROCm
            with torch.no_grad():
                output = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)

            # Compare
            diff = (output - baseline).abs()
            max_error = diff.max().item()
            mean_error = diff.mean().item()

            passed = max_error < 0.01  # 1% tolerance

            print(f"  Non-causal: {'PASS' if passed else 'FAIL'} "
                  f"(max_err={max_error:.6f}, mean_err={mean_error:.6f})")

            all_passed &= passed

        except Exception as e:
            print(f"  Non-causal: ERROR - {e}")
            all_passed = False

        # Test causal
        try:
            # PyTorch baseline
            with torch.no_grad():
                baseline = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            # SageAttention ROCm
            with torch.no_grad():
                output = sageattn(q, k, v, tensor_layout="NHD", is_causal=True)

            # Compare
            diff = (output - baseline).abs()
            max_error = diff.max().item()
            mean_error = diff.mean().item()

            passed = max_error < 0.01

            print(f"  Causal:     {'PASS' if passed else 'FAIL'} "
                  f"(max_err={max_error:.6f}, mean_err={mean_error:.6f})")

            all_passed &= passed

        except Exception as e:
            print(f"  Causal: ERROR - {e}")
            all_passed = False

    return all_passed


def benchmark_performance():
    """Run basic performance benchmarks."""
    print("\n" + "="*60)
    print("Performance Benchmarks")
    print("-" * 40)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return

    device = 'cuda'
    dtype = torch.float16
    batch = 2
    heads = 16
    head_dim = 128

    seq_lens = [256, 512, 1024, 2048]

    print(f"\nConfig: B={batch}, H={heads}, D={head_dim}")
    print(f"{'SeqLen':>8} {'PyTorch(ms)':>12} {'SageAttn(ms)':>13} {'Speedup':>10}")
    print("-" * 45)

    for seq_len in seq_lens:
        try:
            q = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=dtype, device=device)
            k = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=dtype, device=device)
            v = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=dtype, device=device)

            # Warmup
            for _ in range(3):
                _ = F.scaled_dot_product_attention(q, k, v)
                _ = sageattn(q, k, v, tensor_layout="NHD")
            torch.cuda.synchronize()

            # Benchmark PyTorch
            num_iters = 20
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_iters):
                    _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start) / num_iters * 1000

            # Benchmark SageAttention
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_iters):
                    _ = sageattn(q, k, v, tensor_layout="NHD")
            torch.cuda.synchronize()
            sage_time = (time.perf_counter() - start) / num_iters * 1000

            speedup = pytorch_time / sage_time

            print(f"{seq_len:8d} {pytorch_time:12.2f} {sage_time:13.2f} {speedup:9.2f}x")

        except Exception as e:
            print(f"{seq_len:8d} Error: {e}")


def main():
    """Run validation tests."""

    # Print system info
    if not print_system_info():
        return 1

    # Check if using fallback
    device_info = get_rocm_device_info()
    if device_info and not device_info.get('triton_available', False):
        print("\n⚠️  WARNING: Using PyTorch fallback implementation!")
        print("   This is MUCH slower than Triton kernels.")
        print("   For full performance, install Triton.")
        print("")

    # Run validation
    print("\nRunning validation tests...")
    correctness_passed = validate_correctness()

    # Run benchmarks
    benchmark_performance()

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    if correctness_passed:
        print("✅ Correctness tests PASSED")
    else:
        print("❌ Correctness tests FAILED")

    if device_info and not device_info.get('triton_available', False):
        print("\n📝 Note: Install Triton for GPU acceleration:")
        print("   - Linux: pip install triton")
        print("   - Windows: Use WSL2 or Docker with Linux")

    return 0 if correctness_passed else 1


if __name__ == "__main__":
    exit(main())