"""
Test script for SageAttention Triton kernels on ROCm 7
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.dirname(__file__))

# Force ROCm backend
os.environ['TRITON_HIP_USE_ROCM'] = '1'

# Import our ROCm implementation
from sageattention_rocm import sageattn, sageattn_varlen, get_rocm_device_info


def print_device_info():
    """Print ROCm device information."""
    print("=" * 80)
    print("ROCm Device Information")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("No CUDA/ROCm devices available!")
        return False

    device_info = get_rocm_device_info()
    if device_info:
        print(f"Device: {device_info['name']}")
        print(f"Architecture: {device_info.get('arch', 'unknown')}")
        print(f"Memory: {device_info['total_memory'] / 1e9:.1f} GB")
        print(f"Multiprocessors: {device_info['multiprocessor_count']}")
        print(f"Has FP8: {device_info.get('has_fp8', False)}")
        print(f"Has MFMA: {device_info.get('has_mfma', False)}")
    else:
        print("Failed to get device information")
        return False

    # Additional PyTorch info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")

    # Check Triton
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton not installed!")
        return False

    print("=" * 80)
    return True


def test_basic_attention(batch_size=2, num_heads=8, seq_len=512, head_dim=64):
    """Test basic attention operation."""
    print(f"\nTesting basic attention: B={batch_size}, H={num_heads}, L={seq_len}, D={head_dim}")

    # Create random tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')

    # Test NHD layout
    try:
        output = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
        print(f"✓ NHD layout: output shape {output.shape}")
    except Exception as e:
        print(f"✗ NHD layout failed: {e}")
        return False

    # Test HND layout
    q_hnd = q.permute(1, 0, 2, 3).contiguous().view(batch_size * num_heads, seq_len, head_dim)
    k_hnd = k.permute(1, 0, 2, 3).contiguous().view(batch_size * num_heads, seq_len, head_dim)
    v_hnd = v.permute(1, 0, 2, 3).contiguous().view(batch_size * num_heads, seq_len, head_dim)

    try:
        output_hnd = sageattn(q_hnd, k_hnd, v_hnd, tensor_layout="HND", is_causal=False)
        print(f"✓ HND layout: output shape {output_hnd.shape}")
    except Exception as e:
        print(f"✗ HND layout failed: {e}")
        return False

    return True


def test_causal_attention(batch_size=2, num_heads=8, seq_len=512, head_dim=64):
    """Test causal attention."""
    print(f"\nTesting causal attention: B={batch_size}, H={num_heads}, L={seq_len}, D={head_dim}")

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')

    try:
        output = sageattn(q, k, v, tensor_layout="NHD", is_causal=True)
        print(f"✓ Causal attention: output shape {output.shape}")

        # Verify causality (upper triangle should be affected by mask)
        # This is a basic check - in production you'd compare against reference
        if output.isnan().any():
            print("✗ Output contains NaN values!")
            return False

    except Exception as e:
        print(f"✗ Causal attention failed: {e}")
        return False

    return True


def test_different_dimensions():
    """Test various head dimensions."""
    print("\nTesting different head dimensions...")

    test_configs = [
        (1, 4, 256, 64),   # Small config
        (2, 8, 512, 128),  # Medium config
        (1, 16, 1024, 256), # Large config
    ]

    for batch, heads, seq, dim in test_configs:
        print(f"\nConfig: B={batch}, H={heads}, L={seq}, D={dim}")

        q = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device='cuda')
        k = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device='cuda')
        v = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device='cuda')

        try:
            output = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
            print(f"  ✓ Success: output shape {output.shape}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False

    return True


def test_accumulation_dtypes():
    """Test different accumulation data types."""
    print("\nTesting accumulation dtypes...")

    batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')

    # Test FP32 accumulation
    try:
        output_fp32 = sageattn(q, k, v, pv_accum_dtype="fp32")
        print(f"✓ FP32 accumulation: output shape {output_fp32.shape}")
    except Exception as e:
        print(f"✗ FP32 accumulation failed: {e}")
        return False

    # Test FP16 accumulation
    try:
        output_fp16 = sageattn(q, k, v, pv_accum_dtype="fp16")
        print(f"✓ FP16 accumulation: output shape {output_fp16.shape}")
    except Exception as e:
        print(f"✗ FP16 accumulation failed: {e}")
        return False

    # Compare outputs (should be similar but not identical)
    diff = (output_fp32 - output_fp16).abs().mean().item()
    print(f"  Mean absolute difference: {diff:.6f}")

    return True


def test_key_smoothing():
    """Test key smoothing functionality."""
    print("\nTesting key smoothing...")

    batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')

    try:
        # With smoothing (default)
        output_smooth = sageattn(q, k, v, smooth_k=True)
        print(f"✓ With key smoothing: output shape {output_smooth.shape}")

        # Without smoothing
        output_no_smooth = sageattn(q, k, v, smooth_k=False)
        print(f"✓ Without key smoothing: output shape {output_no_smooth.shape}")

        # Compare outputs
        diff = (output_smooth - output_no_smooth).abs().mean().item()
        print(f"  Mean absolute difference: {diff:.6f}")

    except Exception as e:
        print(f"✗ Key smoothing test failed: {e}")
        return False

    return True


def benchmark_performance():
    """Benchmark attention performance."""
    print("\nBenchmarking performance...")

    configs = [
        (1, 32, 512, 128, "Small"),
        (4, 32, 1024, 128, "Medium"),
        (8, 32, 2048, 128, "Large"),
    ]

    for batch, heads, seq, dim, name in configs:
        q = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device='cuda')
        k = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device='cuda')
        v = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device='cuda')

        # Warmup
        for _ in range(3):
            _ = sageattn(q, k, v, tensor_layout="NHD")

        torch.cuda.synchronize()

        # Benchmark
        num_iters = 10
        start = time.time()
        for _ in range(num_iters):
            _ = sageattn(q, k, v, tensor_layout="NHD")
        torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_time = elapsed / num_iters * 1000  # Convert to ms
        tflops = 4 * batch * heads * seq * seq * dim / (avg_time / 1000) / 1e12

        print(f"  {name:10s}: {avg_time:7.2f} ms, {tflops:6.2f} TFLOPS")


def test_varlen_attention():
    """Test variable-length attention."""
    print("\nTesting variable-length attention...")

    # Create variable length sequences
    batch_size = 4
    num_heads = 8
    head_dim = 64
    seq_lens = [128, 256, 192, 320]  # Different lengths per batch

    total_q = sum(seq_lens)
    total_k = total_q

    # Cumulative sequence lengths
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).cpu().numpy()),
                             dtype=torch.int32, device='cuda')

    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(total_k, num_heads, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(total_k, num_heads, head_dim, dtype=torch.float16, device='cuda')

    try:
        output = sageattn_varlen(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max(seq_lens), max(seq_lens)
        )
        print(f"✓ Variable-length attention: output shape {output.shape}")
    except Exception as e:
        print(f"✗ Variable-length attention failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("SageAttention ROCm Triton Test Suite")
    print("=" * 80)

    # Check device
    if not print_device_info():
        print("Device check failed. Exiting.")
        return 1

    # Run tests
    tests = [
        ("Basic Attention", test_basic_attention),
        ("Causal Attention", test_causal_attention),
        ("Different Dimensions", test_different_dimensions),
        ("Accumulation Dtypes", test_accumulation_dtypes),
        ("Key Smoothing", test_key_smoothing),
        ("Variable-Length", test_varlen_attention),
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 80)
    print("Running Tests")
    print("=" * 80)

    for name, test_func in tests:
        try:
            if test_func():
                print(f"✓ {name} passed")
                passed += 1
            else:
                print(f"✗ {name} failed")
                failed += 1
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            failed += 1

    # Run benchmarks if all tests pass
    if failed == 0:
        print("\n" + "=" * 80)
        print("Performance Benchmarks")
        print("=" * 80)
        benchmark_performance()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())