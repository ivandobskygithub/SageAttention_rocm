"""
CPU validation script for SageAttention ROCm (tests logic without GPU)
"""

import torch
import torch.nn.functional as F
import numpy as np

# Force CPU mode for testing
torch.cuda.is_available = lambda: False

# Import our implementation
from sageattention_rocm.triton.fallback_implementations import (
    quantize,
    dequantize,
    attn_qk_int8_per_block,
    attn_qk_int8_per_block_causal,
)


def test_quantization():
    """Test quantization and dequantization."""
    print("\nTesting Quantization")
    print("-" * 40)

    # Create test tensor
    x = torch.randn(4, 256, 128, dtype=torch.float32)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

    # Quantize
    x_int8, scale = quantize(x, block_size=128)
    print(f"Quantized shape: {x_int8.shape}")
    print(f"Scale shape: {scale.shape}")
    print(f"Quantized range: [{x_int8.min()}, {x_int8.max()}]")

    # Dequantize
    x_deq = dequantize(x_int8, scale, block_size=128)
    print(f"Dequantized shape: {x_deq.shape}")

    # Check error
    error = (x - x_deq).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    rel_error = (error / (x.abs() + 1e-6)).mean().item()

    print(f"Max error: {max_error:.6f}")
    print(f"Mean error: {mean_error:.6f}")
    print(f"Relative error: {rel_error:.6f}")

    # Pass if relative error is less than 5%
    passed = rel_error < 0.05
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_attention():
    """Test attention computation."""
    print("\nTesting Attention")
    print("-" * 40)

    batch_heads = 4
    seq_len = 64
    head_dim = 32

    # Create small test tensors
    q = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_heads, seq_len, head_dim, dtype=torch.float32)

    print(f"Q/K/V shape: {q.shape}")

    # Quantize Q and K
    q_int8, q_scale = quantize(q, block_size=32)
    k_int8, k_scale = quantize(k, block_size=32)

    sm_scale = head_dim ** -0.5

    # Test non-causal attention
    print("\nNon-causal attention:")
    try:
        output = attn_qk_int8_per_block.forward(
            q_int8, k_int8, v,
            q_scale, k_scale,
            sm_scale,
            use_fp32_accum=True
        )
        print(f"  Output shape: {output.shape}")

        # Compare with PyTorch
        baseline = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        diff = (output - baseline).abs()
        max_error = diff.max().item()
        mean_error = diff.mean().item()

        print(f"  Max error vs PyTorch: {max_error:.6f}")
        print(f"  Mean error vs PyTorch: {mean_error:.6f}")

        non_causal_passed = max_error < 0.1  # Higher tolerance for quantized
        print(f"  Result: {'PASS' if non_causal_passed else 'FAIL'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        non_causal_passed = False

    # Test causal attention
    print("\nCausal attention:")
    try:
        output = attn_qk_int8_per_block_causal.forward(
            q_int8, k_int8, v,
            q_scale, k_scale,
            sm_scale,
            use_fp32_accum=True
        )
        print(f"  Output shape: {output.shape}")

        # Compare with PyTorch
        baseline = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)
        diff = (output - baseline).abs()
        max_error = diff.max().item()
        mean_error = diff.mean().item()

        print(f"  Max error vs PyTorch: {max_error:.6f}")
        print(f"  Mean error vs PyTorch: {mean_error:.6f}")

        causal_passed = max_error < 0.1
        print(f"  Result: {'PASS' if causal_passed else 'FAIL'}")

    except Exception as e:
        print(f"  ERROR: {e}")
        causal_passed = False

    return non_causal_passed and causal_passed


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("\nTesting Numerical Stability")
    print("-" * 40)

    tests_passed = True

    # Test 1: Very small values
    print("Test 1: Small values")
    x = torch.randn(2, 32, 16) * 1e-4
    q_int8, q_scale = quantize(x, block_size=16)
    x_deq = dequantize(q_int8, q_scale, block_size=16)
    error = (x - x_deq).abs().mean().item()
    print(f"  Mean error: {error:.8f}")
    tests_passed &= not np.isnan(error)

    # Test 2: Large values
    print("Test 2: Large values")
    x = torch.randn(2, 32, 16) * 100
    q_int8, q_scale = quantize(x, block_size=16)
    x_deq = dequantize(q_int8, q_scale, block_size=16)
    error = (x - x_deq).abs().mean().item()
    print(f"  Mean error: {error:.4f}")
    tests_passed &= not np.isnan(error)

    # Test 3: Mixed scales
    print("Test 3: Mixed scales")
    x = torch.randn(2, 32, 16)
    x[0] *= 100  # First batch has large values
    x[1] *= 0.001  # Second batch has small values
    q_int8, q_scale = quantize(x, block_size=16)
    x_deq = dequantize(q_int8, q_scale, block_size=16)
    error = (x - x_deq).abs().mean().item()
    print(f"  Mean error: {error:.6f}")
    tests_passed &= not np.isnan(error)

    print(f"\nResult: {'PASS' if tests_passed else 'FAIL'}")
    return tests_passed


def main():
    """Run CPU validation tests."""
    print("=" * 60)
    print("SageAttention ROCm CPU Validation")
    print("=" * 60)
    print("\nThis tests the logic without requiring a GPU.")
    print("Using PyTorch fallback implementations.")

    all_passed = True

    # Test components
    all_passed &= test_quantization()
    all_passed &= test_attention()
    all_passed &= test_numerical_stability()

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    if all_passed:
        print("[SUCCESS] All CPU tests PASSED")
        print("\nThe implementation logic is correct.")
        print("For GPU acceleration, install Triton and run on a CUDA/ROCm device.")
    else:
        print("[ERROR] Some tests FAILED")
        print("\nPlease check the implementation.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())