"""
GPU Kernel Validation Tests for SageAttention3 ROCm

This test suite validates the correctness of the HIP kernels by comparing
against PyTorch reference implementations.

Tests include:
- INT4 quantization/dequantization accuracy
- Attention forward pass correctness
- Memory efficiency validation
- Numerical stability tests
- Edge case handling
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import pytest
from typing import Tuple

from torch_integration import SageAttentionROCm


class TestINT4Quantization:
    """Tests for INT4 quantization and dequantization."""

    @pytest.fixture(scope="class")
    def sage(self):
        """Initialize SageAttention instance once for all tests."""
        return SageAttentionROCm()

    @pytest.fixture
    def device(self):
        """Get CUDA device."""
        return torch.device("cuda:0")

    def test_quantize_dequantize_roundtrip(self, sage, device):
        """Test that quantize->dequantize recovers approximate input."""
        # Create test tensor
        batch, tokens, heads, dim = 2, 128, 8, 64
        x = torch.randn(batch, tokens, heads, dim, dtype=torch.float16, device=device)

        # Quantize
        x_quant, scales = sage.quantize_int4(x, block_size=16)

        # Validate quantized shape
        assert x_quant.shape == (batch, tokens, heads, dim // 2)
        assert x_quant.dtype == torch.uint8
        assert scales.shape == (batch, tokens, heads, dim // 16)
        assert scales.dtype == torch.float16

        # Dequantize
        x_recon = sage.dequantize_int4(x_quant, scales, head_dim=dim)

        # Validate reconstructed shape
        assert x_recon.shape == x.shape
        assert x_recon.dtype == torch.float16

        # Check reconstruction error
        # INT4 has limited precision, so we expect some error
        max_error = torch.max(torch.abs(x - x_recon)).item()
        mean_error = torch.mean(torch.abs(x - x_recon)).item()
        relative_error = mean_error / (torch.mean(torch.abs(x)).item() + 1e-6)

        print(f"\nQuantization error stats:")
        print(f"  Max error: {max_error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")
        print(f"  Relative error: {relative_error:.4%}")

        # INT4 should be reasonably accurate (within ~10% relative error)
        assert relative_error < 0.15, f"Relative error too high: {relative_error:.4%}"

    def test_quantization_symmetry(self, sage, device):
        """Test that quantization is symmetric around zero."""
        batch, tokens, heads, dim = 1, 64, 4, 64

        # Create symmetric input
        x = torch.randn(batch, tokens, heads, dim, dtype=torch.float16, device=device)

        # Quantize positive and negative
        x_pos_quant, scales_pos = sage.quantize_int4(x, block_size=16)
        x_neg_quant, scales_neg = sage.quantize_int4(-x, block_size=16)

        # Dequantize
        x_pos_recon = sage.dequantize_int4(x_pos_quant, scales_pos, head_dim=dim)
        x_neg_recon = sage.dequantize_int4(x_neg_quant, scales_neg, head_dim=dim)

        # Check symmetry: -Q(x) should be approximately Q(-x)
        symmetry_error = torch.mean(torch.abs(x_pos_recon + x_neg_recon)).item()
        print(f"\nSymmetry error: {symmetry_error:.6f}")

        # Should be very close to zero
        assert symmetry_error < 0.1

    def test_quantization_zero_input(self, sage, device):
        """Test quantization of zero tensor."""
        batch, tokens, heads, dim = 1, 32, 2, 64
        x = torch.zeros(batch, tokens, heads, dim, dtype=torch.float16, device=device)

        x_quant, scales = sage.quantize_int4(x, block_size=16)
        x_recon = sage.dequantize_int4(x_quant, scales, head_dim=dim)

        # Should recover exact zeros
        max_error = torch.max(torch.abs(x_recon)).item()
        assert max_error < 1e-4, f"Zero input should reconstruct to zero, got max={max_error}"

    def test_quantization_memory_savings(self, sage, device):
        """Test that INT4 reduces memory by ~4x."""
        batch, tokens, heads, dim = 4, 512, 16, 128
        x = torch.randn(batch, tokens, heads, dim, dtype=torch.float16, device=device)

        # Original memory (FP16 = 2 bytes per element)
        original_bytes = x.numel() * 2

        # Quantize
        x_quant, scales = sage.quantize_int4(x, block_size=16)

        # Quantized memory (0.5 bytes per element + scales)
        quantized_bytes = x_quant.numel() * 1  # UINT8
        scale_bytes = scales.numel() * 2  # FP16
        total_quantized_bytes = quantized_bytes + scale_bytes

        compression_ratio = original_bytes / total_quantized_bytes

        print(f"\nMemory usage:")
        print(f"  Original (FP16): {original_bytes / 1e6:.2f} MB")
        print(f"  Quantized (INT4): {quantized_bytes / 1e6:.2f} MB")
        print(f"  Scales (FP16): {scale_bytes / 1e6:.2f} MB")
        print(f"  Total compressed: {total_quantized_bytes / 1e6:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")

        # Should achieve close to 4x compression
        assert compression_ratio > 3.0, f"Insufficient compression: {compression_ratio:.2f}x"

    def test_quantization_range_values(self, sage, device):
        """Test quantization with extreme values."""
        batch, tokens, heads, dim = 1, 64, 4, 64

        # Test with various ranges
        for scale_factor in [0.1, 1.0, 10.0, 100.0]:
            x = torch.randn(batch, tokens, heads, dim, dtype=torch.float16, device=device) * scale_factor

            x_quant, scales = sage.quantize_int4(x, block_size=16)
            x_recon = sage.dequantize_int4(x_quant, scales, head_dim=dim)

            # Check relative error is reasonable
            relative_error = torch.mean(torch.abs(x - x_recon)) / (torch.mean(torch.abs(x)) + 1e-6)
            print(f"Scale {scale_factor:6.1f}: relative error = {relative_error:.4%}")

            assert relative_error < 0.2


class TestAttentionForward:
    """Tests for attention forward pass."""

    @pytest.fixture(scope="class")
    def sage(self):
        """Initialize SageAttention instance."""
        return SageAttentionROCm()

    @pytest.fixture
    def device(self):
        """Get CUDA device."""
        return torch.device("cuda:0")

    def pytorch_attention_reference(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Reference implementation using PyTorch.

        Args:
            Q: [batch, num_heads, seq_len_q, head_dim]
            K: [batch, num_heads, seq_len_k, head_dim]
            V: [batch, num_heads, seq_len_k, head_dim]
            scale: Scaling factor
            is_causal: Apply causal mask

        Returns:
            Output: [batch, num_heads, seq_len_q, head_dim]
        """
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply causal mask if needed
        if is_causal:
            seq_len_q = Q.size(2)
            seq_len_k = K.size(2)
            mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=Q.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply to values
        output = torch.matmul(attn_weights, V)

        return output

    def test_attention_basic(self, sage, device):
        """Test basic attention computation."""
        batch, num_heads, seq_len, head_dim = 2, 8, 128, 64

        # Create inputs
        Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        K = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

        scale = 1.0 / (head_dim ** 0.5)

        # Compute with our kernel
        O_rocm = sage.attention_forward(Q, K, V, scale=scale, is_causal=False)

        # Compute with PyTorch
        O_pytorch = self.pytorch_attention_reference(Q, K, V, scale, is_causal=False)

        # Compare results
        max_error = torch.max(torch.abs(O_rocm - O_pytorch)).item()
        mean_error = torch.mean(torch.abs(O_rocm - O_pytorch)).item()
        relative_error = mean_error / (torch.mean(torch.abs(O_pytorch)).item() + 1e-6)

        print(f"\nAttention error vs PyTorch:")
        print(f"  Max error: {max_error:.6f}")
        print(f"  Mean error: {mean_error:.6f}")
        print(f"  Relative error: {relative_error:.4%}")

        # FP16 operations have some numerical error, allow reasonable tolerance
        assert relative_error < 0.05, f"Relative error too high: {relative_error:.4%}"

    def test_attention_causal(self, sage, device):
        """Test attention with causal masking."""
        batch, num_heads, seq_len, head_dim = 2, 4, 64, 64

        Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        K = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

        scale = 1.0 / (head_dim ** 0.5)

        # Compute with causal mask
        O_rocm = sage.attention_forward(Q, K, V, scale=scale, is_causal=True)
        O_pytorch = self.pytorch_attention_reference(Q, K, V, scale, is_causal=True)

        # Compare
        relative_error = torch.mean(torch.abs(O_rocm - O_pytorch)) / (torch.mean(torch.abs(O_pytorch)) + 1e-6)
        print(f"\nCausal attention relative error: {relative_error:.4%}")

        assert relative_error < 0.05

    def test_attention_different_seq_lengths(self, sage, device):
        """Test attention with different Q and K/V sequence lengths."""
        batch, num_heads, seq_len_q, seq_len_k, head_dim = 2, 4, 64, 128, 64

        Q = torch.randn(batch, num_heads, seq_len_q, head_dim, dtype=torch.float16, device=device)
        K = torch.randn(batch, num_heads, seq_len_k, head_dim, dtype=torch.float16, device=device)
        V = torch.randn(batch, num_heads, seq_len_k, head_dim, dtype=torch.float16, device=device)

        scale = 1.0 / (head_dim ** 0.5)

        O_rocm = sage.attention_forward(Q, K, V, scale=scale, is_causal=False)
        O_pytorch = self.pytorch_attention_reference(Q, K, V, scale, is_causal=False)

        relative_error = torch.mean(torch.abs(O_rocm - O_pytorch)) / (torch.mean(torch.abs(O_pytorch)) + 1e-6)
        print(f"\nDifferent seq lengths relative error: {relative_error:.4%}")

        assert relative_error < 0.05

    def test_attention_single_token(self, sage, device):
        """Test attention with single token (edge case)."""
        batch, num_heads, head_dim = 1, 4, 64

        Q = torch.randn(batch, num_heads, 1, head_dim, dtype=torch.float16, device=device)
        K = torch.randn(batch, num_heads, 1, head_dim, dtype=torch.float16, device=device)
        V = torch.randn(batch, num_heads, 1, head_dim, dtype=torch.float16, device=device)

        scale = 1.0 / (head_dim ** 0.5)

        O_rocm = sage.attention_forward(Q, K, V, scale=scale, is_causal=False)
        O_pytorch = self.pytorch_attention_reference(Q, K, V, scale, is_causal=False)

        relative_error = torch.mean(torch.abs(O_rocm - O_pytorch)) / (torch.mean(torch.abs(O_pytorch)) + 1e-6)
        print(f"\nSingle token relative error: {relative_error:.4%}")

        assert relative_error < 0.05

    def test_attention_numerical_stability(self, sage, device):
        """Test attention with extreme values to check stability."""
        batch, num_heads, seq_len, head_dim = 1, 2, 32, 64

        # Test with large values
        Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device) * 10
        K = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device) * 10
        V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device) * 10

        scale = 1.0 / (head_dim ** 0.5)

        O_rocm = sage.attention_forward(Q, K, V, scale=scale, is_causal=False)

        # Check for NaN or Inf
        assert not torch.isnan(O_rocm).any(), "Output contains NaN"
        assert not torch.isinf(O_rocm).any(), "Output contains Inf"

        # Compare with PyTorch
        O_pytorch = self.pytorch_attention_reference(Q, K, V, scale, is_causal=False)
        relative_error = torch.mean(torch.abs(O_rocm - O_pytorch)) / (torch.mean(torch.abs(O_pytorch)) + 1e-6)

        print(f"\nNumerical stability test relative error: {relative_error:.4%}")
        assert relative_error < 0.1  # Allow more tolerance for extreme values


class TestMemoryEfficiency:
    """Tests for memory efficiency."""

    @pytest.fixture(scope="class")
    def sage(self):
        return SageAttentionROCm()

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def test_memory_usage_large_batch(self, sage, device):
        """Test that kernel can handle large batches efficiently."""
        # This should work within 87GB unified memory
        batch, num_heads, seq_len, head_dim = 8, 16, 512, 128

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create inputs
        Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        K = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

        scale = 1.0 / (head_dim ** 0.5)

        # Run attention
        O = sage.attention_forward(Q, K, V, scale=scale, is_causal=False)

        # Check memory usage
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak memory usage: {peak_memory:.2f} GB")

        # Should be reasonable
        assert peak_memory < 10.0, f"Memory usage too high: {peak_memory:.2f} GB"

    def test_int4_memory_vs_fp16(self, sage, device):
        """Test memory savings with INT4 vs FP16."""
        batch, tokens, heads, dim = 4, 1024, 16, 128

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # FP16 baseline
        x_fp16 = torch.randn(batch, tokens, heads, dim, dtype=torch.float16, device=device)
        fp16_memory = torch.cuda.max_memory_allocated()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # INT4 quantized
        x_quant, scales = sage.quantize_int4(x_fp16, block_size=16)
        int4_memory = torch.cuda.max_memory_allocated()

        memory_ratio = fp16_memory / int4_memory

        print(f"\nMemory comparison:")
        print(f"  FP16: {fp16_memory / 1e6:.2f} MB")
        print(f"  INT4: {int4_memory / 1e6:.2f} MB")
        print(f"  Ratio: {memory_ratio:.2f}x")

        # Should save significant memory
        assert memory_ratio > 2.5, f"Insufficient memory savings: {memory_ratio:.2f}x"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture(scope="class")
    def sage(self):
        return SageAttentionROCm()

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def test_minimum_dimensions(self, sage, device):
        """Test with minimum supported dimensions."""
        batch, num_heads, seq_len, head_dim = 1, 1, 1, 64

        Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        K = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
        V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

        O = sage.attention_forward(Q, K, V)

        assert O.shape == Q.shape
        assert not torch.isnan(O).any()

    def test_power_of_two_dimensions(self, sage, device):
        """Test with various power-of-2 dimensions."""
        head_dims = [64, 128]
        seq_lens = [64, 128, 256, 512]

        for head_dim in head_dims:
            for seq_len in seq_lens:
                Q = torch.randn(1, 4, seq_len, head_dim, dtype=torch.float16, device=device)
                K = torch.randn(1, 4, seq_len, head_dim, dtype=torch.float16, device=device)
                V = torch.randn(1, 4, seq_len, head_dim, dtype=torch.float16, device=device)

                O = sage.attention_forward(Q, K, V)

                assert O.shape == Q.shape
                assert not torch.isnan(O).any()
                print(f"  seq_len={seq_len}, head_dim={head_dim}: OK")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 80)
    print("SageAttention3 ROCm GPU Kernel Validation Tests")
    print("=" * 80)

    # Run pytest with verbose output
    args = [
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "--color=yes"
    ]

    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(run_all_tests())
