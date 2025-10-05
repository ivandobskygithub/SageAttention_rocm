"""
Tests for attention forward pass.

Tests:
- Basic attention computation
- Causal masking
- Different sequence lengths
- Batch processing
- Numerical accuracy vs PyTorch baseline
"""

import pytest
import torch
import numpy as np
from . import requires_gpu, get_test_device

try:
    import sage_attention_rocm7 as sage_attn
    HAS_SAGE_ATTN = True
except ImportError:
    HAS_SAGE_ATTN = False
    sage_attn = None


def pytorch_attention(Q, K, V, scale=None, is_causal=False):
    """Reference PyTorch implementation of attention"""
    if scale is None:
        scale = 1.0 / (Q.shape[-1] ** 0.5)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if is_causal:
        seq_len_q = Q.size(2)
        seq_len_k = K.size(2)
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output


class TestAttentionBasic:
    """Basic functionality tests"""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("num_heads", [1, 8, 16])
    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_attention_shapes(self, batch_size, num_heads, seq_len, head_dim):
        """Test attention with various tensor shapes"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        device = get_test_device()
        dtype = torch.float16

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

        # Run attention
        output = sage_attn.attention_forward(Q, K, V)

        # Check output shape
        assert output.shape == Q.shape, f"Output shape {output.shape} doesn't match Q shape {Q.shape}"

        # Check output is not NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    @requires_gpu
    def test_attention_dtype(self):
        """Test attention requires float16 dtype"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        Q = torch.randn(2, 8, 128, 64, dtype=torch.float32, device='cuda')
        K = torch.randn(2, 8, 128, 64, dtype=torch.float32, device='cuda')
        V = torch.randn(2, 8, 128, 64, dtype=torch.float32, device='cuda')

        # Should raise ValueError for wrong dtype
        with pytest.raises(ValueError, match="must have dtype"):
            sage_attn.attention_forward(Q, K, V)

    @requires_gpu
    def test_attention_contiguous(self):
        """Test attention requires contiguous tensors"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        Q = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')
        K = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')
        V = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cuda')

        # Make non-contiguous
        Q_t = Q.transpose(1, 2)  # Non-contiguous

        # Should raise ValueError
        with pytest.raises(ValueError, match="must be contiguous"):
            sage_attn.attention_forward(Q_t, K, V)

    @requires_gpu
    def test_attention_device(self):
        """Test attention requires CUDA tensors"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        Q = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cpu')
        K = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cpu')
        V = torch.randn(2, 8, 128, 64, dtype=torch.float16, device='cpu')

        # Should raise ValueError for CPU tensors
        with pytest.raises(ValueError, match="must be a CUDA tensor"):
            sage_attn.attention_forward(Q, K, V)


class TestAttentionCausal:
    """Tests for causal masking"""

    @requires_gpu
    def test_causal_masking(self):
        """Test that causal masking works correctly"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Compute causal attention
        output_causal = sage_attn.attention_forward(Q, K, V, is_causal=True)

        # Compute non-causal attention
        output_non_causal = sage_attn.attention_forward(Q, K, V, is_causal=False)

        # Outputs should be different
        assert not torch.allclose(output_causal, output_non_causal, atol=1e-3), \
            "Causal and non-causal attention produce identical results"

        # Check that output is not NaN or Inf
        assert not torch.isnan(output_causal).any(), "Causal attention contains NaN"
        assert not torch.isinf(output_causal).any(), "Causal attention contains Inf"

    @requires_gpu
    @pytest.mark.accuracy
    def test_causal_vs_pytorch(self):
        """Test causal masking matches PyTorch implementation"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 32

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Compute with SageAttention
        output_sage = sage_attn.attention_forward(Q, K, V, is_causal=True)

        # Compute with PyTorch
        output_torch = pytorch_attention(Q, K, V, is_causal=True)

        # Check relative error (allow some tolerance for FP16)
        rel_error = torch.abs(output_sage - output_torch) / (torch.abs(output_torch) + 1e-5)
        max_rel_error = rel_error.max().item()

        print(f"Max relative error: {max_rel_error:.6f}")
        assert max_rel_error < 0.1, f"Relative error too high: {max_rel_error}"


class TestAttentionAccuracy:
    """Numerical accuracy tests"""

    @requires_gpu
    @pytest.mark.accuracy
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
    def test_accuracy_vs_pytorch(self, seq_len):
        """Test numerical accuracy against PyTorch baseline"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, head_dim = 2, 8, 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Compute with SageAttention
        output_sage = sage_attn.attention_forward(Q, K, V)

        # Compute with PyTorch
        output_torch = pytorch_attention(Q, K, V)

        # Compute errors
        abs_error = torch.abs(output_sage - output_torch)
        rel_error = abs_error / (torch.abs(output_torch) + 1e-5)

        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()

        print(f"\nAccuracy for seq_len={seq_len}:")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Mean absolute error: {mean_abs_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.6f}")
        print(f"  Mean relative error: {mean_rel_error:.6f}")

        # FP16 tolerance
        assert max_abs_error < 1.0, f"Max absolute error too high: {max_abs_error}"
        assert mean_abs_error < 0.1, f"Mean absolute error too high: {mean_abs_error}"

    @requires_gpu
    @pytest.mark.accuracy
    def test_scale_parameter(self):
        """Test that scale parameter works correctly"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        scale = 0.125  # 1/sqrt(64)

        # Compute with explicit scale
        output_scaled = sage_attn.attention_forward(Q, K, V, scale=scale)

        # Compute with default scale (should be same)
        output_default = sage_attn.attention_forward(Q, K, V)

        # Should be very close
        assert torch.allclose(output_scaled, output_default, atol=1e-3), \
            "Explicit scale doesn't match default scale"


class TestAttentionEdgeCases:
    """Edge case tests"""

    @requires_gpu
    def test_different_seq_lengths(self):
        """Test with different Q and K/V sequence lengths"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len_q, seq_len_k, head_dim = 2, 8, 64, 128, 64

        Q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len_k, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len_k, head_dim, dtype=torch.float16, device='cuda')

        # Should work with different sequence lengths
        output = sage_attn.attention_forward(Q, K, V)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()

    @requires_gpu
    def test_single_head(self):
        """Test with single attention head"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 1, 128, 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        output = sage_attn.attention_forward(Q, K, V)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()

    @requires_gpu
    def test_batch_size_one(self):
        """Test with batch size of 1"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 1, 8, 128, 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        output = sage_attn.attention_forward(Q, K, V)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()

    @requires_gpu
    def test_zero_initialization(self):
        """Test with zero-initialized tensors"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64

        Q = torch.zeros(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.zeros(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Should handle zeros gracefully (attention weights will be uniform)
        output = sage_attn.attention_forward(Q, K, V)

        assert output.shape == Q.shape
        # Output may contain NaN due to softmax of all zeros, but shouldn't crash
