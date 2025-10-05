"""
Integration tests for SageAttention3 ROCm port.

Tests the complete workflow:
- Attention with INT4 quantization
- End-to-end accuracy
- Real-world usage patterns
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
    """Reference PyTorch implementation"""
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


class TestInt4AttentionIntegration:
    """Integration tests for INT4 quantized attention"""

    @requires_gpu
    @pytest.mark.accuracy
    def test_int4_attention_vs_fp16(self):
        """Compare INT4 quantized attention with FP16 baseline"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # FP16 attention (baseline)
        output_fp16 = sage_attn.attention_forward(Q, K, V, use_int4_quantization=False)

        # INT4 quantized attention
        output_int4 = sage_attn.attention_forward(Q, K, V, use_int4_quantization=True)

        # Compute error
        abs_error = torch.abs(output_fp16 - output_int4)
        rel_error = abs_error / (torch.abs(output_fp16) + 1e-5)

        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()

        print(f"\nINT4 vs FP16 Attention:")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Mean absolute error: {mean_abs_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.6f}")
        print(f"  Mean relative error: {mean_rel_error:.6f}")

        # INT4 quantization will have some error, but should be bounded
        assert mean_rel_error < 0.15, f"Mean relative error too high: {mean_rel_error}"

    @requires_gpu
    def test_int4_memory_savings(self):
        """Verify memory savings with INT4 quantization"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 4, 16, 1024, 128

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Original K, V size
        original_kv_bytes = (K.numel() + V.numel()) * K.element_size()

        # Quantize K and V
        K_flat = K.view(-1, head_dim)
        V_flat = V.view(-1, head_dim)

        K_q, K_scales = sage_attn.quantize_int4(K_flat)
        V_q, V_scales = sage_attn.quantize_int4(V_flat)

        # Quantized size
        quantized_bytes = (
            K_q.numel() * K_q.element_size() +
            V_q.numel() * V_q.element_size() +
            K_scales.numel() * K_scales.element_size() +
            V_scales.numel() * V_scales.element_size()
        )

        compression_ratio = original_kv_bytes / quantized_bytes

        print(f"\nMemory savings (K, V):")
        print(f"  Original: {original_kv_bytes:,} bytes")
        print(f"  Quantized: {quantized_bytes:,} bytes")
        print(f"  Compression: {compression_ratio:.2f}x")
        print(f"  Savings: {(1 - quantized_bytes/original_kv_bytes) * 100:.1f}%")

        # Should achieve at least 3x compression
        assert compression_ratio > 3.0, f"Compression ratio too low: {compression_ratio:.2f}x"


class TestRealWorldPatterns:
    """Tests for real-world usage patterns"""

    @requires_gpu
    def test_llm_inference_pattern(self):
        """Test pattern typical in LLM inference"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        # Typical LLM settings
        batch_size = 1
        num_heads = 32
        seq_len = 2048
        head_dim = 128

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Causal attention (typical for autoregressive generation)
        output = sage_attn.attention_forward(Q, K, V, is_causal=True, use_int4_quantization=True)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @requires_gpu
    def test_incremental_generation(self):
        """Test incremental decoding pattern (single token generation)"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size = 1
        num_heads = 16
        kv_seq_len = 512  # Previous context
        head_dim = 64

        # Single query token (next token prediction)
        Q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, kv_seq_len, head_dim, dtype=torch.float16, device='cuda')

        output = sage_attn.attention_forward(Q, K, V, is_causal=True)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()

    @requires_gpu
    def test_multi_query_attention(self):
        """Test multi-query attention pattern (shared K, V across heads)"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size = 2
        num_q_heads = 32
        num_kv_heads = 4  # Fewer KV heads (shared)
        seq_len = 512
        head_dim = 64

        Q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Repeat K, V to match Q heads
        K_expanded = K.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        V_expanded = V.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

        output = sage_attn.attention_forward(Q, K_expanded, V_expanded)

        assert output.shape == Q.shape
        assert not torch.isnan(output).any()


class TestEndToEnd:
    """End-to-end workflow tests"""

    @requires_gpu
    @pytest.mark.accuracy
    def test_full_pipeline_accuracy(self):
        """Test complete pipeline: quantize K,V -> attention -> compare"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Baseline: PyTorch FP16 attention
        output_baseline = pytorch_attention(Q, K, V)

        # SageAttention FP16
        output_sage_fp16 = sage_attn.attention_forward(Q, K, V, use_int4_quantization=False)

        # SageAttention INT4
        output_sage_int4 = sage_attn.attention_forward(Q, K, V, use_int4_quantization=True)

        # Compare SageAttention FP16 with PyTorch baseline
        error_fp16 = torch.abs(output_sage_fp16 - output_baseline)
        max_error_fp16 = error_fp16.max().item()
        mean_error_fp16 = error_fp16.mean().item()

        print(f"\nSageAttention FP16 vs PyTorch:")
        print(f"  Max error: {max_error_fp16:.6f}")
        print(f"  Mean error: {mean_error_fp16:.6f}")

        # Compare INT4 with FP16
        error_int4 = torch.abs(output_sage_int4 - output_sage_fp16)
        max_error_int4 = error_int4.max().item()
        mean_error_int4 = error_int4.mean().item()

        print(f"\nSageAttention INT4 vs FP16:")
        print(f"  Max error: {max_error_int4:.6f}")
        print(f"  Mean error: {mean_error_int4:.6f}")

        # FP16 should match PyTorch closely
        assert mean_error_fp16 < 0.1, f"FP16 error too high: {mean_error_fp16}"

        # INT4 should be reasonable
        assert mean_error_int4 < 0.2, f"INT4 error too high: {mean_error_int4}"

    @requires_gpu
    def test_batch_processing(self):
        """Test processing multiple batches"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        num_heads, seq_len, head_dim = 8, 256, 64

        # Process multiple batches
        for batch_size in [1, 2, 4, 8]:
            Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
            K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
            V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

            output = sage_attn.attention_forward(Q, K, V)

            assert output.shape == Q.shape
            assert not torch.isnan(output).any()

    @requires_gpu
    def test_gradient_checkpointing_compatible(self):
        """Test that attention works in no_grad context (inference)"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64

        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

        # Should work in no_grad context
        with torch.no_grad():
            output = sage_attn.attention_forward(Q, K, V)
            assert output.shape == Q.shape

        # Should also work in inference mode
        with torch.inference_mode():
            output = sage_attn.attention_forward(Q, K, V)
            assert output.shape == Q.shape
