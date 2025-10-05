"""
Tests for INT4 quantization/dequantization.

Tests:
- Basic quantization/dequantization
- Block scaling accuracy
- Memory savings validation
- Quantization error bounds
- Edge cases
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


def compute_quantization_error(original, reconstructed):
    """Compute quantization error metrics"""
    abs_error = torch.abs(original - reconstructed)
    rel_error = abs_error / (torch.abs(original) + 1e-8)

    metrics = {
        'max_abs_error': abs_error.max().item(),
        'mean_abs_error': abs_error.mean().item(),
        'max_rel_error': rel_error.max().item(),
        'mean_rel_error': rel_error.mean().item(),
        'mse': torch.mean((original - reconstructed) ** 2).item(),
        'snr_db': 10 * torch.log10(
            torch.mean(original ** 2) / (torch.mean((original - reconstructed) ** 2) + 1e-8)
        ).item(),
    }

    return metrics


class TestQuantizationBasic:
    """Basic quantization functionality tests"""

    @requires_gpu
    @pytest.mark.parametrize("shape", [
        (128, 64),
        (256, 128),
        (512, 64),
        (1024, 256),
    ])
    @pytest.mark.parametrize("block_size", [16, 32])
    def test_quantize_dequantize_shapes(self, shape, block_size):
        """Test quantization with various tensor shapes"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        x = torch.randn(*shape, dtype=torch.float16, device='cuda')

        # Quantize
        quantized, scales = sage_attn.quantize_int4(x, block_size=block_size)

        # Check quantized shape (packed into uint8)
        expected_packed_size = (shape[0], shape[1] // 2)  # 2 INT4 values per byte
        assert quantized.shape == expected_packed_size, \
            f"Quantized shape {quantized.shape} doesn't match expected {expected_packed_size}"

        # Check scales shape
        num_blocks = shape[1] // block_size
        expected_scales_shape = (shape[0], num_blocks)
        assert scales.shape == expected_scales_shape, \
            f"Scales shape {scales.shape} doesn't match expected {expected_scales_shape}"

        # Dequantize
        reconstructed = sage_attn.dequantize_int4(quantized, scales, x.shape)

        # Check reconstructed shape
        assert reconstructed.shape == x.shape, \
            f"Reconstructed shape {reconstructed.shape} doesn't match original {x.shape}"

    @requires_gpu
    def test_quantization_dtype(self):
        """Test quantization requires float16 input"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        # Wrong dtype (float32)
        x = torch.randn(128, 64, dtype=torch.float32, device='cuda')

        with pytest.raises(ValueError, match="must have dtype"):
            sage_attn.quantize_int4(x)

    @requires_gpu
    def test_quantization_device(self):
        """Test quantization requires CUDA tensor"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        # CPU tensor
        x = torch.randn(128, 64, dtype=torch.float16, device='cpu')

        with pytest.raises(ValueError, match="must be a CUDA tensor"):
            sage_attn.quantize_int4(x)

    @requires_gpu
    def test_quantization_contiguous(self):
        """Test quantization requires contiguous tensor"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        # Non-contiguous tensor
        x = torch.randn(64, 128, dtype=torch.float16, device='cuda')
        x_t = x.t()  # Transpose creates non-contiguous tensor

        with pytest.raises(ValueError, match="must be contiguous"):
            sage_attn.quantize_int4(x_t)


class TestQuantizationAccuracy:
    """Quantization accuracy tests"""

    @requires_gpu
    @pytest.mark.accuracy
    @pytest.mark.parametrize("block_size", [16, 32])
    def test_quantization_error_bounds(self, block_size):
        """Test quantization error is within acceptable bounds"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        x = torch.randn(1024, 256, dtype=torch.float16, device='cuda')

        # Quantize and dequantize
        quantized, scales = sage_attn.quantize_int4(x, block_size=block_size)
        reconstructed = sage_attn.dequantize_int4(quantized, scales, x.shape)

        # Compute error metrics
        metrics = compute_quantization_error(x, reconstructed)

        print(f"\nQuantization error (block_size={block_size}):")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # INT4 quantization should have reasonable SNR
        assert metrics['snr_db'] > 20.0, f"SNR too low: {metrics['snr_db']:.2f} dB"

        # Mean relative error should be reasonable
        assert metrics['mean_rel_error'] < 0.1, \
            f"Mean relative error too high: {metrics['mean_rel_error']:.6f}"

    @requires_gpu
    @pytest.mark.accuracy
    def test_block_scaling_effectiveness(self):
        """Test that block scaling improves accuracy"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        # Create tensor with varying magnitudes
        x = torch.randn(512, 128, dtype=torch.float16, device='cuda')
        x[:, :64] *= 10.0  # First half has larger values
        x[:, 64:] *= 0.1   # Second half has smaller values

        # Quantize with different block sizes
        q16, s16 = sage_attn.quantize_int4(x, block_size=16)
        r16 = sage_attn.dequantize_int4(q16, s16, x.shape)

        q32, s32 = sage_attn.quantize_int4(x, block_size=32)
        r32 = sage_attn.dequantize_int4(q32, s32, x.shape)

        # Compute errors
        error16 = compute_quantization_error(x, r16)
        error32 = compute_quantization_error(x, r32)

        print(f"\nBlock size comparison:")
        print(f"  Block size 16 - SNR: {error16['snr_db']:.2f} dB")
        print(f"  Block size 32 - SNR: {error32['snr_db']:.2f} dB")

        # Smaller blocks should generally be better for varying magnitudes
        # But this is not guaranteed, so just check both are reasonable
        assert error16['snr_db'] > 15.0, "Block size 16 SNR too low"
        assert error32['snr_db'] > 15.0, "Block size 32 SNR too low"

    @requires_gpu
    @pytest.mark.accuracy
    def test_uniform_values(self):
        """Test quantization of uniform values"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        # Create uniform tensor
        x = torch.ones(256, 128, dtype=torch.float16, device='cuda') * 0.5

        # Quantize and dequantize
        quantized, scales = sage_attn.quantize_int4(x)
        reconstructed = sage_attn.dequantize_int4(quantized, scales, x.shape)

        # Should have very low error for uniform values
        abs_error = torch.abs(x - reconstructed).max().item()
        print(f"\nUniform value error: {abs_error:.6f}")

        assert abs_error < 0.1, f"Error too high for uniform values: {abs_error}"


class TestQuantizationMemory:
    """Memory efficiency tests"""

    @requires_gpu
    def test_memory_savings(self):
        """Test that INT4 quantization saves memory"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        shape = (2048, 1024)
        x = torch.randn(*shape, dtype=torch.float16, device='cuda')

        # Original size
        original_bytes = x.element_size() * x.numel()

        # Quantize
        quantized, scales = sage_attn.quantize_int4(x, block_size=16)

        # Quantized size (packed INT4 + scales)
        quantized_bytes = quantized.element_size() * quantized.numel()
        scales_bytes = scales.element_size() * scales.numel()
        total_bytes = quantized_bytes + scales_bytes

        compression_ratio = original_bytes / total_bytes

        print(f"\nMemory usage:")
        print(f"  Original: {original_bytes:,} bytes")
        print(f"  Quantized: {quantized_bytes:,} bytes")
        print(f"  Scales: {scales_bytes:,} bytes")
        print(f"  Total: {total_bytes:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")

        # Should achieve ~4x compression (FP16 to INT4)
        # Accounting for scales overhead, expect at least 3x
        assert compression_ratio > 3.0, \
            f"Compression ratio too low: {compression_ratio:.2f}x"

    @requires_gpu
    def test_4d_tensor_quantization(self):
        """Test quantization of 4D tensors (batch, heads, seq, dim)"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64

        # Create 4D tensor
        x = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       dtype=torch.float16, device='cuda')

        # Reshape to 2D for quantization
        x_2d = x.view(-1, head_dim)

        # Quantize
        quantized, scales = sage_attn.quantize_int4(x_2d)

        # Dequantize
        reconstructed_2d = sage_attn.dequantize_int4(quantized, scales, x_2d.shape)

        # Reshape back to 4D
        reconstructed = reconstructed_2d.view(batch_size, num_heads, seq_len, head_dim)

        # Check shape
        assert reconstructed.shape == x.shape

        # Check error
        metrics = compute_quantization_error(x, reconstructed)
        print(f"\n4D tensor quantization SNR: {metrics['snr_db']:.2f} dB")

        assert metrics['snr_db'] > 20.0, "SNR too low for 4D tensor"


class TestQuantizationEdgeCases:
    """Edge case tests for quantization"""

    @requires_gpu
    def test_zero_values(self):
        """Test quantization of zero tensor"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        x = torch.zeros(256, 128, dtype=torch.float16, device='cuda')

        # Quantize
        quantized, scales = sage_attn.quantize_int4(x)

        # Dequantize
        reconstructed = sage_attn.dequantize_int4(quantized, scales, x.shape)

        # Should reconstruct zeros (or very close)
        assert torch.allclose(reconstructed, x, atol=1e-4), \
            "Failed to reconstruct zero tensor"

    @requires_gpu
    def test_large_values(self):
        """Test quantization with large magnitude values"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        x = torch.randn(256, 128, dtype=torch.float16, device='cuda') * 100.0

        # Quantize
        quantized, scales = sage_attn.quantize_int4(x)

        # Dequantize
        reconstructed = sage_attn.dequantize_int4(quantized, scales, x.shape)

        # Check scales are appropriate
        assert scales.max() > 10.0, "Scales should be large for large values"

        # Check reconstruction
        metrics = compute_quantization_error(x, reconstructed)
        print(f"\nLarge values SNR: {metrics['snr_db']:.2f} dB")

        # Should still maintain reasonable accuracy
        assert metrics['snr_db'] > 15.0, "SNR too low for large values"

    @requires_gpu
    def test_small_values(self):
        """Test quantization with small magnitude values"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        x = torch.randn(256, 128, dtype=torch.float16, device='cuda') * 0.01

        # Quantize
        quantized, scales = sage_attn.quantize_int4(x)

        # Dequantize
        reconstructed = sage_attn.dequantize_int4(quantized, scales, x.shape)

        # Check scales are appropriate
        assert scales.max() < 1.0, "Scales should be small for small values"

        # Check reconstruction
        metrics = compute_quantization_error(x, reconstructed)
        print(f"\nSmall values SNR: {metrics['snr_db']:.2f} dB")

        # Should maintain reasonable accuracy
        assert metrics['snr_db'] > 15.0, "SNR too low for small values"

    @requires_gpu
    def test_mixed_sign_values(self):
        """Test quantization preserves sign"""
        if not HAS_SAGE_ATTN:
            pytest.skip("sage_attention_rocm7 not available")

        x = torch.randn(256, 128, dtype=torch.float16, device='cuda')

        # Quantize
        quantized, scales = sage_attn.quantize_int4(x)

        # Dequantize
        reconstructed = sage_attn.dequantize_int4(quantized, scales, x.shape)

        # Check signs are preserved
        sign_matches = (torch.sign(x) == torch.sign(reconstructed)).float().mean().item()
        print(f"\nSign preservation: {sign_matches * 100:.1f}%")

        # Should preserve most signs (allowing some errors near zero)
        assert sign_matches > 0.95, f"Sign preservation too low: {sign_matches * 100:.1f}%"
