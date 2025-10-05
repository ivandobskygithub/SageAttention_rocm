"""
High-level Python API for SageAttention3 ROCm kernels

This module provides PyTorch-compatible functions that wrap the HIP kernels.
"""

import torch
from typing import Optional, Tuple

# Import C extension
try:
    from . import _C
except ImportError:
    _C = None
    import warnings
    warnings.warn(
        "sage_attention_rocm7 C extension not available. "
        "Falling back to PyTorch native implementation.",
        ImportWarning
    )

def _check_extension():
    """Verify C extension is loaded"""
    if _C is None:
        raise RuntimeError(
            "sage_attention_rocm7 C extension not loaded. "
            "Please rebuild with: python setup.py build_ext --inplace"
        )

def _check_tensor(tensor: torch.Tensor, name: str, dtype: torch.dtype = torch.float16):
    """Validate tensor properties"""
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")

    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")

    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")

def attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    delta_s: Optional[torch.Tensor] = None,
    use_int4_quantization: bool = False,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with optional INT4 quantization.

    This function implements the attention mechanism:
        Attention(Q, K, V) = softmax(Q @ K^T / scale) @ V

    With optional INT4 quantization for K and V to reduce memory bandwidth.

    Args:
        Q: Query tensor of shape [batch, num_heads, seq_len_q, head_dim]
        K: Key tensor of shape [batch, num_heads, seq_len_k, head_dim]
        V: Value tensor of shape [batch, num_heads, seq_len_k, head_dim]
        scale: Scaling factor for attention scores. Default: 1.0 / sqrt(head_dim)
        is_causal: If True, apply causal masking
        delta_s: Optional per-block mean subtraction values
        use_int4_quantization: If True, use INT4 quantization for K and V

    Returns:
        Output tensor of shape [batch, num_heads, seq_len_q, head_dim]

    Example:
        >>> Q = torch.randn(2, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> K = torch.randn(2, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> V = torch.randn(2, 8, 512, 64, dtype=torch.float16, device='cuda')
        >>> output = attention_forward(Q, K, V)
    """
    # Validate inputs
    _check_tensor(Q, 'Q')
    _check_tensor(K, 'K')
    _check_tensor(V, 'V')

    # Check shapes
    if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
        raise ValueError("Q, K, V must be 4D tensors")

    batch_size, num_heads, seq_len_q, head_dim = Q.shape
    _, _, seq_len_k, _ = K.shape

    if K.shape[0] != batch_size or K.shape[1] != num_heads or K.shape[3] != head_dim:
        raise ValueError(f"K shape mismatch: expected [{batch_size}, {num_heads}, ?, {head_dim}], got {K.shape}")

    if V.shape != K.shape:
        raise ValueError(f"V shape must match K shape, got V={V.shape}, K={K.shape}")

    # Default scale
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # Validate delta_s if provided
    if delta_s is not None:
        _check_tensor(delta_s, 'delta_s', dtype=torch.float32)

    # Allocate output tensor
    output = torch.zeros_like(Q)

    # Use compiled kernel if available
    if _C is not None:
        _check_extension()

        # Call HIP kernel
        _C.attention_forward(
            Q, K, V, output,
            scale,
            is_causal,
            delta_s,
            use_int4_quantization,
        )
    else:
        # Fallback to PyTorch implementation
        output = _attention_forward_pytorch(Q, K, V, scale, is_causal)

    return output

def _attention_forward_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """
    Fallback PyTorch implementation of attention.

    Used when compiled kernels are not available.
    """
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        seq_len_q = Q.size(2)
        seq_len_k = K.size(2)
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, V)

    return output

def quantize_int4(
    tensor: torch.Tensor,
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP16 tensor to INT4 with block scaling.

    Args:
        tensor: Input tensor of shape [..., features]
        block_size: Number of elements per quantization block (default: 16)

    Returns:
        quantized: INT4 values packed into uint8 tensor (2 values per byte)
        scales: FP16 scale factors, one per block

    Example:
        >>> x = torch.randn(1024, 128, dtype=torch.float16, device='cuda')
        >>> quantized, scales = quantize_int4(x)
        >>> print(quantized.shape)  # [1024, 64] - half the size due to packing
        >>> print(scales.shape)     # [1024, 8] - one per 16 elements
    """
    _check_tensor(tensor, 'tensor')

    if _C is not None:
        _check_extension()
        return _C.quantize_int4(tensor, block_size)
    else:
        # Fallback: return original tensor (no quantization)
        import warnings
        warnings.warn("Quantization kernel not available, returning original tensor", RuntimeWarning)
        return tensor, torch.ones(1, dtype=torch.float16, device=tensor.device)

def dequantize_int4(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    output_shape: Tuple[int, ...],
) -> torch.Tensor:
    """
    Dequantize INT4 tensor back to FP16.

    Args:
        quantized: Packed INT4 values (uint8 tensor)
        scales: FP16 scale factors
        output_shape: Shape of the dequantized output

    Returns:
        Dequantized FP16 tensor

    Example:
        >>> quantized, scales = quantize_int4(x)
        >>> reconstructed = dequantize_int4(quantized, scales, x.shape)
    """
    _check_tensor(quantized, 'quantized', dtype=torch.uint8)
    _check_tensor(scales, 'scales')

    if _C is not None:
        _check_extension()
        return _C.dequantize_int4(quantized, scales, output_shape)
    else:
        # Fallback: return quantized tensor as-is
        import warnings
        warnings.warn("Dequantization kernel not available", RuntimeWarning)
        return quantized.to(torch.float16)

# Autograd function for attention (for future gradient support)
class SageAttentionFunction(torch.autograd.Function):
    """
    Autograd function for SageAttention with INT4 quantization.

    This enables automatic differentiation through the attention operation.
    Currently only forward pass is implemented.
    """

    @staticmethod
    def forward(ctx, Q, K, V, scale, is_causal, delta_s, use_int4_quantization):
        """Forward pass"""
        output = attention_forward(Q, K, V, scale, is_causal, delta_s, use_int4_quantization)

        # Save for backward (not yet implemented)
        ctx.save_for_backward(Q, K, V)
        ctx.scale = scale
        ctx.is_causal = is_causal

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass (not yet implemented)"""
        raise NotImplementedError(
            "Backward pass for SageAttention is not yet implemented. "
            "Use with torch.no_grad() for inference only."
        )

def sage_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    delta_s: Optional[torch.Tensor] = None,
    use_int4_quantization: bool = False,
) -> torch.Tensor:
    """
    Autograd-enabled version of attention_forward.

    Same interface as attention_forward, but with gradient support (when implemented).
    """
    if not torch.is_grad_enabled():
        # No gradients needed, use direct function
        return attention_forward(Q, K, V, scale, is_causal, delta_s, use_int4_quantization)
    else:
        # Use autograd function
        return SageAttentionFunction.apply(Q, K, V, scale, is_causal, delta_s, use_int4_quantization)
