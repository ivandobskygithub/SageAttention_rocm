"""
Simplified Triton core that works with or without Triton installed
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import os
import warnings

# Try to set ROCm backend if Triton is available
try:
    os.environ['TRITON_HIP_USE_ROCM'] = '1'
except:
    pass

# Check if we have Triton or need to use fallbacks
TRITON_AVAILABLE = False
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    warnings.warn("Triton not available, using PyTorch fallback implementations")

# Import our compatibility layer
from .triton_compat import check_triton_availability

# Import implementations based on availability
if TRITON_AVAILABLE:
    try:
        from .triton import (
            attn_qk_int8_per_block,
            attn_qk_int8_per_block_causal,
            quant_per_block,
        )
    except ImportError:
        TRITON_AVAILABLE = False

if not TRITON_AVAILABLE:
    # Use fallback implementations
    from .triton.fallback_implementations import (
        attn_qk_int8_per_block,
        attn_qk_int8_per_block_causal,
        quant_per_block,
        quantize,
    )


def get_rocm_device_info():
    """Get ROCm device information."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_name = props.name

        info = {
            'name': device_name,
            'total_memory': props.total_memory,
            'multiprocessor_count': props.multi_processor_count,
            'triton_available': TRITON_AVAILABLE,
        }

        # Determine architecture
        device_name_lower = device_name.lower()
        if 'mi300' in device_name_lower:
            info['arch'] = 'gfx940'
        elif 'mi250' in device_name_lower or 'mi200' in device_name_lower:
            info['arch'] = 'gfx90a'
        elif 'mi100' in device_name_lower:
            info['arch'] = 'gfx908'
        elif '7900' in device_name_lower:
            info['arch'] = 'gfx1100'
        else:
            info['arch'] = 'unknown'

        return info
    return None


def smooth_keys(k: torch.Tensor) -> torch.Tensor:
    """Apply smoothing to keys."""
    k_mean = k.mean(dim=-1, keepdim=True)
    return k - k_mean


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    smooth_k: bool = True,
    pv_accum_dtype: str = "fp32",
) -> torch.Tensor:
    """
    SageAttention function that works with or without Triton.

    Args:
        q, k, v: Query, Key, Value tensors
        tensor_layout: "HND" or "NHD"
        is_causal: Whether to apply causal masking
        smooth_k: Whether to smooth keys
        pv_accum_dtype: Accumulation dtype

    Returns:
        Attention output
    """

    # Input validation
    assert tensor_layout in ["HND", "NHD"], f"Invalid tensor_layout: {tensor_layout}"
    assert q.is_cuda, "Tensors must be on CUDA/ROCm device"

    # Warn if using fallback
    if not TRITON_AVAILABLE:
        warnings.warn(
            "Using PyTorch fallback implementation (slow). "
            "Install Triton for GPU acceleration.",
            stacklevel=2
        )

    # Handle tensor layout
    original_shape = None
    if tensor_layout == "NHD":
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        seq_len_q = q.shape[2]
        seq_len_kv = k.shape[2]
        head_dim = q.shape[3]
        original_shape = q.shape

        # Reshape to [batch*heads, seq_len, head_dim]
        q = q.view(batch_size * num_heads, seq_len_q, head_dim)
        k = k.view(batch_size * num_heads, seq_len_kv, head_dim)
        v = v.view(batch_size * num_heads, seq_len_kv, head_dim)
    else:
        num_heads = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_kv = k.shape[1]
        head_dim = q.shape[2]

    # Apply key smoothing
    if smooth_k:
        k = smooth_keys(k)

    # Scale factor
    sm_scale = head_dim ** -0.5

    # Quantize Q and K
    if TRITON_AVAILABLE:
        q_int8, q_scale = quant_per_block.quantize(q, 128)
        k_int8, k_scale = quant_per_block.quantize(k, 128)
    else:
        # Use fallback quantization
        q_int8, q_scale = quantize(q, 128)
        k_int8, k_scale = quantize(k, 128)

    # Run attention
    if is_causal:
        output = attn_qk_int8_per_block_causal.forward(
            q_int8, k_int8, v,
            q_scale, k_scale,
            sm_scale,
            pv_accum_dtype == "fp32"
        )
    else:
        output = attn_qk_int8_per_block.forward(
            q_int8, k_int8, v,
            q_scale, k_scale,
            sm_scale,
            pv_accum_dtype == "fp32"
        )

    # Reshape output back if needed
    if tensor_layout == "NHD" and original_shape is not None:
        output = output.view(original_shape)

    return output


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    smooth_k: bool = True,
    pv_accum_dtype: str = "fp32",
) -> torch.Tensor:
    """Variable-length attention (simplified for fallback)."""

    if smooth_k:
        k = smooth_keys(k)

    # For simplicity, just use regular attention in fallback mode
    return sageattn(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        tensor_layout="NHD",
        is_causal=False,
        smooth_k=False,  # Already smoothed
        pv_accum_dtype=pv_accum_dtype
    ).squeeze(0)