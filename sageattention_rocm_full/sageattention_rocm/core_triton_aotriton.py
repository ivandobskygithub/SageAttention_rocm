"""
SageAttention ROCm implementation with AOTriton support
Uses AMD's optimized Triton when available
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import os
import warnings

# Enable AOTriton experimental support
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Try to set ROCm backend
try:
    os.environ['TRITON_HIP_USE_ROCM'] = '1'
except:
    pass

# Check for AOTriton availability
AOTRITON_AVAILABLE = False
try:
    # Try fixed version first
    from .aotriton_compat_fixed import (
        get_aotriton_config,
        aotriton_attention,
        check_aotriton_available
    )
    aotriton_config = get_aotriton_config()
    AOTRITON_AVAILABLE = aotriton_config.available
    if AOTRITON_AVAILABLE:
        print("[INFO] Using AOTriton for GPU acceleration")
except ImportError:
    try:
        # Fall back to original
        from .aotriton_compat import (
            get_aotriton_config,
            aotriton_attention,
            check_aotriton_available
        )
        aotriton_config = get_aotriton_config()
        AOTRITON_AVAILABLE = aotriton_config.available
        if AOTRITON_AVAILABLE:
            print("[INFO] Using AOTriton for GPU acceleration")
    except ImportError as e:
        warnings.warn(f"AOTriton not available: {e}")

# Check if we have regular Triton as fallback
TRITON_AVAILABLE = False
if not AOTRITON_AVAILABLE:
    try:
        import triton
        TRITON_AVAILABLE = True
        print("Using standard Triton")
    except ImportError:
        warnings.warn("Neither AOTriton nor Triton available, using PyTorch fallback")

# Import implementations based on availability
if AOTRITON_AVAILABLE:
    # AOTriton path - use PyTorch's built-in ops with custom quantization
    from .triton.fallback_implementations import (
        quantize,
        dequantize
    )

    def attn_with_aotriton(q_int8, k_int8, v, q_scale, k_scale, sm_scale, is_causal, use_fp32_accum):
        """Wrapper to use AOTriton with quantized inputs."""
        # Dequantize
        batch_heads, seq_len, head_dim = q_int8.shape
        block_size = max(1, seq_len // max(1, q_scale.shape[0]))

        q = dequantize(q_int8, q_scale, block_size=block_size).to(v.dtype)
        k = dequantize(k_int8, k_scale, block_size=block_size).to(v.dtype)

        # Reshape for AOTriton [batch, heads, seq, dim] format
        # Assuming input is [batch*heads, seq, dim], need to infer batch
        # For simplicity, treat as batch=1 with batch*heads as heads
        q = q.unsqueeze(0)  # [1, batch*heads, seq, dim]
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

        # Use AOTriton attention
        output = aotriton_attention(q, k, v, is_causal=is_causal, scale=sm_scale)

        # Reshape back
        output = output.squeeze(0)  # [batch*heads, seq, dim]

        if use_fp32_accum:
            output = output.to(v.dtype)

        return output

elif TRITON_AVAILABLE:
    try:
        from .triton import (
            attn_qk_int8_per_block,
            attn_qk_int8_per_block_causal,
            quant_per_block,
        )
        from .triton.fallback_implementations import quantize
    except ImportError:
        TRITON_AVAILABLE = False

if not AOTRITON_AVAILABLE and not TRITON_AVAILABLE:
    # Use pure PyTorch fallback
    from .triton.fallback_implementations import (
        attn_qk_int8_per_block,
        attn_qk_int8_per_block_causal,
        quant_per_block,
        quantize,
    )


def get_rocm_device_info():
    """Get ROCm device information with AOTriton status."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_name = props.name

        info = {
            'name': device_name,
            'total_memory': props.total_memory,
            'multiprocessor_count': props.multi_processor_count,
            'triton_available': TRITON_AVAILABLE,
            'aotriton_available': AOTRITON_AVAILABLE,
        }

        # Add AOTriton status if available
        if AOTRITON_AVAILABLE:
            info['aotriton_status'] = aotriton_config.get_status()

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
    SageAttention with AOTriton support.

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

    # Status message
    if AOTRITON_AVAILABLE:
        pass  # Already printed status
    elif not TRITON_AVAILABLE:
        warnings.warn(
            "Using PyTorch fallback (slow). Install Triton or enable AOTriton.",
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
    if AOTRITON_AVAILABLE or TRITON_AVAILABLE:
        if hasattr(quant_per_block if TRITON_AVAILABLE else None, 'quantize'):
            q_int8, q_scale = quant_per_block.quantize(q, 128)
            k_int8, k_scale = quant_per_block.quantize(k, 128)
        else:
            q_int8, q_scale = quantize(q, 128)
            k_int8, k_scale = quantize(k, 128)
    else:
        # Use fallback quantization
        q_int8, q_scale = quantize(q, 128)
        k_int8, k_scale = quantize(k, 128)

    # Run attention
    if AOTRITON_AVAILABLE:
        # Use AOTriton path
        output = attn_with_aotriton(
            q_int8, k_int8, v,
            q_scale, k_scale,
            sm_scale,
            is_causal,
            pv_accum_dtype == "fp32"
        )
    elif is_causal:
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
    """Variable-length attention with AOTriton support."""

    if smooth_k:
        k = smooth_keys(k)

    # For simplicity, just use regular attention
    return sageattn(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        tensor_layout="NHD",
        is_causal=False,
        smooth_k=False,  # Already smoothed
        pv_accum_dtype=pv_accum_dtype
    ).squeeze(0)