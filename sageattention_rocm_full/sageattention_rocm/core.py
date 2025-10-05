"""
Core attention implementations for ROCm
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings

# Try to import HIP kernels
HAS_HIP_KERNELS = False
try:
    import sageattention_rocm._qattn_hip as qattn_hip
    HAS_HIP_KERNELS = True
except ImportError:
    warnings.warn("HIP kernels not available. Falling back to Triton implementations.")

# Import Triton kernels (these should work on ROCm with Triton ROCm backend)
from . import triton as triton_kernels


def get_device_capability():
    """Get ROCm device capability."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # Map device names to architectures
        device_name = props.name.lower()

        if 'mi300' in device_name:
            return 'gfx940'
        elif 'mi250' in device_name or 'mi200' in device_name:
            return 'gfx90a'
        elif 'mi100' in device_name:
            return 'gfx908'
        elif '7900' in device_name:
            return 'gfx1100'
        elif '6900' in device_name or '6800' in device_name:
            return 'gfx1030'

    return None


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    smooth_k: bool = True,
    pv_accum_dtype: str = "fp32",
    attn_backend: str = "auto"
) -> torch.Tensor:
    """
    Main SageAttention function for ROCm.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        tensor_layout: Layout of input tensors ("HND" or "NHD")
        is_causal: Whether to apply causal masking
        smooth_k: Whether to apply key smoothing
        pv_accum_dtype: Accumulation dtype for P*V ("fp32" or "fp16")
        attn_backend: Backend to use ("auto", "triton", "hip")

    Returns:
        Output tensor
    """

    # Input validation
    assert tensor_layout in ["HND", "NHD"], f"Invalid tensor_layout: {tensor_layout}"
    assert pv_accum_dtype in ["fp32", "fp16"], f"Invalid pv_accum_dtype: {pv_accum_dtype}"

    # Determine backend
    if attn_backend == "auto":
        if HAS_HIP_KERNELS:
            device_cap = get_device_capability()
            if device_cap in ['gfx940', 'gfx941', 'gfx90a']:
                attn_backend = "hip"
            else:
                attn_backend = "triton"
        else:
            attn_backend = "triton"

    # Reshape tensors based on layout
    if tensor_layout == "HND":
        batch_size = 1
        num_heads = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_kv = k.shape[1]
        head_dim = q.shape[2]
    else:  # NHD
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        seq_len_q = q.shape[2]
        seq_len_kv = k.shape[2]
        head_dim = q.shape[3]

        # Reshape to HND for processing
        q = q.view(batch_size * num_heads, seq_len_q, head_dim)
        k = k.view(batch_size * num_heads, seq_len_kv, head_dim)
        v = v.view(batch_size * num_heads, seq_len_kv, head_dim)

    # Apply key smoothing if requested
    if smooth_k:
        k = smooth_keys(k)

    # Select implementation
    if attn_backend == "hip" and HAS_HIP_KERNELS:
        output = sageattn_qk_int8_pv_fp16_hip(
            q, k, v, is_causal=is_causal, pv_accum_dtype=pv_accum_dtype
        )
    else:
        # Use Triton implementation
        output = sageattn_qk_int8_pv_fp16_triton(
            q, k, v, is_causal=is_causal, pv_accum_dtype=pv_accum_dtype
        )

    # Reshape output back if needed
    if tensor_layout == "NHD":
        output = output.view(batch_size, num_heads, seq_len_q, head_dim)

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
    attn_backend: str = "auto"
) -> torch.Tensor:
    """
    Variable-length SageAttention for ROCm.

    Args:
        q: Query tensor (total_q, num_heads, head_dim)
        k: Key tensor (total_k, num_heads, head_dim)
        v: Value tensor (total_k, num_heads, head_dim)
        cu_seqlens_q: Cumulative sequence lengths for queries
        cu_seqlens_k: Cumulative sequence lengths for keys
        max_seqlen_q: Maximum sequence length in queries
        max_seqlen_k: Maximum sequence length in keys
        smooth_k: Whether to apply key smoothing
        pv_accum_dtype: Accumulation dtype for P*V
        attn_backend: Backend to use

    Returns:
        Output tensor
    """

    # For now, use Triton implementation which should handle varlen
    if smooth_k:
        k = smooth_keys(k)

    return triton_kernels.attn_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        pv_accum_dtype=pv_accum_dtype
    )


def smooth_keys(k: torch.Tensor) -> torch.Tensor:
    """
    Apply smoothing to keys to improve quantization.

    Args:
        k: Key tensor

    Returns:
        Smoothed key tensor
    """
    # Simple smoothing: subtract mean from each head
    k_mean = k.mean(dim=-1, keepdim=True)
    return k - k_mean


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    pv_accum_dtype: str = "fp32"
) -> torch.Tensor:
    """
    Triton implementation of INT8 QK with FP16 PV attention.
    """
    return triton_kernels.qk_int8_pv_fp16_attn(
        q, k, v,
        is_causal=is_causal,
        pv_accum_dtype=pv_accum_dtype
    )


def sageattn_qk_int8_pv_fp16_hip(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    pv_accum_dtype: str = "fp32"
) -> torch.Tensor:
    """
    HIP implementation of INT8 QK with FP16 PV attention.
    """
    if not HAS_HIP_KERNELS:
        warnings.warn("HIP kernels not available, falling back to Triton")
        return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal, pv_accum_dtype)

    # Call HIP kernel
    # This would call the actual HIP implementation
    # For now, return a placeholder
    raise NotImplementedError("HIP kernel implementation pending")


def sageattn_qk_int8_pv_fp8_hip(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False
) -> torch.Tensor:
    """
    HIP implementation of INT8 QK with FP8 PV attention (MI300 only).
    """
    if not HAS_HIP_KERNELS:
        raise RuntimeError("FP8 attention requires HIP kernels which are not available")

    device_cap = get_device_capability()
    if device_cap not in ['gfx940', 'gfx941']:
        raise RuntimeError(f"FP8 attention requires MI300 (gfx940/941), got {device_cap}")

    # Call HIP FP8 kernel
    raise NotImplementedError("HIP FP8 kernel implementation pending")