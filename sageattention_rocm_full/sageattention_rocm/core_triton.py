"""
Triton-only core attention implementations for ROCm 7
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings
import os

# Force ROCm backend for Triton
os.environ['TRITON_HIP_USE_ROCM'] = '1'

# Import Triton kernels
from .triton import (
    attn_qk_int8_per_block,
    attn_qk_int8_per_block_causal,
    attn_qk_int8_per_block_causal_varlen,
    attn_qk_int8_block_varlen,
    quant_per_block,
    quant_per_block_varlen,
    quant_per_thread,
)


def get_rocm_device_info():
    """Get ROCm device information."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        device_name = props.name

        # Get compute capability info
        info = {
            'name': device_name,
            'total_memory': props.total_memory,
            'multiprocessor_count': props.multi_processor_count,
        }

        # Determine architecture
        device_name_lower = device_name.lower()
        if 'mi300' in device_name_lower:
            info['arch'] = 'gfx940'
            info['has_fp8'] = True
            info['has_mfma'] = True
        elif 'mi250' in device_name_lower or 'mi200' in device_name_lower:
            info['arch'] = 'gfx90a'
            info['has_fp8'] = False
            info['has_mfma'] = True
        elif 'mi100' in device_name_lower:
            info['arch'] = 'gfx908'
            info['has_fp8'] = False
            info['has_mfma'] = True
        elif '7900' in device_name_lower:
            info['arch'] = 'gfx1100'
            info['has_fp8'] = False
            info['has_mfma'] = False
        else:
            info['arch'] = 'unknown'
            info['has_fp8'] = False
            info['has_mfma'] = False

        return info
    return None


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
    Main SageAttention function using Triton kernels on ROCm.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim] or [num_heads, seq_len, head_dim]
        k: Key tensor
        v: Value tensor
        tensor_layout: Layout of input tensors ("HND" or "NHD")
        is_causal: Whether to apply causal masking
        smooth_k: Whether to apply key smoothing
        pv_accum_dtype: Accumulation dtype for P*V ("fp32" or "fp16")

    Returns:
        Output tensor with same shape as q
    """

    # Input validation
    assert tensor_layout in ["HND", "NHD"], f"Invalid tensor_layout: {tensor_layout}"
    assert pv_accum_dtype in ["fp32", "fp16"], f"Invalid pv_accum_dtype: {pv_accum_dtype}"

    # Check device
    assert q.is_cuda, "Input tensors must be on CUDA device"
    assert k.is_cuda and v.is_cuda, "All tensors must be on same device"

    # Get device info
    device_info = get_rocm_device_info()
    if device_info:
        print(f"Running on {device_info['name']} ({device_info.get('arch', 'unknown')})")

    # Handle tensor layout
    original_shape = None
    if tensor_layout == "NHD":
        # N=batch, H=heads, D=seq_len, head_dim
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
    else:  # HND
        # H=heads, N=seq_len, D=head_dim
        num_heads = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_kv = k.shape[1]
        head_dim = q.shape[2]

    # Validate dimensions
    assert q.shape[0] == k.shape[0] == v.shape[0], "Batch/head dimension mismatch"
    assert k.shape[1] == v.shape[1], "KV sequence length mismatch"
    assert q.shape[2] == k.shape[2] == v.shape[2], "Head dimension mismatch"
    assert head_dim % 16 == 0, f"Head dimension must be divisible by 16, got {head_dim}"

    # Apply key smoothing if requested
    if smooth_k:
        k = smooth_keys(k)

    # Call Triton implementation
    output = sageattn_qk_int8_pv_fp16_triton(
        q, k, v,
        is_causal=is_causal,
        pv_accum_dtype=pv_accum_dtype
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
    """
    Variable-length SageAttention using Triton on ROCm.

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

    Returns:
        Output tensor
    """

    # Validation
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA"
    assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda, "Sequence lengths must be on CUDA"

    # Apply smoothing if requested
    if smooth_k:
        k = smooth_keys(k)

    # Use variable-length Triton kernel
    return sageattn_qk_int8_pv_fp16_triton_varlen(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        pv_accum_dtype=pv_accum_dtype
    )


def smooth_keys(k: torch.Tensor) -> torch.Tensor:
    """
    Apply smoothing to keys to improve quantization accuracy.

    Args:
        k: Key tensor [batch*heads, seq_len, head_dim]

    Returns:
        Smoothed key tensor
    """
    # Compute per-head mean and subtract
    k_mean = k.mean(dim=-1, keepdim=True)
    k_smooth = k - k_mean
    return k_smooth


def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    pv_accum_dtype: str = "fp32"
) -> torch.Tensor:
    """
    Triton implementation of INT8 QK with FP16 PV attention.

    This function:
    1. Quantizes Q and K to INT8
    2. Computes attention scores with INT8
    3. Applies softmax
    4. Computes output with FP16 values
    """

    batch_heads, seq_len_q, head_dim = q.shape
    seq_len_kv = k.shape[1]

    # Scale factor for softmax
    sm_scale = head_dim ** -0.5

    # Quantize Q and K to INT8
    # Using per-block quantization for better accuracy
    q_int8, q_scale = quantize_tensor(q, granularity='per_block')
    k_int8, k_scale = quantize_tensor(k, granularity='per_block')

    # Call the appropriate Triton kernel based on causal mask
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

    return output


def sageattn_qk_int8_pv_fp16_triton_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    pv_accum_dtype: str = "fp32"
) -> torch.Tensor:
    """
    Variable-length Triton implementation.
    """

    # Get dimensions
    total_q = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    sm_scale = head_dim ** -0.5

    # Quantize with variable-length support
    q_int8, q_scale = quantize_tensor_varlen(
        q, cu_seqlens_q, max_seqlen_q
    )
    k_int8, k_scale = quantize_tensor_varlen(
        k, cu_seqlens_k, max_seqlen_k
    )

    # Call variable-length kernel
    output = attn_qk_int8_block_varlen.forward(
        q_int8, k_int8, v,
        q_scale, k_scale,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        sm_scale,
        pv_accum_dtype == "fp32"
    )

    return output


def quantize_tensor(
    tensor: torch.Tensor,
    granularity: str = 'per_block',
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8 with specified granularity.

    Args:
        tensor: Input tensor [batch*heads, seq_len, head_dim]
        granularity: 'per_block', 'per_row', or 'per_tensor'
        block_size: Size of quantization blocks

    Returns:
        Quantized INT8 tensor and scales
    """

    if granularity == 'per_block':
        # Use Triton kernel for block quantization
        return quant_per_block.quantize(tensor, block_size)
    elif granularity == 'per_row':
        # Quantize per sequence position
        scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 127.0
        scale = scale.clamp(min=1e-6)
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale
    else:  # per_tensor
        scale = tensor.abs().max() / 127.0
        scale = scale.clamp(min=1e-6)
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale


def quantize_tensor_varlen(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize variable-length tensor to INT8.
    """
    return quant_per_block_varlen.quantize(
        tensor, cu_seqlens, max_seqlen, block_size
    )