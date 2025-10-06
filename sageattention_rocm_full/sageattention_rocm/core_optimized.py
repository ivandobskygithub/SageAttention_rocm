"""
Optimized SageAttention for ROCm using PyTorch's built-in kernels
Works without Triton/AOTriton by leveraging Flash Attention if available
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


def quantize_int8(tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8 with block-wise scaling.
    Optimized for GPU execution.
    """
    batch_heads, seq_len, head_dim = tensor.shape

    # Reshape for block quantization
    num_blocks = (seq_len + block_size - 1) // block_size

    # Pad if necessary
    if seq_len % block_size != 0:
        pad_len = num_blocks * block_size - seq_len
        tensor = F.pad(tensor, (0, 0, 0, pad_len))

    # Reshape to blocks
    tensor_blocks = tensor.view(batch_heads, num_blocks, block_size, head_dim)

    # Compute scales per block
    scales = tensor_blocks.abs().amax(dim=(2, 3), keepdim=True) / 127.0
    scales = scales.clamp(min=1e-6)

    # Quantize
    quantized = (tensor_blocks / scales).round().clamp(-128, 127).to(torch.int8)

    # Reshape back
    quantized = quantized.view(batch_heads, num_blocks * block_size, head_dim)

    # Remove padding
    if seq_len % block_size != 0:
        quantized = quantized[:, :seq_len, :]

    scales = scales.squeeze(-1).squeeze(-1)  # [batch_heads, num_blocks]

    return quantized, scales


def dequantize_int8(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128
) -> torch.Tensor:
    """
    Dequantize INT8 tensor with block-wise scaling.
    Optimized for GPU execution.
    """
    batch_heads, seq_len, head_dim = quantized.shape
    num_blocks = scales.shape[1]

    # Pad if necessary
    if seq_len % block_size != 0:
        pad_len = num_blocks * block_size - seq_len
        quantized = F.pad(quantized, (0, 0, 0, pad_len))

    # Reshape to blocks
    quantized_blocks = quantized.view(batch_heads, num_blocks, block_size, head_dim)
    scales = scales.view(batch_heads, num_blocks, 1, 1)

    # Dequantize
    dequantized = quantized_blocks.float() * scales

    # Reshape back
    dequantized = dequantized.view(batch_heads, num_blocks * block_size, head_dim)

    # Remove padding
    if seq_len % block_size != 0:
        dequantized = dequantized[:, :seq_len, :]

    return dequantized


def smooth_keys(k: torch.Tensor) -> torch.Tensor:
    """Apply key smoothing for better quantization."""
    k_mean = k.mean(dim=-1, keepdim=True)
    return k - k_mean


def sageattn_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    smooth_k: bool = True,
    use_int8: bool = True,
    block_size: int = 128,
) -> torch.Tensor:
    """
    Optimized SageAttention for ROCm.
    Uses PyTorch's optimized kernels (Flash Attention if available).

    Args:
        q, k, v: Query, Key, Value tensors
        tensor_layout: "HND" or "NHD"
        is_causal: Whether to apply causal masking
        smooth_k: Whether to smooth keys
        use_int8: Whether to use INT8 quantization
        block_size: Block size for quantization

    Returns:
        Attention output
    """

    # Handle tensor layout
    original_shape = None
    if tensor_layout == "NHD":
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        seq_len_q = q.shape[2]
        seq_len_kv = k.shape[2]
        head_dim = q.shape[3]
        original_shape = q.shape

        # Convert to [batch*heads, seq_len, head_dim] for processing
        q = q.view(batch_size * num_heads, seq_len_q, head_dim)
        k = k.view(batch_size * num_heads, seq_len_kv, head_dim)
        v = v.view(batch_size * num_heads, seq_len_kv, head_dim)

    # Apply key smoothing if requested
    if smooth_k:
        k = smooth_keys(k)

    if use_int8:
        # Quantize Q and K to INT8
        q_int8, q_scale = quantize_int8(q, block_size)
        k_int8, k_scale = quantize_int8(k, block_size)

        # Dequantize for attention computation
        q_fp = dequantize_int8(q_int8, q_scale, block_size).to(v.dtype)
        k_fp = dequantize_int8(k_int8, k_scale, block_size).to(v.dtype)
    else:
        q_fp = q
        k_fp = k

    # Reshape for batch processing
    # PyTorch's scaled_dot_product_attention expects [batch, heads, seq, dim]
    batch_heads = q_fp.shape[0]
    seq_len_q = q_fp.shape[1]
    seq_len_kv = k_fp.shape[1]
    head_dim = q_fp.shape[2]

    # For simplicity, treat batch*heads as batch dimension with 1 head
    q_fp = q_fp.unsqueeze(1)  # [batch*heads, 1, seq_q, dim]
    k_fp = k_fp.unsqueeze(1)  # [batch*heads, 1, seq_kv, dim]
    v = v.unsqueeze(1)        # [batch*heads, 1, seq_kv, dim]

    # Use PyTorch's optimized attention
    # This will automatically use Flash Attention, Memory Efficient Attention,
    # or Math backend depending on what's available
    output = F.scaled_dot_product_attention(
        q_fp, k_fp, v,
        is_causal=is_causal,
        scale=head_dim ** -0.5
    )

    # Reshape back
    output = output.squeeze(1)  # [batch*heads, seq_q, dim]

    # Restore original shape if needed
    if tensor_layout == "NHD" and original_shape is not None:
        output = output.view(original_shape)

    return output


def get_rocm_device_info():
    """Get device information with backend status."""
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    device_name = props.name

    info = {
        'name': device_name,
        'total_memory': props.total_memory,
        'multiprocessor_count': props.multi_processor_count,
        'compute_capability': f"{props.major}.{props.minor}",
    }

    # Test available backends
    try:
        q = torch.randn(1, 1, 64, 32, device='cuda', dtype=torch.float16)
        k = torch.randn(1, 1, 64, 32, device='cuda', dtype=torch.float16)
        v = torch.randn(1, 1, 64, 32, device='cuda', dtype=torch.float16)

        # Test Flash Attention
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False
            ):
                _ = F.scaled_dot_product_attention(q, k, v)
            info['flash_attention'] = True
        except:
            info['flash_attention'] = False

        # Test Memory Efficient Attention
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=False,
                enable_mem_efficient=True
            ):
                _ = F.scaled_dot_product_attention(q, k, v)
            info['mem_efficient_attention'] = True
        except:
            info['mem_efficient_attention'] = False
    except:
        pass

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


# Alias for compatibility
sageattn = sageattn_optimized


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    smooth_k: bool = True,
    use_int8: bool = True,
) -> torch.Tensor:
    """Variable-length attention (simplified version)."""

    if smooth_k:
        k = smooth_keys(k)

    # For simplicity, use regular attention
    return sageattn_optimized(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        tensor_layout="NHD",
        is_causal=False,
        smooth_k=False,  # Already smoothed
        use_int8=use_int8
    ).squeeze(0)