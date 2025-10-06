"""
Fixed fallback implementations when Triton is not available
Uses PyTorch for basic functionality
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def quantize(tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed quantization that properly handles block sizes.

    Args:
        tensor: Input tensor to quantize [batch*heads, seq_len, head_dim]
        block_size: Size of quantization blocks

    Returns:
        Quantized INT8 tensor and scale factors
    """
    orig_shape = tensor.shape

    # Handle the quantization along the sequence dimension
    batch_heads, seq_len, head_dim = orig_shape

    # We quantize blocks along the sequence dimension
    # Each block gets its own scale
    num_blocks = (seq_len + block_size - 1) // block_size

    scales = []
    quantized_blocks = []

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, seq_len)

        # Extract block
        block = tensor[:, start_idx:end_idx, :]

        # Find scale for this block (max abs value)
        block_scale = block.abs().max() / 127.0
        block_scale = block_scale.clamp(min=1e-6)
        scales.append(block_scale)

        # Quantize block
        block_quant = (block / block_scale).round().clamp(-128, 127).to(torch.int8)
        quantized_blocks.append(block_quant)

    # Concatenate blocks
    quantized = torch.cat(quantized_blocks, dim=1)
    scale = torch.tensor(scales, dtype=torch.float32, device=tensor.device)

    return quantized, scale


def dequantize(quantized: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Fixed dequantization that properly handles block sizes.
    """
    batch_heads, seq_len, head_dim = quantized.shape
    num_blocks = scale.shape[0]

    dequantized_blocks = []

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, seq_len)

        # Extract block
        block = quantized[:, start_idx:end_idx, :]

        # Dequantize with corresponding scale
        block_deq = block.to(torch.float32) * scale[i]
        dequantized_blocks.append(block_deq)

    # Concatenate blocks
    dequantized = torch.cat(dequantized_blocks, dim=1)

    return dequantized


def forward(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    sm_scale: float,
    use_fp32_accum: bool = True
) -> torch.Tensor:
    """
    Fixed attention implementation using PyTorch.
    """
    # Dequantize Q and K with proper block size
    # Infer block size from scale tensor
    batch_heads, seq_len, head_dim = q_int8.shape
    block_size = max(1, seq_len // max(1, q_scale.shape[0]))

    q = dequantize(q_int8, q_scale, block_size=block_size).to(v.dtype)
    k = dequantize(k_int8, k_scale, block_size=block_size).to(v.dtype)

    # Compute attention scores
    scores = torch.bmm(q, k.transpose(1, 2)) * sm_scale

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply to values
    if use_fp32_accum:
        attn_weights = attn_weights.to(torch.float32)
        v_float = v.to(torch.float32)
        output = torch.bmm(attn_weights, v_float)
        output = output.to(v.dtype)
    else:
        output = torch.bmm(attn_weights, v)

    return output


class attn_qk_int8_per_block:
    """Fallback for per-block attention."""

    @staticmethod
    def forward(q_int8, k_int8, v, q_scale, k_scale, sm_scale, use_fp32_accum=True):
        return forward(q_int8, k_int8, v, q_scale, k_scale, sm_scale, use_fp32_accum)


class attn_qk_int8_per_block_causal:
    """Fallback for causal attention."""

    @staticmethod
    def forward(q_int8, k_int8, v, q_scale, k_scale, sm_scale, use_fp32_accum=True):
        # Infer block size
        batch_heads, seq_len, head_dim = q_int8.shape
        block_size = max(1, seq_len // max(1, q_scale.shape[0]))

        # Dequantize
        q = dequantize(q_int8, q_scale, block_size=block_size).to(v.dtype)
        k = dequantize(k_int8, k_scale, block_size=block_size).to(v.dtype)

        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * sm_scale

        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
        mask = mask.unsqueeze(0).expand(batch_heads, -1, -1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply to values
        if use_fp32_accum:
            attn_weights = attn_weights.to(torch.float32)
            v_float = v.to(torch.float32)
            output = torch.bmm(attn_weights, v_float)
            output = output.to(v.dtype)
        else:
            output = torch.bmm(attn_weights, v)

        return output


class attn_qk_int8_block_varlen:
    """Fallback for variable length attention."""

    @staticmethod
    def forward(q_int8, k_int8, v, q_scale, k_scale, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, sm_scale, use_fp32_accum=True):
        # Simplified - use regular attention
        return forward(q_int8, k_int8, v, q_scale, k_scale, sm_scale, use_fp32_accum)


class attn_qk_int8_per_block_causal_varlen:
    """Fallback for variable length causal attention."""

    @staticmethod
    def forward(q_int8, k_int8, v, q_scale, k_scale, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, sm_scale, use_fp32_accum=True):
        return attn_qk_int8_per_block_causal.forward(
            q_int8, k_int8, v, q_scale, k_scale, sm_scale, use_fp32_accum
        )


class quant_per_block:
    """Fallback for per-block quantization."""

    @staticmethod
    def quantize(tensor, block_size=128):
        return quantize(tensor, block_size)


class quant_per_block_varlen:
    """Fallback for variable length quantization."""

    @staticmethod
    def quantize(tensor, cu_seqlens, max_seqlen, block_size=128):
        # Simplified - use regular quantization
        return quantize(tensor, block_size)


class quant_per_thread:
    """Fallback for per-thread quantization."""

    @staticmethod
    def quantize(tensor):
        # Use finer granularity
        return quantize(tensor, block_size=32)


# Print warning when using fallbacks
import warnings
warnings.warn(
    "Using PyTorch fallback implementations. "
    "This is much slower than Triton kernels! "
    "Install Triton for GPU acceleration."
)