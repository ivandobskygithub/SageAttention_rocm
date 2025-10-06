"""
Fallback implementations when Triton is not available
Uses PyTorch for basic functionality
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def quantize(tensor: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback quantization using PyTorch.

    Args:
        tensor: Input tensor to quantize
        block_size: Size of quantization blocks

    Returns:
        Quantized INT8 tensor and scale factors
    """
    # Reshape for block quantization
    orig_shape = tensor.shape
    tensor = tensor.view(-1, block_size)

    # Find scale per block
    scale = tensor.abs().max(dim=1, keepdim=True)[0] / 127.0
    scale = scale.clamp(min=1e-6)

    # Quantize
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)

    # Reshape back
    quantized = quantized.view(orig_shape)
    scale = scale.view(-1)

    return quantized, scale


def dequantize(quantized: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Fallback dequantization using PyTorch.
    """
    orig_shape = quantized.shape
    quantized = quantized.view(-1, block_size)
    scale = scale.view(-1, 1)

    dequantized = quantized.to(torch.float32) * scale
    return dequantized.view(orig_shape)


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
    Fallback attention implementation using PyTorch.
    This is much slower than Triton but provides correct results for testing.
    """
    # Dequantize Q and K
    q = dequantize(q_int8, q_scale, block_size=128).to(v.dtype)
    k = dequantize(k_int8, k_scale, block_size=128).to(v.dtype)

    # Standard attention computation
    # Q, K, V are in [batch*heads, seq_len, head_dim] format
    batch_heads, seq_len, head_dim = q.shape

    # Compute attention scores
    scores = torch.bmm(q, k.transpose(1, 2)) * sm_scale

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply to values
    if use_fp32_accum:
        attn_weights = attn_weights.to(torch.float32)
        v = v.to(torch.float32)
        output = torch.bmm(attn_weights, v)
        output = output.to(q.dtype)
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
        # Same as non-causal but with causal mask
        q = dequantize(q_int8, q_scale, block_size=128).to(v.dtype)
        k = dequantize(k_int8, k_scale, block_size=128).to(v.dtype)

        batch_heads, seq_len, head_dim = q.shape

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
            v = v.to(torch.float32)
            output = torch.bmm(attn_weights, v)
            output = output.to(q.dtype)
        else:
            output = torch.bmm(attn_weights, v)

        return output


class attn_qk_int8_block_varlen:
    """Fallback for variable length attention."""

    @staticmethod
    def forward(q_int8, k_int8, v, q_scale, k_scale, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, sm_scale, use_fp32_accum=True):
        # Simplified implementation - just use regular attention
        # In production, this would handle variable lengths properly
        return forward(q_int8, k_int8, v, q_scale, k_scale, sm_scale, use_fp32_accum)


class attn_qk_int8_per_block_causal_varlen:
    """Fallback for variable length causal attention."""

    @staticmethod
    def forward(q_int8, k_int8, v, q_scale, k_scale, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, sm_scale, use_fp32_accum=True):
        # Use causal attention fallback
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
        # Simplified - just use regular quantization
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