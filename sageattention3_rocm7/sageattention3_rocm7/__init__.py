"""
SageAttention3 ROCm7 - High-performance attention with INT4 quantization for AMD GPUs

This package provides a ROCm7/HIP implementation of SageAttention3, offering:
- 4x memory compression through INT4 quantization
- Optimized attention computation for AMD RDNA3+ GPUs
- PyTorch integration with minimal code changes
"""

__version__ = "0.1.0"

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Import the core implementation
from .torch_integration import (
    SageAttentionROCm,
    attention_forward,
    quantize_int4,
    dequantize_int4,
)

# Global instance
_sage_instance: Optional[SageAttentionROCm] = None


def _get_instance() -> SageAttentionROCm:
    """Get or create the global SageAttention instance."""
    global _sage_instance
    if _sage_instance is None:
        # Auto-detect DLL path
        dll_path = Path(__file__).parent / "sage_attention_rocm7.dll"
        if not dll_path.exists():
            # Try parent directory
            dll_path = Path(__file__).parent.parent / "sage_attention_rocm7.dll"

        _sage_instance = SageAttentionROCm(dll_path=str(dll_path) if dll_path.exists() else None)
    return _sage_instance


def preprocess_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   per_block_mean: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess QKV tensors for SageAttention (compatibility layer).

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        per_block_mean: Whether to use per-block mean subtraction

    Returns:
        Tuple of (q_processed, k_processed, v_processed, delta_s)
    """
    def pad_128(x):
        """Pad sequence length to multiple of 128."""
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    # Center K tensor
    k = k - k.mean(dim=-2, keepdim=True)

    # Pad to 128 boundaries
    q, k, v = map(lambda x: pad_128(x), [q, k, v])

    # Compute per-block or global mean for Q
    if per_block_mean:
        # Simplified per-block mean (without Triton kernel)
        B, H, L, D = q.shape
        GROUP_SIZE = 128
        num_groups = L // GROUP_SIZE

        # Reshape for group processing
        q_reshaped = q.view(B, H, num_groups, GROUP_SIZE, D)
        qm = q_reshaped.mean(dim=3, keepdim=False)  # [B, H, num_groups, D]
        q = q_reshaped - qm.unsqueeze(3)
        q = q.view(B, H, L, D)

        # Expand qm for delta_s computation
        qm_expanded = qm.unsqueeze(3).expand(-1, -1, -1, GROUP_SIZE, -1).reshape(B, H, L, D)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
        qm_expanded = qm.expand_as(q)

    # Compute delta_s for attention correction
    delta_s = torch.matmul(qm_expanded[:, :, ::128, :], k.transpose(-2, -1)).to(torch.float32).contiguous()

    return q, k, v, delta_s


def scale_and_quant_int4(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT4 with block scaling (Blackwell API compatibility).

    Args:
        x: Input tensor [batch, heads, seq_len, head_dim]

    Returns:
        Tuple of (quantized_data, scales)
    """
    sage = _get_instance()

    # Reshape to match expected format [batch, seq_len, heads, head_dim]
    B, H, N, D = x.shape
    x_reshaped = x.transpose(1, 2).contiguous()  # [B, N, H, D]

    # Quantize
    quantized, scales = sage.quantize_int4(x_reshaped)

    # Reshape back to [B, H, N, D//2] format expected by Blackwell API
    quantized = quantized.transpose(1, 2).contiguous()
    scales = scales.transpose(1, 2).contiguous()

    return quantized, scales


def scale_and_quant_int4_permute(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize with permutation for K tensor (Blackwell API compatibility).
    Currently uses standard quantization - permutation optimization TODO.
    """
    return scale_and_quant_int4(x)


def scale_and_quant_int4_transpose(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize with transpose for V tensor (Blackwell API compatibility).
    """
    sage = _get_instance()
    B, H, N, D = x.shape

    # For V tensor, we need transposed output [B, H, D, N//2]
    # First reshape to expected input format
    x_reshaped = x.transpose(1, 2).contiguous()  # [B, N, H, D]

    # Use transpose quantization if available, otherwise standard + transpose
    # TODO: Implement optimized transpose quantization in DLL
    quantized, scales = sage.quantize_int4(x_reshaped)

    # Reshape and transpose for V tensor format
    quantized = quantized.transpose(1, 2)  # [B, H, N, D//2]
    quantized = quantized.transpose(2, 3).contiguous()  # [B, H, D//2, N]

    scales = scales.transpose(1, 2)  # [B, H, N, scale_dim]
    scales = scales.transpose(2, 3).contiguous()  # [B, H, scale_dim, N]

    return quantized, scales


def sageattn3_rocm7(q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    attn_mask: Optional[torch.Tensor] = None,
                    is_causal: bool = False,
                    per_block_mean: bool = True,
                    **kwargs) -> torch.Tensor:
    """
    Main SageAttention3 ROCm7 API - compatible with Blackwell interface.

    Args:
        q: Query tensor [batch, heads, seq_len_q, head_dim]
        k: Key tensor [batch, heads, seq_len_k, head_dim]
        v: Value tensor [batch, heads, seq_len_k, head_dim]
        attn_mask: Optional attention mask (not yet supported)
        is_causal: Whether to apply causal masking
        per_block_mean: Whether to use per-block mean subtraction
        **kwargs: Additional arguments for compatibility

    Returns:
        Attention output [batch, heads, seq_len_q, head_dim]
    """
    # Check head dimension constraint
    if q.size(-1) >= 256:
        warnings.warn(f"Head dimension {q.size(-1)} >= 256, falling back to PyTorch SDPA")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

    # Check if mask is provided (not yet supported)
    if attn_mask is not None:
        warnings.warn("Attention mask not yet supported, ignoring")

    # Store original sequence length
    QL = q.size(2)
    KL = k.size(2)

    # Convert to FP16 if needed (INT4 quantization works with FP16)
    dtype_orig = q.dtype
    if q.dtype == torch.bfloat16:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    elif q.dtype != torch.float16:
        warnings.warn(f"Converting from {q.dtype} to float16 for processing")
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

    # Preprocess QKV
    q_proc, k_proc, v_proc, delta_s = preprocess_qkv(q, k, v, per_block_mean)

    # Get SageAttention instance
    sage = _get_instance()

    # Compute attention with preprocessing
    output = sage.attention_forward(
        Q=q_proc,
        K=k_proc,
        V=v_proc,
        scale=None,  # Will use default 1/sqrt(d)
        is_causal=is_causal,
        delta_s=delta_s if per_block_mean else None
    )

    # Trim to original sequence length
    output = output[:, :, :QL, :].contiguous()

    # Convert back to original dtype if needed
    if dtype_orig == torch.bfloat16:
        output = output.to(torch.bfloat16)
    elif dtype_orig != torch.float16:
        output = output.to(dtype_orig)

    return output


# Alternative API name for compatibility
sageattn3_blackwell = sageattn3_rocm7


# Export main APIs
__all__ = [
    "sageattn3_rocm7",
    "sageattn3_blackwell",  # Alias for compatibility
    "SageAttentionROCm",
    "preprocess_qkv",
    "scale_and_quant_int4",
    "scale_and_quant_int4_permute",
    "scale_and_quant_int4_transpose",
    "attention_forward",
    "quantize_int4",
    "dequantize_int4",
    "__version__",
]