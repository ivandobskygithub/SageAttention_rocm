"""
AOTriton compatibility layer for Windows ROCm
Uses PyTorch's built-in AOTriton when available
"""

import os
import torch
import warnings
from typing import Optional, Tuple

# Enable experimental AOTriton support
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

def check_aotriton_available() -> bool:
    """Check if AOTriton is available in PyTorch."""
    try:
        # Check for AOTriton ops in torch
        if hasattr(torch, '_triton_multi_head_attention'):
            # Try to access it to ensure it's actually available
            _ = torch._triton_multi_head_attention
            return True

        # Alternative check for scaled dot attention
        if hasattr(torch, '_triton_scaled_dot_attention'):
            _ = torch._triton_scaled_dot_attention
            return True

        # Check for ops module variants
        if hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, '_triton_multi_head_attention'):
            return True

        return False
    except Exception as e:
        warnings.warn(f"AOTriton check failed: {e}")
        return False


def aotriton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Use PyTorch's AOTriton attention implementation.

    Args:
        q, k, v: Query, key, value tensors [batch, heads, seq_len, head_dim]
        is_causal: Whether to use causal masking
        scale: Attention scale factor

    Returns:
        Attention output tensor
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    try:
        # Try different AOTriton entry points
        if hasattr(torch, '_triton_scaled_dot_attention'):
            # Use scaled dot product attention
            return torch._triton_scaled_dot_attention(
                q, k, v,
                scale=scale,
                is_causal=is_causal
            )
        elif hasattr(torch.nn.functional, '_scaled_dot_product_attention_math'):
            # Fallback to math implementation
            return torch.nn.functional._scaled_dot_product_attention_math(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=scale
            )
        else:
            # Use standard PyTorch
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                scale=scale
            )
    except Exception as e:
        warnings.warn(f"AOTriton attention failed: {e}, using PyTorch fallback")
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            scale=scale
        )


class AOTritonConfig:
    """Configuration for AOTriton usage."""

    def __init__(self):
        self.available = check_aotriton_available()
        self.experimental_enabled = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1"

        if self.available and self.experimental_enabled:
            print("[SUCCESS] AOTriton is available and enabled")
        elif self.available and not self.experimental_enabled:
            warnings.warn("AOTriton found but experimental flag not set. Setting now...")
            os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
            self.experimental_enabled = True
        else:
            warnings.warn("AOTriton not available, will use fallback implementations")

    def get_status(self) -> dict:
        """Get AOTriton status information."""
        return {
            "available": self.available,
            "experimental_enabled": self.experimental_enabled,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        }


# Global config instance
_aotriton_config = None

def get_aotriton_config() -> AOTritonConfig:
    """Get or create AOTriton configuration."""
    global _aotriton_config
    if _aotriton_config is None:
        _aotriton_config = AOTritonConfig()
    return _aotriton_config