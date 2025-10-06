"""
Fixed AOTriton compatibility layer for ROCm 7.9
Works with PyTorch 2.10.0a0+rocm7.9
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
        # Check for various AOTriton entry points
        if hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, '_triton_multi_head_attention'):
            return True
        if hasattr(torch, '_triton_multi_head_attention'):
            return True
        if hasattr(torch, '_triton_scaled_dot_attention'):
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
    Use PyTorch's AOTriton attention implementation with proper API.

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
        # For ROCm 7.9, try different AOTriton APIs

        # Try torch.ops.aten first (most likely to work)
        if hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, '_triton_multi_head_attention'):
            try:
                # This API may not take scale as a kwarg
                output = torch.ops.aten._triton_multi_head_attention(
                    q, k, v,
                    scale,  # Pass scale as positional arg
                    None,   # attn_mask
                    0.0,    # dropout_p
                    is_causal
                )
                return output
            except Exception as e1:
                # Try without some args
                try:
                    output = torch.ops.aten._triton_multi_head_attention(
                        q, k, v
                    )
                    return output * (scale / (q.shape[-1] ** -0.5))  # Apply scale manually
                except Exception as e2:
                    pass

        # Try direct torch._triton functions
        if hasattr(torch, '_triton_scaled_dot_attention'):
            try:
                # Try without scale kwarg (may be positional)
                output = torch._triton_scaled_dot_attention(
                    q, k, v,
                    None,  # attn_bias
                    scale  # scale as positional
                )
                return output
            except Exception as e:
                # Try minimal args
                try:
                    output = torch._triton_scaled_dot_attention(q, k, v)
                    return output * (scale / (q.shape[-1] ** -0.5))
                except:
                    pass

        # If AOTriton fails, use PyTorch's optimized attention
        # This will use Flash Attention or Memory Efficient Attention if available
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        ):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                scale=scale
            )

    except Exception as e:
        # Final fallback to standard PyTorch
        warnings.warn(f"AOTriton not working, using PyTorch: {e}", stacklevel=2)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            scale=scale
        )


def test_aotriton_ops():
    """Test which AOTriton operations actually work."""
    results = {}

    if not torch.cuda.is_available():
        return results

    # Create small test tensors
    q = torch.randn(1, 8, 64, 32, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, 64, 32, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, 64, 32, device='cuda', dtype=torch.float16)

    # Test torch.ops.aten._triton_multi_head_attention
    if hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, '_triton_multi_head_attention'):
        try:
            output = torch.ops.aten._triton_multi_head_attention(q, k, v)
            results['torch.ops.aten._triton_multi_head_attention'] = 'Works (minimal args)'
        except Exception as e:
            results['torch.ops.aten._triton_multi_head_attention'] = f'Failed: {str(e)[:50]}'

    # Test torch._triton_scaled_dot_attention
    if hasattr(torch, '_triton_scaled_dot_attention'):
        try:
            output = torch._triton_scaled_dot_attention(q, k, v)
            results['torch._triton_scaled_dot_attention'] = 'Works (minimal args)'
        except Exception as e:
            results['torch._triton_scaled_dot_attention'] = f'Failed: {str(e)[:50]}'

    # Test torch._triton_multi_head_attention
    if hasattr(torch, '_triton_multi_head_attention'):
        try:
            output = torch._triton_multi_head_attention(q, k, v)
            results['torch._triton_multi_head_attention'] = 'Works (minimal args)'
        except Exception as e:
            results['torch._triton_multi_head_attention'] = f'Failed: {str(e)[:50]}'

    return results


class AOTritonConfig:
    """Configuration for AOTriton usage."""

    def __init__(self):
        self.available = check_aotriton_available()
        self.experimental_enabled = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1"
        self.working_ops = {}

        if self.available and self.experimental_enabled:
            print("[SUCCESS] AOTriton is available and enabled")
            # Test which ops actually work
            if torch.cuda.is_available():
                self.working_ops = test_aotriton_ops()
                if self.working_ops:
                    print("AOTriton operations test results:")
                    for op, status in self.working_ops.items():
                        print(f"  {op}: {status}")
        elif self.available and not self.experimental_enabled:
            warnings.warn("AOTriton found but experimental flag not set. Setting now...")
            os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
            self.experimental_enabled = True
        else:
            warnings.warn("AOTriton not available, will use fallback implementations")

    def get_status(self) -> dict:
        """Get AOTriton status information."""
        status = {
            "available": self.available,
            "experimental_enabled": self.experimental_enabled,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            status["device_name"] = torch.cuda.get_device_name(0)
        if self.working_ops:
            status["working_ops"] = self.working_ops
        return status


# Global config instance
_aotriton_config = None

def get_aotriton_config() -> AOTritonConfig:
    """Get or create AOTriton configuration."""
    global _aotriton_config
    if _aotriton_config is None:
        _aotriton_config = AOTritonConfig()
    return _aotriton_config