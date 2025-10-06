"""
SageAttention ROCm Implementation - Works with AOTriton, Triton, or PyTorch fallback
"""

import warnings
import os

# Enable AOTriton experimental support
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Try to use AOTriton-aware implementation first
AOTRITON_AVAILABLE = False
try:
    from .core_triton_aotriton import (
        sageattn,
        sageattn_varlen,
        get_rocm_device_info,
        AOTRITON_AVAILABLE,
    )
    if AOTRITON_AVAILABLE:
        IMPLEMENTATION = "AOTriton"
    else:
        IMPLEMENTATION = "Triton/Fallback"
except ImportError as e:
    # Try simple implementation
    try:
        from .core_triton_simple import (
            sageattn,
            sageattn_varlen,
            get_rocm_device_info,
        )
        IMPLEMENTATION = "Simple"
    except ImportError:
        # Final fallback to original
        warnings.warn(f"Failed to import optimized kernels: {e}\nUsing fallback implementation.")
        from .core_triton import (
            sageattn,
            sageattn_varlen,
            get_rocm_device_info,
        )
        IMPLEMENTATION = "Fallback"

# Export Triton kernel modules for direct access if needed
try:
    from . import triton
except ImportError:
    triton = None

__version__ = "1.0.0"
__all__ = [
    "sageattn",
    "sageattn_varlen",
    "get_rocm_device_info",
    "triton",
    "AOTRITON_AVAILABLE",
]

# Print device info on import
import torch
if torch.cuda.is_available():
    device_info = get_rocm_device_info()
    if device_info:
        print(f"SageAttention ROCm: {IMPLEMENTATION} implementation on {device_info['name']}")
        if device_info.get('aotriton_available'):
            print(f"  [SUCCESS] AOTriton: Enabled (experimental)")
        elif device_info.get('triton_available'):
            print(f"  Triton: Available")
        else:
            print("  [WARNING] Using PyTorch fallback (slow)")