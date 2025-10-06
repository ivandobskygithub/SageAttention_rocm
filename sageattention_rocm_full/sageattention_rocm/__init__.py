"""
SageAttention ROCm Implementation - Works with or without Triton
"""

import warnings

# Try to use full implementation, fall back to simple if imports fail
try:
    from .core_triton import (
        sageattn,
        sageattn_varlen,
        get_rocm_device_info,
    )
    USING_FALLBACK = False
except ImportError as e:
    warnings.warn(f"Failed to import Triton kernels: {e}\nUsing simplified fallback implementation.")
    from .core_triton_simple import (
        sageattn,
        sageattn_varlen,
        get_rocm_device_info,
    )
    USING_FALLBACK = True

# Export Triton kernel modules for direct access if needed
try:
    from . import triton
except ImportError:
    triton = None
    warnings.warn("Triton modules not available")

__version__ = "1.0.0"
__all__ = [
    "sageattn",
    "sageattn_varlen",
    "get_rocm_device_info",
    "triton",
]

# Print device info on import
import torch
if torch.cuda.is_available():
    device_info = get_rocm_device_info()
    if device_info:
        status = "Fallback" if USING_FALLBACK else "Full"
        triton_status = "Available" if device_info.get('triton_available', False) else "Not Available"
        print(f"SageAttention ROCm: {status} implementation on {device_info['name']}")
        print(f"  Triton: {triton_status}")
        if USING_FALLBACK:
            print("  ⚠️ Using PyTorch fallback (slow). Install Triton for acceleration.")