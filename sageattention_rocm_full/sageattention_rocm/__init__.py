"""
SageAttention ROCm Implementation - Triton Backend
"""

# Use Triton-only implementation for ROCm 7
from .core_triton import (
    sageattn,
    sageattn_varlen,
    get_rocm_device_info,
)

# Export Triton kernel modules for direct access if needed
from . import triton

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
        print(f"SageAttention ROCm: Initialized on {device_info['name']} ({device_info.get('arch', 'unknown')})")