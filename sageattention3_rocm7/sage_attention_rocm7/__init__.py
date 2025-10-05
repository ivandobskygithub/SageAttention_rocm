"""
SageAttention3 ROCm7 Port

INT4 quantized attention implementation optimized for AMD RDNA3.5 (gfx1151)
using HIP and rocWMMA.

This package provides efficient attention kernels for AMD GPUs using:
- INT4 block-scaled quantization for reduced memory bandwidth
- rocWMMA for matrix operations on RDNA architecture
- HIP for GPU acceleration on ROCm

Example usage:
    import torch
    import sage_attention_rocm7 as sage_attn

    # Create input tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    # Compute attention
    output = sage_attn.attention_forward(Q, K, V, scale=1.0 / (head_dim ** 0.5))
"""

__version__ = "0.1.0"
__author__ = "SageAttention ROCm Port Team"

# Try to import the compiled C extension
try:
    from . import _C
    _has_extension = True
except ImportError as e:
    _has_extension = False
    _import_error = e

# Import Python API functions
from .ops import (
    attention_forward,
    quantize_int4,
    dequantize_int4,
)

# Export public API
__all__ = [
    'attention_forward',
    'quantize_int4',
    'dequantize_int4',
    '__version__',
]

# Version and build information
def get_build_info():
    """Get information about the build configuration"""
    info = {
        'version': __version__,
        'has_compiled_extension': _has_extension,
    }

    if not _has_extension:
        info['import_error'] = str(_import_error)

    return info

def check_compatibility():
    """Check if the package is compatible with the current system"""
    import torch

    issues = []

    # Check PyTorch installation
    if not torch.cuda.is_available():
        issues.append("PyTorch CUDA/ROCm support not available")

    # Check for ROCm
    try:
        device_name = torch.cuda.get_device_name(0)
        if 'AMD' not in device_name and 'Radeon' not in device_name:
            issues.append(f"Non-AMD GPU detected: {device_name}")
    except:
        issues.append("Unable to detect GPU")

    # Check compiled extension
    if not _has_extension:
        issues.append(f"Compiled extension not loaded: {_import_error}")

    return {
        'compatible': len(issues) == 0,
        'issues': issues,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

# Print warning if extension failed to load
if not _has_extension:
    import warnings
    warnings.warn(
        f"sage_attention_rocm7 C extension failed to load: {_import_error}\n"
        "The package is installed but the compiled kernels are not available.\n"
        "Please rebuild the package with: python setup.py build_ext --inplace",
        ImportWarning
    )
