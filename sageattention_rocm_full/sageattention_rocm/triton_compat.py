"""
Compatibility module for Triton on different platforms
Handles differences between standard Triton and pytorch-triton-rocm
"""

import sys
import warnings

# Try different Triton imports
TRITON_AVAILABLE = False
triton = None

# Try standard triton first
try:
    import triton as _triton
    import triton.language as tl
    triton = _triton
    TRITON_AVAILABLE = True
    print("Using standard Triton")
except ImportError:
    pass

# Try pytorch-triton-rocm
if not TRITON_AVAILABLE:
    try:
        import pytorch_triton_rocm as _triton
        import pytorch_triton_rocm.language as tl
        triton = _triton
        TRITON_AVAILABLE = True
        print("Using pytorch-triton-rocm")
    except ImportError:
        pass

# Try torch's internal triton
if not TRITON_AVAILABLE:
    try:
        import torch._inductor.triton_heuristics as _triton
        # This is limited but might work for basic ops
        triton = _triton
        TRITON_AVAILABLE = True
        print("Using torch internal Triton (limited functionality)")
    except ImportError:
        pass

# If still not available, create mock for testing
if not TRITON_AVAILABLE:
    warnings.warn(
        "Triton not available. Creating mock implementation for testing. "
        "This will not provide actual GPU acceleration!"
    )

    class MockTriton:
        """Mock Triton for testing without GPU acceleration."""

        @staticmethod
        def jit(*args, **kwargs):
            """Mock JIT decorator."""
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def Config(*args, **kwargs):
            """Mock Config."""
            return {}

        @staticmethod
        def autotune(*args, **kwargs):
            """Mock autotune decorator."""
            def decorator(func):
                return func
            return decorator

        class language:
            """Mock Triton language module."""

            @staticmethod
            def load(*args, **kwargs):
                import torch
                return torch.zeros(1)

            @staticmethod
            def store(*args, **kwargs):
                pass

            @staticmethod
            def arange(*args, **kwargs):
                import torch
                return torch.arange(*args, **kwargs)

            @staticmethod
            def dot(*args, **kwargs):
                import torch
                if len(args) == 2:
                    return torch.matmul(args[0], args[1])
                return torch.zeros(1)

            # Add more mock functions as needed
            @staticmethod
            def exp(*args, **kwargs):
                import torch
                return torch.exp(args[0])

            @staticmethod
            def max(*args, **kwargs):
                import torch
                return torch.max(*args, **kwargs)

            @staticmethod
            def sum(*args, **kwargs):
                import torch
                return torch.sum(*args, **kwargs)

            # Constants
            float32 = 'float32'
            float16 = 'float16'
            int8 = 'int8'
            int32 = 'int32'

    triton = MockTriton()
    tl = MockTriton.language
    TRITON_AVAILABLE = False  # Mark as false since it's just mock

# Export the module and availability flag
__all__ = ['triton', 'tl', 'TRITON_AVAILABLE']

def check_triton_availability():
    """Check if real Triton is available and provide installation instructions."""
    if not TRITON_AVAILABLE:
        print("\n" + "="*60)
        print("Triton is not available!")
        print("="*60)
        print("\nFor ROCm on Linux:")
        print("  pip install triton")
        print("\nFor Windows with ROCm:")
        print("  Triton is not officially supported on Windows.")
        print("  Consider using WSL2 or Linux for full functionality.")
        print("\nCurrently using mock implementation for testing only.")
        print("This will NOT provide GPU acceleration!")
        print("="*60 + "\n")
        return False
    return True