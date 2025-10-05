"""
Validation suite for SageAttention3 ROCm kernels.

This package provides:
- GPU kernel correctness tests (test_gpu_kernels.py)
- Performance benchmarks (benchmark_gpu.py)
"""

from pathlib import Path

__version__ = "0.1.0"
__all__ = ["test_gpu_kernels", "benchmark_gpu"]
