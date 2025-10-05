"""
PyTorch Integration Layer for SageAttention3 ROCm DLL

This module provides a Python/PyTorch interface to the compiled HIP kernels
in sage_attention_rocm7.dll. It handles:
- DLL loading on Windows with proper dependency resolution
- PyTorch tensor wrapping and validation
- HIP stream synchronization
- Error handling and device management
"""

import os
import sys
import ctypes
from pathlib import Path
from typing import Optional, Tuple
import warnings

import torch

# Type aliases for clarity
half = ctypes.c_uint16  # FP16 represented as uint16 in ctypes


class SageAttentionROCm:
    """
    Python interface to SageAttention3 ROCm kernels.

    This class loads the compiled DLL and provides PyTorch-compatible
    methods for attention computation and INT4 quantization.
    """

    def __init__(self, dll_path: Optional[str] = None, venv_path: Optional[str] = None):
        """
        Initialize the ROCm kernel interface.

        Args:
            dll_path: Path to sage_attention_rocm7.dll (auto-detected if None)
            venv_path: Path to .venv directory (auto-detected if None)
        """
        self.dll = None
        self.device = None
        self._load_dll(dll_path, venv_path)
        self._setup_function_signatures()
        self._detect_gpu()

    def _load_dll(self, dll_path: Optional[str], venv_path: Optional[str]):
        """Load the DLL and its dependencies."""
        # Auto-detect paths if not provided
        if dll_path is None:
            dll_path = Path(__file__).parent / "sage_attention_rocm7.dll"
        else:
            dll_path = Path(dll_path)

        if not dll_path.exists():
            raise FileNotFoundError(f"DLL not found at {dll_path}")

        # Detect venv path
        if venv_path is None:
            # Try common locations
            repo_root = Path(__file__).parent.parent
            venv_candidates = [
                repo_root / ".venv",
                Path.cwd() / ".venv",
                Path(os.environ.get("VIRTUAL_ENV", ""))
            ]
            for candidate in venv_candidates:
                if candidate.exists() and (candidate / "Lib").exists():
                    venv_path = candidate
                    break

            if venv_path is None or not Path(venv_path).exists():
                warnings.warn(
                    "Could not auto-detect .venv path. HIP DLL dependencies may not load correctly."
                )

        # Add ROCm DLL path to system PATH for dependency resolution
        if venv_path:
            rocm_dll_path = Path(venv_path) / "Lib" / "site-packages" / "_rocm_sdk_core" / "lib" / "llvm" / "bin"
            if rocm_dll_path.exists():
                # Add to PATH so Windows can find amdhip64_7.dll
                os.environ["PATH"] = str(rocm_dll_path) + os.pathsep + os.environ.get("PATH", "")
                print(f"Added ROCm DLL path: {rocm_dll_path}")
            else:
                warnings.warn(f"ROCm DLL path not found: {rocm_dll_path}")

        # Load the DLL
        try:
            self.dll = ctypes.CDLL(str(dll_path))
            print(f"Successfully loaded DLL: {dll_path}")
        except OSError as e:
            raise RuntimeError(
                f"Failed to load DLL: {e}\n"
                f"DLL path: {dll_path}\n"
                f"Make sure amdhip64_7.dll is in PATH or in the same directory."
            ) from e

    def _setup_function_signatures(self):
        """Define C function signatures for ctypes."""
        # Get HIP stream type (opaque pointer)
        hipStream_t = ctypes.c_void_p

        # launch_sage_attention_forward
        self.dll.launch_sage_attention_forward.argtypes = [
            ctypes.c_void_p,  # Q
            ctypes.c_void_p,  # K
            ctypes.c_void_p,  # V
            ctypes.c_void_p,  # O
            ctypes.c_void_p,  # delta_s (can be NULL)
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # num_heads
            ctypes.c_int,     # seq_len_q
            ctypes.c_int,     # seq_len_k
            ctypes.c_int,     # head_dim
            ctypes.c_float,   # scale
            ctypes.c_bool,    # is_causal
            hipStream_t       # stream
        ]
        self.dll.launch_sage_attention_forward.restype = None

        # launch_quantize_int4
        self.dll.launch_quantize_int4.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # scales
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # num_tokens
            ctypes.c_int,     # num_heads
            ctypes.c_int,     # head_dim
            ctypes.c_int,     # stride_batch_in
            ctypes.c_int,     # stride_token_in
            ctypes.c_int,     # stride_head_in
            ctypes.c_int,     # stride_batch_out
            ctypes.c_int,     # stride_token_out
            ctypes.c_int,     # stride_head_out
            ctypes.c_int,     # stride_batch_scale
            ctypes.c_int,     # stride_token_scale
            ctypes.c_int,     # stride_head_scale
            hipStream_t       # stream
        ]
        self.dll.launch_quantize_int4.restype = None

        # launch_dequantize_int4
        self.dll.launch_dequantize_int4.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # scales
            ctypes.c_void_p,  # output
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # num_tokens
            ctypes.c_int,     # num_heads
            ctypes.c_int,     # head_dim
            ctypes.c_int,     # stride_batch_in
            ctypes.c_int,     # stride_token_in
            ctypes.c_int,     # stride_head_in
            ctypes.c_int,     # stride_batch_scale
            ctypes.c_int,     # stride_token_scale
            ctypes.c_int,     # stride_head_scale
            ctypes.c_int,     # stride_batch_out
            ctypes.c_int,     # stride_token_out
            ctypes.c_int,     # stride_head_out
            hipStream_t       # stream
        ]
        self.dll.launch_dequantize_int4.restype = None

        # launch_quantize_int4_transpose
        self.dll.launch_quantize_int4_transpose.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # scales
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # num_tokens
            ctypes.c_int,     # num_heads
            ctypes.c_int,     # head_dim
            ctypes.c_int,     # stride_batch_in
            ctypes.c_int,     # stride_token_in
            ctypes.c_int,     # stride_head_in
            ctypes.c_int,     # stride_batch_out
            ctypes.c_int,     # stride_head_out
            ctypes.c_int,     # stride_dim_out
            ctypes.c_int,     # stride_batch_scale
            ctypes.c_int,     # stride_head_scale
            ctypes.c_int,     # stride_dim_scale
            hipStream_t       # stream
        ]
        self.dll.launch_quantize_int4_transpose.restype = None

    def _detect_gpu(self):
        """Detect and configure AMD GPU."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm not available in PyTorch")

        self.device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Total memory: {props.total_memory / 1e9:.2f} GB")

    def _validate_tensor(self, tensor: torch.Tensor, name: str,
                        dtype: torch.dtype, ndim: int):
        """Validate tensor properties."""
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on CUDA/HIP device")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tensor.dim() != ndim:
            raise ValueError(f"{name} must be {ndim}D, got {tensor.dim()}D")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def _get_hip_stream(self) -> ctypes.c_void_p:
        """Get current HIP stream from PyTorch."""
        # PyTorch's CUDA stream is compatible with HIP
        stream = torch.cuda.current_stream(self.device)
        # Get the raw stream pointer
        return ctypes.c_void_p(stream.cuda_stream)

    def attention_forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: Optional[float] = None,
        is_causal: bool = False,
        delta_s: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention forward pass.

        Args:
            Q: Query tensor [batch, num_heads, seq_len_q, head_dim], FP16
            K: Key tensor [batch, num_heads, seq_len_k, head_dim], FP16
            V: Value tensor [batch, num_heads, seq_len_k, head_dim], FP16
            scale: Scaling factor (default: 1/sqrt(head_dim))
            is_causal: Apply causal mask
            delta_s: Optional per-block mean subtraction [batch, num_heads, num_blocks], FP32

        Returns:
            Output tensor [batch, num_heads, seq_len_q, head_dim], FP16
        """
        # Validate inputs
        self._validate_tensor(Q, "Q", torch.float16, 4)
        self._validate_tensor(K, "K", torch.float16, 4)
        self._validate_tensor(V, "V", torch.float16, 4)

        # Extract dimensions
        batch_size, num_heads, seq_len_q, head_dim = Q.shape
        seq_len_k = K.size(2)

        # Validate shapes
        assert K.shape == (batch_size, num_heads, seq_len_k, head_dim)
        assert V.shape == (batch_size, num_heads, seq_len_k, head_dim)

        # Default scale
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)

        # Allocate output
        O = torch.zeros_like(Q)

        # Validate delta_s if provided
        delta_s_ptr = None
        if delta_s is not None:
            assert delta_s.dtype == torch.float32
            assert delta_s.is_cuda and delta_s.is_contiguous()
            delta_s_ptr = delta_s.data_ptr()

        # Get HIP stream
        stream = self._get_hip_stream()

        # Call kernel
        self.dll.launch_sage_attention_forward(
            Q.data_ptr(),
            K.data_ptr(),
            V.data_ptr(),
            O.data_ptr(),
            delta_s_ptr,
            batch_size,
            num_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            scale,
            is_causal,
            stream
        )

        # Synchronize to catch errors
        torch.cuda.synchronize()

        return O

    def quantize_int4(
        self,
        input: torch.Tensor,
        block_size: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize FP16 tensor to INT4 with block scaling.

        Args:
            input: Input tensor [batch, num_tokens, num_heads, head_dim], FP16
            block_size: Number of elements per quantization block

        Returns:
            Tuple of (quantized_data, scales):
                - quantized_data: [batch, num_tokens, num_heads, head_dim//2], UINT8
                - scales: [batch, num_tokens, num_heads, head_dim//block_size], FP16
        """
        self._validate_tensor(input, "input", torch.float16, 4)

        batch_size, num_tokens, num_heads, head_dim = input.shape

        # Validate head_dim is divisible by 2 (packing) and block_size
        assert head_dim % 2 == 0, "head_dim must be even for INT4 packing"
        assert head_dim % block_size == 0, f"head_dim must be divisible by block_size={block_size}"

        # Allocate outputs
        output = torch.zeros(
            (batch_size, num_tokens, num_heads, head_dim // 2),
            dtype=torch.uint8,
            device=input.device
        )
        scales = torch.zeros(
            (batch_size, num_tokens, num_heads, head_dim // block_size),
            dtype=torch.float16,
            device=input.device
        )

        # Get strides
        stride_batch_in, stride_token_in, stride_head_in, _ = input.stride()
        stride_batch_out, stride_token_out, stride_head_out, _ = output.stride()
        stride_batch_scale, stride_token_scale, stride_head_scale, _ = scales.stride()

        # Get HIP stream
        stream = self._get_hip_stream()

        # Call kernel
        self.dll.launch_quantize_int4(
            input.data_ptr(),
            output.data_ptr(),
            scales.data_ptr(),
            batch_size, num_tokens, num_heads, head_dim,
            stride_batch_in, stride_token_in, stride_head_in,
            stride_batch_out, stride_token_out, stride_head_out,
            stride_batch_scale, stride_token_scale, stride_head_scale,
            stream
        )

        torch.cuda.synchronize()

        return output, scales

    def dequantize_int4(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        head_dim: int
    ) -> torch.Tensor:
        """
        Dequantize INT4 tensor back to FP16.

        Args:
            quantized: Quantized data [batch, num_tokens, num_heads, head_dim//2], UINT8
            scales: Scale factors [batch, num_tokens, num_heads, num_blocks], FP16
            head_dim: Original head dimension before quantization

        Returns:
            Reconstructed tensor [batch, num_tokens, num_heads, head_dim], FP16
        """
        self._validate_tensor(quantized, "quantized", torch.uint8, 4)
        self._validate_tensor(scales, "scales", torch.float16, 4)

        batch_size, num_tokens, num_heads, _ = quantized.shape

        # Allocate output
        output = torch.zeros(
            (batch_size, num_tokens, num_heads, head_dim),
            dtype=torch.float16,
            device=quantized.device
        )

        # Get strides
        stride_batch_in, stride_token_in, stride_head_in, _ = quantized.stride()
        stride_batch_scale, stride_token_scale, stride_head_scale, _ = scales.stride()
        stride_batch_out, stride_token_out, stride_head_out, _ = output.stride()

        # Get HIP stream
        stream = self._get_hip_stream()

        # Call kernel
        self.dll.launch_dequantize_int4(
            quantized.data_ptr(),
            scales.data_ptr(),
            output.data_ptr(),
            batch_size, num_tokens, num_heads, head_dim,
            stride_batch_in, stride_token_in, stride_head_in,
            stride_batch_scale, stride_token_scale, stride_head_scale,
            stride_batch_out, stride_token_out, stride_head_out,
            stream
        )

        torch.cuda.synchronize()

        return output


# Global instance for convenience
_global_instance: Optional[SageAttentionROCm] = None


def get_sage_attention() -> SageAttentionROCm:
    """Get or create the global SageAttention instance."""
    global _global_instance
    if _global_instance is None:
        _global_instance = SageAttentionROCm()
    return _global_instance


# Convenience functions
def attention_forward(*args, **kwargs) -> torch.Tensor:
    """Convenience wrapper for attention_forward."""
    return get_sage_attention().attention_forward(*args, **kwargs)


def quantize_int4(*args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper for quantize_int4."""
    return get_sage_attention().quantize_int4(*args, **kwargs)


def dequantize_int4(*args, **kwargs) -> torch.Tensor:
    """Convenience wrapper for dequantize_int4."""
    return get_sage_attention().dequantize_int4(*args, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("Testing SageAttention ROCm DLL loading...")
    sage = SageAttentionROCm()
    print("Successfully initialized!")
    print(f"Device: {sage.device}")
