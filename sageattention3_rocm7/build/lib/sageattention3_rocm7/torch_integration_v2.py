"""
PyTorch Integration Layer for SageAttention3 ROCm DLL - Version 2
Enhanced with HIP 7 compatibility and proper library loading
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
    Enhanced for HIP 7 compatibility.
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
        self.hip_version = 7  # Assume HIP 7

        # Override system ROCm installation
        self._override_system_rocm()

        # Pre-load HIP libraries from venv
        self._preload_hip_libraries(venv_path)

        # Load our DLL
        self._load_dll(dll_path, venv_path)

        # Setup function signatures
        self._setup_function_signatures()

        # Detect GPU
        self._detect_gpu()

    def _override_system_rocm(self):
        """Override system ROCm environment variables to use venv."""
        # Clear system ROCm paths
        if "HIP_PATH" in os.environ:
            print(f"[WARNING] System HIP_PATH detected: {os.environ['HIP_PATH']}")
            print("[INFO] Overriding with venv ROCm...")
            del os.environ["HIP_PATH"]

        # Clear other ROCm variables
        for var in ["ROCM_HOME", "ROCM_PATH", "HSA_PATH"]:
            if var in os.environ:
                del os.environ[var]

    def _preload_hip_libraries(self, venv_path: Optional[str]):
        """Pre-load HIP libraries from venv to prevent system library conflicts."""
        # Detect venv path
        if venv_path is None:
            repo_root = Path(__file__).parent.parent.parent
            venv_path = repo_root / ".venv"
        else:
            venv_path = Path(venv_path)

        if not venv_path.exists():
            print(f"[WARNING] venv not found at {venv_path}")
            return

        # Find ROCm SDK in venv
        rocm_base = venv_path / "Lib/site-packages/_rocm_sdk_core"
        if not rocm_base.exists():
            print(f"[WARNING] ROCm SDK not found in venv at {rocm_base}")
            return

        # Set environment for venv ROCm
        os.environ["HIP_PATH"] = str(rocm_base)
        os.environ["ROCM_HOME"] = str(rocm_base)
        os.environ["HIP_PLATFORM"] = "amd"

        # Add DLL directories
        dll_dirs = [
            rocm_base / "lib/llvm/bin",
            rocm_base / "lib/rocm/bin",
            rocm_base / "lib",
        ]

        for dll_dir in dll_dirs:
            if dll_dir.exists():
                # Add to PATH
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

                # For Python 3.8+, use add_dll_directory
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(str(dll_dir))
                    print(f"[INFO] Added DLL directory: {dll_dir}")

        # Pre-load critical HIP DLLs in order
        critical_dlls = [
            "amdhip64_7.dll",
            "amd_comgr_2.dll",
            "hiprtc0705_7.dll",
            "rocblas.dll",
        ]

        hip_dll_path = rocm_base / "lib/llvm/bin"
        for dll_name in critical_dlls:
            dll_file = hip_dll_path / dll_name
            if dll_file.exists():
                try:
                    ctypes.CDLL(str(dll_file))
                    print(f"[INFO] Pre-loaded {dll_name}")
                except OSError as e:
                    print(f"[WARNING] Could not pre-load {dll_name}: {e}")

    def _load_dll(self, dll_path: Optional[str], venv_path: Optional[str]):
        """Load the DLL with proper dependency resolution."""
        # Auto-detect DLL path
        if dll_path is None:
            dll_path = Path(__file__).parent / "sage_attention_rocm7.dll"
        else:
            dll_path = Path(dll_path)

        if not dll_path.exists():
            raise FileNotFoundError(f"DLL not found at {dll_path}")

        # Try to load the DLL
        try:
            self.dll = ctypes.CDLL(str(dll_path.resolve()))
            print(f"[SUCCESS] Loaded DLL: {dll_path}")
        except OSError as e:
            # Detailed error reporting
            error_msg = f"Failed to load DLL: {e}\n"
            error_msg += f"DLL path: {dll_path}\n"
            error_msg += f"Current PATH: {os.environ.get('PATH', 'NOT SET')}\n"
            error_msg += f"HIP_PATH: {os.environ.get('HIP_PATH', 'NOT SET')}\n"

            # Check for amdhip64_7.dll specifically
            hip_dll = Path(os.environ.get("HIP_PATH", "")) / "lib/llvm/bin/amdhip64_7.dll"
            if hip_dll.exists():
                error_msg += f"amdhip64_7.dll found at: {hip_dll}\n"
            else:
                error_msg += "amdhip64_7.dll NOT FOUND in expected location\n"

            raise RuntimeError(error_msg) from e

    def _setup_function_signatures(self):
        """Define C function signatures for ctypes with HIP 7 compatibility."""
        # Get HIP stream type (opaque pointer)
        hipStream_t = ctypes.c_void_p

        # launch_sage_attention_forward - Updated for HIP 7
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
            hipStream_t       # stream (must be valid or NULL for default)
        ]
        self.dll.launch_sage_attention_forward.restype = ctypes.c_int  # Return hipError_t

        # Other function signatures remain the same but with return types
        self.dll.launch_quantize_int4.restype = ctypes.c_int
        self.dll.launch_dequantize_int4.restype = ctypes.c_int
        self.dll.launch_quantize_int4_transpose.restype = ctypes.c_int

    def _detect_gpu(self):
        """Detect and configure AMD GPU."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm not available in PyTorch")

        self.device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        print(f"[INFO] GPU: {props.name}")
        print(f"[INFO] Total memory: {props.total_memory / 1e9:.2f} GB")

    def _validate_tensor(self, tensor: torch.Tensor, name: str,
                        dtype: torch.dtype, ndim: int):
        """Validate tensor properties with HIP 7 requirements."""
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on CUDA/HIP device")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tensor.dim() != ndim:
            raise ValueError(f"{name} must be {ndim}D, got {tensor.dim()}D")
        if not tensor.is_contiguous():
            # HIP 7 requires contiguous memory
            raise ValueError(f"{name} must be contiguous")

        # Check alignment (HIP 7 is stricter)
        if tensor.data_ptr() % 16 != 0:
            warnings.warn(f"{name} is not 16-byte aligned, may cause issues")

    def _get_hip_stream(self) -> ctypes.c_void_p:
        """Get current HIP stream from PyTorch, handling HIP 7 requirements."""
        # PyTorch's CUDA stream is compatible with HIP
        stream = torch.cuda.current_stream(self.device)

        # HIP 7: NULL streams can cause segfaults, use default stream
        if stream is None:
            stream = torch.cuda.default_stream(self.device)

        # Get the raw stream pointer
        stream_ptr = stream.cuda_stream

        # HIP 7 validation: ensure stream is not 0 (NULL)
        if stream_ptr == 0:
            # Create a new stream if needed
            warnings.warn("NULL stream detected, using default stream")
            stream_ptr = None  # Let HIP use default

        return ctypes.c_void_p(stream_ptr)

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
        Compute attention forward pass with HIP 7 compatibility.

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
        # Ensure contiguous memory (HIP 7 requirement)
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        # Validate inputs
        self._validate_tensor(Q, "Q", torch.float16, 4)
        self._validate_tensor(K, "K", torch.float16, 4)
        self._validate_tensor(V, "V", torch.float16, 4)

        # Extract dimensions
        batch_size, num_heads, seq_len_q, head_dim = Q.shape
        seq_len_k = K.size(2)

        # HIP 7: Validate dimensions (no zero dimensions)
        if batch_size <= 0 or num_heads <= 0 or seq_len_q <= 0 or seq_len_k <= 0 or head_dim <= 0:
            raise ValueError(f"Invalid dimensions: batch={batch_size}, heads={num_heads}, "
                           f"seq_q={seq_len_q}, seq_k={seq_len_k}, head_dim={head_dim}")

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
            delta_s = delta_s.contiguous()
            assert delta_s.dtype == torch.float32
            assert delta_s.is_cuda
            delta_s_ptr = delta_s.data_ptr()

        # Get HIP stream (HIP 7 safe)
        stream = self._get_hip_stream()

        # Debug info
        print(f"[DEBUG] Launching attention kernel:")
        print(f"  Batch: {batch_size}, Heads: {num_heads}")
        print(f"  SeqQ: {seq_len_q}, SeqK: {seq_len_k}, HeadDim: {head_dim}")
        print(f"  Scale: {scale}, Causal: {is_causal}")
        print(f"  Q ptr: {Q.data_ptr():x}, aligned: {Q.data_ptr() % 16 == 0}")
        print(f"  Stream: {stream}")

        # Call kernel with error checking
        hip_error = self.dll.launch_sage_attention_forward(
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

        # Check for HIP errors
        if hip_error != 0:
            error_messages = {
                1: "hipErrorInvalidValue - Invalid argument",
                2: "hipErrorMemoryAllocation - Out of memory",
                3: "hipErrorInitializationError - Initialization failed",
                11: "hipErrorInvalidDevice - Invalid device",
                13: "hipErrorInvalidConfiguration - Invalid kernel configuration",
                35: "hipErrorNoDevice - No AMD GPU found",
                98: "hipErrorInvalidKernelFile - Invalid kernel",
                100: "hipErrorNoDevice - No device found",
                201: "hipErrorInvalidDevicePointer - Invalid device pointer",
                400: "hipErrorInvalidResourceHandle - Invalid resource handle",
            }
            error_msg = error_messages.get(hip_error, f"Unknown HIP error {hip_error}")
            raise RuntimeError(f"HIP kernel failed: {error_msg}")

        # Synchronize to catch async errors
        torch.cuda.synchronize()

        return O

    # Other methods remain the same but with error checking added...


# Global instance for convenience
_global_instance: Optional[SageAttentionROCm] = None


def get_sage_attention() -> SageAttentionROCm:
    """Get or create the global SageAttention instance."""
    global _global_instance
    if _global_instance is None:
        _global_instance = SageAttentionROCm()
    return _global_instance