"""
Test the debug DLL to identify kernel execution issues
"""
import os
import sys
import ctypes
import torch
from pathlib import Path

# Setup environment for venv ROCm
venv_path = Path(".venv").resolve()
rocm_path = venv_path / "Lib/site-packages/_rocm_sdk_core"

# Override system ROCm
for var in ["HIP_PATH", "ROCM_HOME", "ROCM_PATH"]:
    if var in os.environ:
        del os.environ[var]

os.environ["HIP_PATH"] = str(rocm_path)
os.environ["ROCM_HOME"] = str(rocm_path)
os.environ["HIP_PLATFORM"] = "amd"

# Add DLL directories and preload dependencies
dll_dirs = [
    rocm_path / "lib/llvm/bin",
    rocm_path / "lib/rocm/bin",
    rocm_path / "lib",
]

for dll_dir in dll_dirs:
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))
        os.environ["PATH"] = str(dll_dir) + ";" + os.environ.get("PATH", "")

# Pre-load HIP DLLs
print("Pre-loading HIP dependencies...")
hip_dll = rocm_path / "lib/llvm/bin/amdhip64_7.dll"
if hip_dll.exists():
    try:
        ctypes.CDLL(str(hip_dll))
        print(f"Loaded: {hip_dll.name}")
    except OSError as e:
        print(f"Warning: Could not load {hip_dll.name}: {e}")

print("=" * 70)
print("Testing Debug DLL with Direct C Interface")
print("=" * 70)

# Load the debug DLL
dll_path = Path("sageattention3_rocm7/sage_attention_rocm7_debug.dll")
if not dll_path.exists():
    print(f"ERROR: Debug DLL not found at {dll_path}")
    sys.exit(1)

print(f"Loading DLL: {dll_path}")

try:
    dll = ctypes.CDLL(str(dll_path))
    print("SUCCESS: DLL loaded")
except OSError as e:
    print(f"ERROR: Failed to load DLL: {e}")
    sys.exit(1)

# Setup function signature for debug version
hipStream_t = ctypes.c_void_p

dll.launch_sage_attention_forward.argtypes = [
    ctypes.c_void_p,  # Q
    ctypes.c_void_p,  # K
    ctypes.c_void_p,  # V
    ctypes.c_void_p,  # O
    ctypes.c_void_p,  # delta_s
    ctypes.c_int,     # batch_size
    ctypes.c_int,     # num_heads
    ctypes.c_int,     # seq_len_q
    ctypes.c_int,     # seq_len_k
    ctypes.c_int,     # head_dim
    ctypes.c_float,   # scale
    ctypes.c_bool,    # is_causal
    hipStream_t       # stream
]
dll.launch_sage_attention_forward.restype = None

print("\n" + "=" * 70)
print("Testing with PyTorch Tensors")
print("=" * 70)

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA/ROCm not available")
    sys.exit(1)

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Test cases - start very simple
test_configs = [
    # (batch, heads, seq_len, head_dim)
    (1, 1, 16, 16),    # Absolute minimum
    (1, 1, 32, 32),    # Slightly larger
    (1, 2, 64, 32),    # Multiple heads
    (2, 4, 128, 64),   # Full test
]

for i, (b, h, s, d) in enumerate(test_configs):
    print(f"\n{'='*50}")
    print(f"Test {i+1}: batch={b}, heads={h}, seq_len={s}, head_dim={d}")
    print('='*50)

    # Create tensors
    Q = torch.randn((b, h, s, d), dtype=torch.float16, device=device).contiguous()
    K = torch.randn((b, h, s, d), dtype=torch.float16, device=device).contiguous()
    V = torch.randn((b, h, s, d), dtype=torch.float16, device=device).contiguous()
    O = torch.zeros_like(Q)

    print(f"Tensor shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Q contiguous: {Q.is_contiguous()}, ptr: 0x{Q.data_ptr():x}")
    print(f"K contiguous: {K.is_contiguous()}, ptr: 0x{K.data_ptr():x}")
    print(f"V contiguous: {V.is_contiguous()}, ptr: 0x{V.data_ptr():x}")
    print(f"O contiguous: {O.is_contiguous()}, ptr: 0x{O.data_ptr():x}")

    # Get stream
    stream = torch.cuda.current_stream(device)
    stream_ptr = stream.cuda_stream

    print(f"Stream: 0x{stream_ptr:x}")

    # Scale
    scale = 1.0 / (d ** 0.5)
    print(f"Scale: {scale}")

    print("\nCalling launch_sage_attention_forward...")
    print("-" * 40)

    try:
        # Call the debug kernel
        dll.launch_sage_attention_forward(
            Q.data_ptr(),
            K.data_ptr(),
            V.data_ptr(),
            O.data_ptr(),
            None,  # delta_s
            b, h, s, s,  # seq_len_k = seq_len_q for now
            d,
            scale,
            False,  # not causal for simplicity
            stream_ptr
        )

        # Synchronize to ensure kernel completes
        torch.cuda.synchronize()

        print("-" * 40)
        print("Kernel call completed")

        # Check output
        if torch.allclose(O, Q, rtol=1e-3):
            print("SUCCESS: Copy kernel worked (O ≈ Q)")
        else:
            o_mean = O.mean().item()
            o_max = O.max().item()
            o_min = O.min().item()
            print(f"Output stats: mean={o_mean:.6f}, max={o_max:.6f}, min={o_min:.6f}")

            if torch.isnan(O).any():
                print("WARNING: Output contains NaN")
            if torch.isinf(O).any():
                print("WARNING: Output contains Inf")

    except Exception as e:
        print(f"ERROR: Kernel call failed: {e}")
        import traceback
        traceback.print_exc()

    # Force cleanup
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("Debug Test Complete")
print("=" * 70)
print("\nCheck the console output above for detailed kernel diagnostics.")
print("The debug kernel prints information about:")
print("- Parameter validation")
print("- Pointer alignment")
print("- Grid/block configuration")
print("- Shared memory usage")
print("- HIP error codes")