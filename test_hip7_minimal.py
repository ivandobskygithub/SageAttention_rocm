"""
Minimal test for HIP 7 compatibility with SageAttention3 ROCm7
"""
import os
import sys
import torch
from pathlib import Path

# Force venv ROCm usage
venv_path = Path(".venv").resolve()
rocm_path = venv_path / "Lib/site-packages/_rocm_sdk_core"

# Override system ROCm
if "HIP_PATH" in os.environ:
    print(f"[WARN] Removing system HIP_PATH: {os.environ['HIP_PATH']}")
    del os.environ["HIP_PATH"]

os.environ["HIP_PATH"] = str(rocm_path)
os.environ["ROCM_HOME"] = str(rocm_path)

# Add DLL directories BEFORE importing
dll_path = rocm_path / "lib/llvm/bin"
if dll_path.exists():
    os.add_dll_directory(str(dll_path))
    os.environ["PATH"] = str(dll_path) + ";" + os.environ["PATH"]
    print(f"[INFO] Added venv ROCm DLL path: {dll_path}")

# Now import our module
sys.path.insert(0, str(Path("sageattention3_rocm7")))
from sageattention3_rocm7.torch_integration_v2 import SageAttentionROCm


def test_minimal():
    """Test with minimal inputs to isolate HIP 7 issues."""
    print("=" * 70)
    print("HIP 7 Minimal Test")
    print("=" * 70)

    # Initialize
    try:
        sage = SageAttentionROCm()
        print("[OK] SageAttention initialized")
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        return False

    # Check PyTorch
    if not torch.cuda.is_available():
        print("[FAIL] CUDA/ROCm not available")
        return False

    print(f"[OK] Device: {torch.cuda.get_device_name(0)}")

    # Test cases with HIP 7 dimension requirements
    test_cases = [
        # (batch, heads, seq_len, head_dim)
        (1, 1, 16, 16),    # Minimal valid case
        (1, 1, 128, 32),   # Padded to 128
        (1, 2, 128, 64),   # Multiple heads
        (2, 4, 256, 64),   # Typical case
    ]

    for i, (b, h, s, d) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {b}x{h}x{s}x{d} ---")

        # Create aligned tensors
        Q = torch.randn((b, h, s, d), dtype=torch.float16, device="cuda").contiguous()
        K = torch.randn((b, h, s, d), dtype=torch.float16, device="cuda").contiguous()
        V = torch.randn((b, h, s, d), dtype=torch.float16, device="cuda").contiguous()

        # Verify alignment
        print(f"Q aligned: {Q.data_ptr() % 16 == 0}, ptr: {Q.data_ptr():x}")
        print(f"K aligned: {K.data_ptr() % 16 == 0}, ptr: {K.data_ptr():x}")
        print(f"V aligned: {V.data_ptr() % 16 == 0}, ptr: {V.data_ptr():x}")

        try:
            # Test non-causal
            output = sage.attention_forward(Q, K, V, is_causal=False)
            print(f"[OK] Non-causal attention succeeded")
            print(f"     Output shape: {output.shape}")
            print(f"     Output mean: {output.mean().item():.6f}")

            # Check for NaN/Inf
            if torch.isnan(output).any():
                print("[WARN] Output contains NaN")
            if torch.isinf(output).any():
                print("[WARN] Output contains Inf")

            # Test causal
            output_causal = sage.attention_forward(Q, K, V, is_causal=True)
            print(f"[OK] Causal attention succeeded")

            # Verify causal mask applied
            diff = (output - output_causal).abs().mean().item()
            if diff > 0.001:
                print(f"[OK] Causal masking verified (diff: {diff:.6f})")
            else:
                print(f"[WARN] Causal mask may not be applied (diff: {diff:.6f})")

        except RuntimeError as e:
            print(f"[FAIL] Kernel failed: {e}")

            # Try to diagnose
            if "hipErrorInvalidValue" in str(e):
                print("      Likely cause: Invalid kernel arguments")
                print("      Check: Dimension validation, pointer alignment")
            elif "hipErrorInvalidConfiguration" in str(e):
                print("      Likely cause: Grid/block configuration exceeds limits")
                print(f"     Max threads per block: {torch.cuda.get_device_properties(0).max_threads_per_block}")
            elif "hipErrorInvalidDevicePointer" in str(e):
                print("      Likely cause: Invalid device pointer")
                print("      Check: Tensor allocation, memory corruption")

        except Exception as e:
            print(f"[FAIL] Unexpected error: {e}")

        # Force cleanup
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)

    return True


if __name__ == "__main__":
    # Set debugging environment
    os.environ["AMD_LOG_LEVEL"] = "3"
    os.environ["HIP_LAUNCH_BLOCKING"] = "1"  # Synchronous kernel execution

    success = test_minimal()
    sys.exit(0 if success else 1)