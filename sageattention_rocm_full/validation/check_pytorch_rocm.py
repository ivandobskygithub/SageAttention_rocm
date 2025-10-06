"""
Check PyTorch installation for ROCm support
"""

import torch
import os
import sys

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

def check_pytorch_rocm():
    print("=" * 60)
    print("PyTorch ROCm Check")
    print("=" * 60)

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"HIP Available: {hasattr(torch.version, 'hip')}")

    # Check for ROCm/HIP in version string
    if hasattr(torch.version, 'hip'):
        print(f"HIP Version: {torch.version.hip}")

    # Check backends
    print("\nBackend Information:")
    print(f"  torch.version.cuda: {getattr(torch.version, 'cuda', 'N/A')}")
    print(f"  torch.version.hip: {getattr(torch.version, 'hip', 'N/A')}")

    # Check for ROCm in version string
    if 'rocm' in torch.__version__.lower():
        print("\n[SUCCESS] ROCm version of PyTorch detected!")
        return True
    elif '+cpu' in torch.__version__.lower():
        print("\n[WARNING] CPU-only version of PyTorch detected")
        print("You have AOTriton functions but no GPU support")
        print("\nTo use GPU acceleration, install PyTorch with ROCm:")
        print("pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        return False
    else:
        print("\n[INFO] PyTorch version unclear, checking further...")

    # Check for AMD GPU
    print("\nChecking for AMD GPU...")
    try:
        import subprocess
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                              capture_output=True, text=True)
        gpu_info = result.stdout
        print(f"GPU Info:\n{gpu_info}")

        if 'AMD' in gpu_info or 'Radeon' in gpu_info:
            print("[INFO] AMD GPU detected but PyTorch may not have ROCm support")
    except Exception as e:
        print(f"Could not query GPU info: {e}")

    # Check AOTriton functions
    print("\nAOTriton Functions Available:")
    aotriton_funcs = []
    if hasattr(torch, '_triton_multi_head_attention'):
        aotriton_funcs.append('_triton_multi_head_attention')
        print("  - torch._triton_multi_head_attention")
    if hasattr(torch, '_triton_scaled_dot_attention'):
        aotriton_funcs.append('_triton_scaled_dot_attention')
        print("  - torch._triton_scaled_dot_attention")

    if aotriton_funcs:
        print(f"\n[INFO] Found {len(aotriton_funcs)} AOTriton functions")
        print("These are available but may not work without GPU support")

    return torch.cuda.is_available()


def test_aotriton_cpu_fallback():
    """Test if AOTriton functions work on CPU as fallback."""
    print("\n" + "=" * 60)
    print("Testing AOTriton Functions on CPU")
    print("=" * 60)

    if not hasattr(torch, '_triton_scaled_dot_attention'):
        print("[ERROR] AOTriton functions not found")
        return False

    try:
        # Small test tensors on CPU
        q = torch.randn(1, 8, 64, 32, dtype=torch.float32)
        k = torch.randn(1, 8, 64, 32, dtype=torch.float32)
        v = torch.randn(1, 8, 64, 32, dtype=torch.float32)

        print("\nTesting torch._triton_scaled_dot_attention on CPU...")

        # Try calling the function
        try:
            output = torch._triton_scaled_dot_attention(q, k, v)
            print("[SUCCESS] AOTriton function works on CPU!")
            print(f"Output shape: {output.shape}")
            return True
        except Exception as e:
            print(f"[INFO] AOTriton function failed on CPU: {e}")
            print("This is expected - AOTriton typically requires GPU")

        # Test regular scaled_dot_product_attention as fallback
        print("\nTesting regular scaled_dot_product_attention...")
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print(f"[SUCCESS] Regular attention works, output shape: {output.shape}")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

    return False


def main():
    print("Checking PyTorch installation for ROCm support...\n")

    has_rocm = check_pytorch_rocm()
    can_use_aotriton_cpu = test_aotriton_cpu_fallback()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if has_rocm:
        print("[SUCCESS] PyTorch has ROCm support")
        print("AOTriton should work with GPU acceleration")
    elif can_use_aotriton_cpu:
        print("[INFO] AOTriton functions work on CPU (fallback mode)")
        print("Performance will be limited without GPU")
    else:
        print("[INFO] PyTorch is CPU-only")
        print("AOTriton functions are present but need GPU for acceleration")
        print("\nNext steps:")
        print("1. Install ROCm drivers for your AMD GPU")
        print("2. Install PyTorch with ROCm support")
        print("3. Re-run validation")

    return 0


if __name__ == "__main__":
    exit(main())