"""
Test AOTriton availability and functionality
"""

import os
import sys
import torch

# Set environment variable before importing anything else
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Fix for Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_aotriton_availability():
    """Test if AOTriton is available in PyTorch."""
    print("=" * 60)
    print("AOTriton Availability Test")
    print("=" * 60)

    # System info
    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")

    print(f"\nEnvironment Variables:")
    print(f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: {os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', 'Not set')}")

    # Check for AOTriton ops
    print("\n" + "-" * 40)
    print("Checking for AOTriton operations...")

    aotriton_ops = []

    # Check various possible locations
    checks = [
        ('torch._triton_multi_head_attention', hasattr(torch, '_triton_multi_head_attention')),
        ('torch._triton_scaled_dot_attention', hasattr(torch, '_triton_scaled_dot_attention')),
        ('torch.ops.aten._triton_multi_head_attention',
         hasattr(torch.ops, 'aten') and hasattr(torch.ops.aten, '_triton_multi_head_attention')),
        ('torch.nn.functional._scaled_dot_product_attention',
         hasattr(torch.nn.functional, '_scaled_dot_product_attention')),
    ]

    for name, exists in checks:
        status = "[FOUND]" if exists else "[NOT FOUND]"
        print(f"  {name}: {status}")
        if exists:
            aotriton_ops.append(name)

    # Try to find any triton-related attributes
    print("\n" + "-" * 40)
    print("Searching for Triton-related attributes in torch...")

    triton_attrs = []
    for attr in dir(torch):
        if 'triton' in attr.lower():
            triton_attrs.append(attr)
            print(f"  Found: torch.{attr}")

    if hasattr(torch, 'ops'):
        for attr in dir(torch.ops):
            if 'triton' in attr.lower():
                triton_attrs.append(f"ops.{attr}")
                print(f"  Found: torch.ops.{attr}")

    # Test scaled_dot_product_attention backends
    print("\n" + "-" * 40)
    print("Testing scaled_dot_product_attention backends...")

    if torch.cuda.is_available():
        try:
            # Create test tensors
            q = torch.randn(1, 8, 64, 32, device='cuda', dtype=torch.float16)
            k = torch.randn(1, 8, 64, 32, device='cuda', dtype=torch.float16)
            v = torch.randn(1, 8, 64, 32, device='cuda', dtype=torch.float16)

            # Try different backends
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False
            ):
                output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                print("  Math backend: [WORKS]")

            # Check if any optimized backend is available
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False
                ):
                    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    print("  Flash backend: [AVAILABLE]")
            except:
                print("  Flash backend: [NOT AVAILABLE]")

            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=True
                ):
                    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    print("  Memory-efficient backend: [AVAILABLE]")
            except:
                print("  Memory-efficient backend: [NOT AVAILABLE]")

        except Exception as e:
            print(f"  Error testing backends: {e}")
    else:
        print("  CUDA not available - cannot test backends")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if aotriton_ops:
        print(f"[SUCCESS] Found {len(aotriton_ops)} AOTriton operations")
        for op in aotriton_ops:
            print(f"   - {op}")
    elif triton_attrs:
        print(f"[WARNING] Found {len(triton_attrs)} Triton-related attributes but no direct AOTriton ops")
    else:
        print("[INFO] No AOTriton operations found")
        print("\nPossible reasons:")
        print("  1. AOTriton not installed in PyTorch build")
        print("  2. Different API in this PyTorch version")
        print("  3. Need different environment setup")
        print("  4. CPU-only PyTorch build (need ROCm version)")

    return len(aotriton_ops) > 0 or len(triton_attrs) > 0


def test_sageattention_import():
    """Test importing our SageAttention implementation."""
    print("\n" + "=" * 60)
    print("SageAttention Import Test")
    print("=" * 60)

    try:
        # Add path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        from sageattention_rocm import sageattn, get_rocm_device_info

        print("[SUCCESS] Successfully imported SageAttention")

        # Get device info
        device_info = get_rocm_device_info()
        if device_info:
            print("\nDevice Information:")
            for key, value in device_info.items():
                if key != 'aotriton_status':
                    print(f"  {key}: {value}")

            if 'aotriton_status' in device_info:
                print("\nAOTriton Status:")
                for key, value in device_info['aotriton_status'].items():
                    print(f"  {key}: {value}")
        else:
            print("\n[INFO] No CUDA/ROCm device available")
            print("Using CPU-only PyTorch - need ROCm version for GPU support")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to import SageAttention: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    success = True

    # Test AOTriton availability
    aotriton_available = test_aotriton_availability()

    # Test SageAttention import
    import_success = test_sageattention_import()

    success = import_success  # Main success based on import

    # Final summary
    print("\n" + "=" * 60)
    print("Final Status")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[INFO] CPU-only PyTorch detected")
        print("To use AOTriton, you need:")
        print("  1. PyTorch with ROCm support")
        print("  2. AMD GPU (MI100/MI200/MI300 or RX 7900)")
        print("  3. ROCm drivers installed")
        print("\nCurrent PyTorch appears to be CPU-only version")
    elif success:
        if aotriton_available:
            print("[SUCCESS] AOTriton appears to be available")
            print("[SUCCESS] SageAttention imported successfully")
            print("\nReady to run validation with AOTriton acceleration!")
        else:
            print("[WARNING] AOTriton not directly detected but SageAttention works")
            print("Will use available PyTorch backends for acceleration")
    else:
        print("[ERROR] Setup incomplete - check errors above")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())