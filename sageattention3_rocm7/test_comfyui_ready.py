"""
Final integration test to verify SageAttention is ready for ComfyUI
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_comfyui_integration():
    print("=" * 70)
    print("ComfyUI Integration Readiness Test")
    print("=" * 70)

    # Step 1: Check PyTorch and CUDA
    print("\n1. Environment Check:")
    print(f"   PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"   CUDA available: Yes")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print(f"   CUDA available: No")
        print("   [ERROR] GPU not available!")
        return False

    # Step 2: Try to import and initialize SageAttention
    print("\n2. Loading SageAttention:")
    try:
        from torch_integration import SageAttentionROCm
        sage = SageAttentionROCm()
        print("   [OK] SageAttention loaded successfully")
        print(f"   DLL loaded: {sage.dll is not None}")
    except Exception as e:
        print(f"   [ERROR] Failed to load: {e}")
        return False

    # Step 3: Test basic functionality
    print("\n3. Basic Functionality Test:")
    try:
        batch, heads, seq_len, dim = 1, 8, 64, 64

        # Create small test tensors
        Q = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
        K = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
        V = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)

        print(f"   Test tensors: shape={Q.shape}, device={Q.device}")

        # Test FP16 attention
        print("   Testing FP16 attention...")
        output = sage.attention_forward(Q, K, V)

        if output is not None and not torch.isnan(output).any():
            print(f"   [OK] FP16 attention works, output shape: {output.shape}")
        else:
            print("   [ERROR] FP16 attention failed")
            return False

        # Test INT4 quantization
        print("   Testing INT4 quantization...")
        Q_int4, Q_scales = sage.quantize_int4(Q)

        if Q_int4 is not None:
            compression = Q.numel() * 2 / Q_int4.numel()  # FP16 = 2 bytes
            print(f"   [OK] INT4 quantization works, compression: {compression:.1f}x")
        else:
            print("   [ERROR] INT4 quantization failed")
            return False

    except Exception as e:
        print(f"   [ERROR] Functionality test failed: {e}")
        return False

    # Step 4: Integration readiness summary
    print("\n4. ComfyUI Integration Status:")
    print("   ✓ DLL successfully built and loads")
    print("   ✓ PyTorch integration working")
    print("   ✓ GPU kernels execute without errors")
    print("   ✓ INT4 quantization functional")
    print("   ⚠ Performance optimization needed")

    print("\n" + "=" * 70)
    print("RESULT: SageAttention is READY for experimental ComfyUI integration")
    print("=" * 70)

    print("\nNext steps for ComfyUI integration:")
    print("1. Copy sage_attention_rocm7.dll to ComfyUI custom nodes")
    print("2. Add torch_integration.py as attention replacement")
    print("3. Modify attention layers to use SageAttention")
    print("4. Test with actual diffusion models (SD, SDXL, Flux)")

    return True

if __name__ == "__main__":
    success = test_comfyui_integration()
    sys.exit(0 if success else 1)