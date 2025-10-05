"""Quick test to check if quantization works"""
import sys
import torch

# Make sure we don't import from source directory
if 'D:\\development\\SageAttention' in sys.path:
    sys.path.remove('D:\\development\\SageAttention')

print("Python path:", sys.path[:3])

try:
    from sageattention3_rocm7 import scale_and_quant_int4, __version__
    import sageattention3_rocm7
    print(f"Package version: {__version__}")
    print(f"Loaded from: {sageattention3_rocm7.__file__}")

    # Test quantization
    device = torch.device("cuda:0")
    x = torch.randn(2, 8, 128, 64, dtype=torch.float16, device=device)

    print(f"\nTesting INT4 quantization...")
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")

    try:
        q_int4, scales = scale_and_quant_int4(x)
        print(f"✓ SUCCESS!")
        print(f"  Quantized: {q_int4.shape}, dtype: {q_int4.dtype}")
        print(f"  Scales: {scales.shape}, dtype: {scales.dtype}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

except Exception as e:
    print(f"Import or test failed: {e}")
    import traceback
    traceback.print_exc()
