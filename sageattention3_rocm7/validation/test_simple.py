"""
Simple diagnostic test to identify the HIP error.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_integration import SageAttentionROCm

print("=" * 80)
print("Simple Diagnostic Test")
print("=" * 80)

# Initialize
print("\n1. Initializing SageAttention...")
sage = SageAttentionROCm()
print("   [OK] Initialization successful")

# Check device
device = torch.device("cuda:0")
print(f"\n2. Device: {torch.cuda.get_device_name(0)}")
print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test simple tensor creation
print("\n3. Creating test tensors...")
try:
    batch, heads, seq_len, dim = 1, 2, 16, 64
    Q = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
    K = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
    V = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
    print(f"   [OK] Tensors created: Q.shape={Q.shape}")
except Exception as e:
    print(f"   [ERROR] Error creating tensors: {e}")
    sys.exit(1)

# Test INT4 quantization (simpler operation)
print("\n4. Testing INT4 quantization...")
try:
    x = torch.randn(1, 32, 4, 64, dtype=torch.float16, device=device)
    print(f"   Input: {x.shape}, device={x.device}, dtype={x.dtype}")

    # Call quantization
    x_quant, scales = sage.quantize_int4(x, block_size=16)
    print(f"   [OK] Quantization successful")
    print(f"     Quantized: {x_quant.shape}, dtype={x_quant.dtype}")
    print(f"     Scales: {scales.shape}, dtype={scales.dtype}")

    # Test dequantization
    x_recon = sage.dequantize_int4(x_quant, scales, head_dim=64)
    print(f"   [OK] Dequantization successful: {x_recon.shape}")

    # Check error
    error = torch.mean(torch.abs(x - x_recon)).item()
    print(f"   Mean reconstruction error: {error:.6f}")

except Exception as e:
    print(f"   [ERROR] Quantization failed: {e}")
    import traceback
    traceback.print_exc()

# Test attention forward (more complex)
print("\n5. Testing attention forward (small)...")
try:
    batch, heads, seq_len, dim = 1, 2, 16, 64
    Q = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
    K = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
    V = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)

    print(f"   Inputs: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"   Q device={Q.device}, dtype={Q.dtype}, contiguous={Q.is_contiguous()}")

    # Enable debugging
    os.environ["AMD_SERIALIZE_KERNEL"] = "3"

    O = sage.attention_forward(Q, K, V, scale=1.0/8.0, is_causal=False)
    print(f"   [OK] Attention successful: O={O.shape}")

    # Check for NaN
    if torch.isnan(O).any():
        print(f"   [WARNING] Output contains NaN")
    else:
        print(f"   [OK] Output is valid (no NaN)")

except Exception as e:
    print(f"   [ERROR] Attention failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Diagnostic Complete")
print("=" * 80)
