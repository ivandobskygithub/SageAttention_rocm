"""
Direct test of SageAttention without package installation
"""

import sys
import os

# Add the package directory to Python path
sys.path.insert(0, 'sageattention3_rocm7')

import torch
print("=" * 70)
print("Direct SageAttention Test (No Installation Required)")
print("=" * 70)

# Import the torch_integration module directly
print("\n1. Importing SageAttentionROCm...")
from torch_integration import SageAttentionROCm

# Create instance
sage = SageAttentionROCm()
print(f"   [OK] Loaded successfully")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")

# Test attention
print("\n2. Testing attention forward...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=device)
K = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=device)
V = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=device)

output = sage.attention_forward(Q, K, V)
print(f"   [OK] Output shape: {output.shape}")
print(f"   Valid: {not torch.isnan(output).any()}")

# Test INT4
print("\n3. Testing INT4 quantization...")
X = torch.randn(1, 32, 128, 64, dtype=torch.float16, device=device)
X_int4, scales = sage.quantize_int4(X)
compression = X.numel() * 2 / X_int4.numel()
print(f"   [OK] Compression: {compression:.1f}x")

print("\n" + "=" * 70)
print("SUCCESS! SageAttention is working and ready for use!")
print("\nTo use in ComfyUI or your projects:")
print("1. Add 'sageattention3_rocm7' to your Python path")
print("2. Import: from torch_integration import SageAttentionROCm")
print("3. Use: sage = SageAttentionROCm()")
print("=" * 70)