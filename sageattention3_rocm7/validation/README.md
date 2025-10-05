# SageAttention3 ROCm Validation Suite

Comprehensive validation and benchmarking suite for the SageAttention3 ROCm DLL.

## Overview

This validation suite ensures that the compiled HIP kernels (`sage_attention_rocm7.dll`) work correctly on AMD RDNA3.5 GPUs and provides performance benchmarks.

## Components

### 1. PyTorch Integration (`torch_integration.py`)

Python wrapper that loads the DLL and provides PyTorch-compatible interfaces:

```python
from torch_integration import SageAttentionROCm

# Initialize
sage = SageAttentionROCm()

# Run attention
Q, K, V = ...  # Your tensors [batch, heads, seq_len, dim], FP16
output = sage.attention_forward(Q, K, V)

# Quantize to INT4
x = ...  # Your tensor [batch, tokens, heads, dim], FP16
quantized, scales = sage.quantize_int4(x, block_size=16)

# Dequantize back to FP16
reconstructed = sage.dequantize_int4(quantized, scales, head_dim=dim)
```

### 2. Correctness Tests (`test_gpu_kernels.py`)

Validates kernel correctness by comparing against PyTorch reference implementations:

- **INT4 Quantization Tests**: Roundtrip accuracy, symmetry, memory savings
- **Attention Tests**: Correctness vs PyTorch, causal masking, edge cases
- **Memory Tests**: Large batch handling, memory efficiency
- **Edge Cases**: Minimum dimensions, numerical stability

### 3. Performance Benchmarks (`benchmark_gpu.py`)

Measures performance metrics:

- **TFLOPS**: Computational throughput
- **Speedup**: Comparison with PyTorch baseline
- **Memory Bandwidth**: GB/s utilization
- **Scaling**: Performance across different batch sizes and sequence lengths

### 4. Validation Runner (`run_validation.py`)

Orchestrates the complete validation workflow:

1. Environment checks
2. Correctness tests
3. Performance benchmarks
4. Report generation

## Quick Start

### Prerequisites

1. **AMD GPU**: RDNA3.5 (gfx1151) or compatible
2. **ROCm 7**: Installed in `.venv` with PyTorch
3. **DLL**: `sage_attention_rocm7.dll` built and in parent directory

### Running Validation

**Option 1: Run everything (recommended)**

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run complete validation
python validation/run_validation.py
```

This will:
- Run all correctness tests
- Run performance benchmarks
- Generate `VALIDATION_REPORT.md`

**Option 2: Run tests only**

```bash
python validation/test_gpu_kernels.py
```

**Option 3: Run benchmarks only**

```bash
python validation/benchmark_gpu.py
```

## Test Coverage

### INT4 Quantization Tests

- ✓ Roundtrip accuracy (quantize → dequantize)
- ✓ Symmetry (Q(-x) ≈ -Q(x))
- ✓ Zero input handling
- ✓ Memory savings (~4x compression)
- ✓ Range value handling (0.1x to 100x scale)

### Attention Forward Tests

- ✓ Basic attention vs PyTorch reference
- ✓ Causal masking
- ✓ Different sequence lengths (Q vs K/V)
- ✓ Single token edge case
- ✓ Numerical stability (extreme values)

### Memory Efficiency Tests

- ✓ Large batch handling
- ✓ INT4 vs FP16 memory comparison

### Edge Case Tests

- ✓ Minimum dimensions
- ✓ Power-of-2 dimensions (64, 128, 256, 512)

## Performance Targets

For AMD Radeon 890M (RDNA3.5, gfx1151):

- **Attention TFLOPS**: 1-5 TFLOPS (depends on config)
- **Speedup vs PyTorch**: 0.8-2.0x
- **Memory Savings (INT4)**: 3-4x compression
- **Bandwidth**: 50-200 GB/s

Note: RDNA3.5 is optimized for gaming/inference, not HPC, so lower TFLOPS than CDNA3 is expected.

## Output Files

- `benchmark_results.json`: Detailed benchmark data
- `VALIDATION_REPORT.md`: Comprehensive validation report

## Troubleshooting

### DLL not found

Make sure `sage_attention_rocm7.dll` is in the parent directory:
```
sageattention3_rocm7/
  ├── sage_attention_rocm7.dll  ← Here
  └── validation/
      ├── test_gpu_kernels.py
      └── ...
```

### amdhip64_7.dll not found

The ROCm HIP runtime must be in your PATH. The integration layer automatically adds:
```
.venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin/
```

If this fails, manually add it to PATH:
```bash
set PATH=D:\development\SageAttention\.venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin;%PATH%
```

### CUDA not available

Make sure PyTorch recognizes your ROCm GPU:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your AMD GPU
```

### Tests failing

1. Check that the DLL exports the correct functions:
   ```python
   import ctypes
   dll = ctypes.CDLL("sage_attention_rocm7.dll")
   print(dir(dll))  # Should include launch_sage_attention_forward, etc.
   ```

2. Check for HIP errors in the output

3. Verify tensor shapes and data types match requirements

## Integration with ComfyUI

Once validation passes, you can integrate into ComfyUI:

1. Copy `torch_integration.py` to your ComfyUI custom nodes
2. Use the `SageAttentionROCm` class in your attention layers
3. Replace standard attention with quantized attention for inference

Example:
```python
from torch_integration import attention_forward

# In your attention layer
def forward(self, Q, K, V):
    # Use SageAttention instead of torch.nn.functional.scaled_dot_product_attention
    return attention_forward(Q, K, V, scale=self.scale, is_causal=self.is_causal)
```

## Contributing

To add new tests:

1. Add test methods to the appropriate test class in `test_gpu_kernels.py`
2. Follow the existing test structure
3. Use descriptive names and docstrings
4. Run the full validation suite to ensure no regressions

## License

Same as parent SageAttention project.
