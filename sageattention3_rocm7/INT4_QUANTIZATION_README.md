# INT4 Quantization Implementation for SageAttention3 ROCm Port

## Overview

This document describes the INT4 quantization implementation for the SageAttention3 ROCm port targeting AMD RDNA3.5 architecture (gfx1151 - Radeon 890M iGPU).

Since RDNA3.5 lacks native FP4 support (unlike NVIDIA Blackwell), we implement **symmetric INT4 quantization** with block-based scaling to achieve similar memory compression and compute efficiency.

## Architecture Details

### Target Hardware: AMD RDNA3.5 (gfx1151)
- **Wave Size**: 32 threads
- **Shared Memory (LDS)**: 32KB per compute unit
- **No native FP4**: Must emulate with INT4
- **Memory Bandwidth**: Critical bottleneck for attention

### Quantization Scheme

**Type**: Symmetric INT4 quantization
- **Range**: -8 to 7 (4-bit signed integer)
- **Block Size**: 16 elements per scale factor
- **Packing**: 2 INT4 values per byte
- **Scale Precision**: FP16

#### Formula

For a block of 16 FP16 values `x[0..15]`:

1. **Compute scale**: `scale = max(|x[i]|) / 7.0`
2. **Quantize**: `q[i] = round(x[i] / scale)`, clamped to `[-8, 7]`
3. **Pack**: Pack pairs `(q[2i], q[2i+1])` into bytes
4. **Dequantize**: `x'[i] = q[i] * scale`

## File Structure

```
sageattention3_rocm7/
├── hip/
│   ├── include/
│   │   └── common.h              # Shared constants and utilities
│   ├── quantization/
│   │   ├── int4_ops.hip.cpp      # INT4 quantization kernels
│   │   └── int4_ops.h            # API header
│   └── attention_forward.hip.cpp # Main attention kernel
├── test_int4_quantization.hip.cpp # Test suite
├── CMakeLists.txt                 # Build configuration
└── INT4_QUANTIZATION_README.md    # This file
```

## Implementation Details

### 1. Quantization Kernel (`quantize_int4_kernel`)

**Input Layout**: `[batch, num_tokens, num_heads, head_dim]`
**Output Layout**: `[batch, num_tokens, num_heads, head_dim/2]` (packed)
**Scales Layout**: `[batch, num_tokens, num_heads, head_dim/16]`

**Algorithm**:
```
For each thread processing 16 elements:
  1. Load 16 FP16 values
  2. Compute local absolute maximum
  3. Reduce across threads in same block (if needed)
  4. Compute scale = max_abs / 7.0
  5. Quantize: q[i] = round(value[i] / scale), clamp to [-8, 7]
  6. Pack pairs into bytes
  7. Write packed output and scale factor
```

**Optimization Techniques**:
- Vectorized loads/stores where possible
- Warp shuffle for reduction within blocks
- Coalesced memory access patterns
- Register-based computation

### 2. Dequantization Kernel (`dequantize_int4_kernel`)

**Input Layout**: `[batch, num_tokens, num_heads, head_dim/2]` (packed)
**Scales Layout**: `[batch, num_tokens, num_heads, head_dim/16]`
**Output Layout**: `[batch, num_tokens, num_heads, head_dim]`

**Algorithm**:
```
For each thread:
  1. Load scale factor for block
  2. Load packed INT4 values
  3. Unpack pairs of INT4 values (with sign extension)
  4. Dequantize: value[i] = q[i] * scale
  5. Write FP16 output
```

### 3. Transpose Quantization Kernel (`quantize_int4_transpose_kernel`)

Used for K matrix in attention computation (Q @ K^T).

**Input Layout**: `[batch, num_tokens, num_heads, head_dim]`
**Output Layout**: `[batch, num_heads, head_dim, num_tokens/2]` (transposed + packed)

**Algorithm**:
```
1. Load input tile to shared memory
2. Synchronize threads
3. Transpose access pattern in shared memory
4. Quantize transposed values
5. Pack and write output
```

**Shared Memory Usage**: `BLOCK_SIZE * HEAD_DIM * sizeof(half)` bytes
- For BLOCK_SIZE=128, HEAD_DIM=64: 16KB
- For BLOCK_SIZE=128, HEAD_DIM=128: 32KB (max for RDNA3.5)

## Memory Layout and Packing

### Packed INT4 Format

Each byte stores two INT4 values:
```
Byte: [b3 b2 b1 b0 | a3 a2 a1 a0]
      MSB nibble    LSB nibble
      (value 1)     (value 0)
```

### Sign Extension

INT4 values use 4 bits with range -8 to 7:
- `0000` = 0
- `0111` = 7 (max positive)
- `1000` = -8 (min negative)
- `1111` = -1

When unpacking, we perform sign extension:
```cpp
int8_t nibble = packed & 0x0F;
int4_t value = (nibble > 7) ? (nibble - 16) : nibble;
```

### Block Scaling Layout

Scale factors are stored with stride:
```
scale_index = (token_id * num_heads + head_id) * (head_dim / 16) + block_id
```

Each scale factor covers 16 consecutive elements in the feature dimension.

## Performance Characteristics

### Memory Compression

- **Original FP16**: 2 bytes per element
- **INT4 Packed**: 0.5 bytes per element
- **Scale Overhead**: 1 FP16 scale per 16 elements = 0.125 bytes/element
- **Total**: 0.625 bytes/element
- **Compression Ratio**: 3.2x

### Accuracy Trade-offs

INT4 quantization introduces quantization error:

- **Theoretical max error**: ±(scale / 2) per element
- **Expected relative error**: ~7-10% for well-distributed values
- **Best case**: Values close to quantization grid points
- **Worst case**: Values between grid points

For attention, this is typically acceptable because:
1. Attention scores are normalized (softmax)
2. Per-block mean subtraction reduces dynamic range
3. Small errors in intermediate results are often negligible

### Throughput Estimates

For RDNA3.5 (Radeon 890M):
- **Memory Bandwidth**: ~120 GB/s
- **Quantization Throughput**: Limited by memory bandwidth
  - FP16 load: 2 bytes/element
  - INT4 store: 0.5 bytes/element
  - Scale store: 0.125 bytes/element
  - Total: 2.625 bytes per input element
  - Expected: ~45 Gelements/s

## Usage Example

### C++ API

```cpp
#include "hip/quantization/int4_ops.h"

// Allocate device memory
half* d_input;           // [batch, tokens, heads, dim]
uint8_t* d_output;       // [batch, tokens, heads, dim/2]
half* d_scales;          // [batch, tokens, heads, dim/16]

// Create HIP stream
hipStream_t stream;
hipStreamCreate(&stream);

// Launch quantization
launch_quantize_int4(
    d_input, d_output, d_scales,
    batch_size, num_tokens, num_heads, head_dim,
    stride_batch_in, stride_token_in, stride_head_in,
    stride_batch_out, stride_token_out, stride_head_out,
    stride_batch_scale, stride_token_scale, stride_head_scale,
    stream
);

// Synchronize
hipStreamSynchronize(stream);
```

## Building and Testing

### Prerequisites

- ROCm 7.0 or later
- HIP compiler (hipcc)
- CMake 3.16+
- C++17 compatible compiler

### Build Instructions

```bash
cd sageattention3_rocm7
mkdir build
cd build

# Configure with HIP
cmake .. -DCMAKE_CXX_COMPILER=hipcc

# Build
cmake --build . -j$(nproc)

# Run test
./test_int4_quantization
```

### Expected Test Output

```
INT4 Quantization Test Suite for RDNA3.5
=========================================

Using HIP device: AMD Radeon 890M
Architecture: gfx1151
Compute units: 16
Max shared memory per block: 65 KB

Testing INT4 Quantization/Dequantization...
Configuration:
  Batch size: 2
  Num tokens: 128
  Num heads: 8
  Head dim: 64

Generated 131072 random FP16 values
Allocated device memory
Launching quantization kernel...
Quantization completed
Launching dequantization kernel...
Dequantization completed

=== Quantization Quality Metrics ===
Mean absolute error: 0.0523
Max absolute error: 0.2341

✓ TEST PASSED: Quantization errors within acceptable range

Sample values (first 8 elements):
Original -> Reconstructed (Error)
  0.523 -> 0.514 (0.009)
  -0.812 -> -0.800 (0.012)
  ...
```

## Integration with Attention Kernel

The INT4 quantization integrates with the attention forward pass:

1. **Input**: FP16 Q, K, V tensors
2. **Quantize K**: Apply `quantize_int4_transpose` to K
3. **Quantize V**: Apply `quantize_int4` to V
4. **Compute Q @ K^T**: Use INT4 K with dequantization on-the-fly
5. **Apply Softmax**: Use per-block mean subtraction (delta_s)
6. **Compute Attn @ V**: Use INT4 V with dequantization
7. **Output**: FP16 output tensor

### Memory Savings in Attention

For sequence length N, head dimension D:
- **Without quantization**: K and V require 2 * N * D * 2 bytes
- **With INT4 quantization**: K and V require 2 * N * D * 0.625 bytes
- **Savings**: ~3.2x reduction in KV cache size

This is crucial for long-context attention where KV cache dominates memory.

## Limitations and Future Work

### Current Limitations

1. **Fixed Block Size**: Currently hardcoded to 16 elements
2. **Head Dimension**: Only supports 64 and 128
3. **Symmetric Quantization**: Asymmetric might be better for skewed distributions
4. **No Mixed Precision**: All quantization is INT4 (no fallback to FP16)

### Future Enhancements

1. **Dynamic Block Size**: Adapt block size based on tensor characteristics
2. **Asymmetric Quantization**: Zero-point offset for better range coverage
3. **Outlier Handling**: Detect and handle outliers separately
4. **Per-Channel Quantization**: Different scales per attention head
5. **INT8 Fallback**: Use INT8 for tensors that don't compress well to INT4
6. **Fused Kernels**: Combine quantization with attention computation

## References

1. SageAttention3 Blackwell Implementation (FP4 quantization)
2. AMD RDNA3 Architecture Whitepaper
3. ROCm HIP Programming Guide
4. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob et al.)

## License

Apache License 2.0 (same as original SageAttention project)
