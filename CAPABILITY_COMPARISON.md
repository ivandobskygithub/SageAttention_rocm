# SageAttention3 Capability Comparison: Blackwell vs ROCm7

## Executive Summary

This document provides a comprehensive comparison between the SageAttention3 Blackwell (CUDA) implementation and the current ROCm7 port, identifying gaps and implementation strategies.

## SageAttention3 Blackwell Capabilities

### Core Features
1. **FP4 Quantization with Block Scaling**
   - 4-bit floating point quantization for 4x memory compression
   - Block-scaled layouts with configurable block size (default: 16)
   - FP8_E4M3FN scale storage for precision
   - Support for permute and transpose operations during quantization

2. **Advanced Attention Mechanisms**
   - Scaled dot-product attention with FP4 compression
   - Causal masking support
   - Per-block mean subtraction for numerical stability
   - Group mean computation with configurable group size (128)
   - Delta S computation for block-wise attention correction

3. **Quantization Functions**
   - `scale_and_quant_fp4`: Standard FP4 quantization
   - `scale_and_quant_fp4_permute`: Quantize with permutation for K tensor
   - `scale_and_quant_fp4_transpose`: Quantize with transpose for V tensor

4. **Preprocessing Pipeline**
   - Automatic padding to 128-element boundaries
   - K-tensor mean centering
   - Q-tensor per-block or global mean subtraction
   - Delta S precomputation for attention correction

5. **Tensor Format Support**
   - Input: [batch, heads, seq_len, head_dim]
   - Head dimension limit: < 256
   - Automatic fallback to PyTorch SDPA for unsupported dimensions

6. **Data Type Support**
   - FP16 and BF16 input support
   - FP4 internal computation
   - FP8_E4M3FN scale factors
   - FP32 delta_s computation

## ROCm7 Implementation Current State

### Implemented Features ✅
1. **Basic INT4 Quantization**
   - INT4 quantization with block scaling (block_size=16)
   - 4x memory compression achieved
   - FP16 scale storage

2. **Basic Attention Forward**
   - Standard attention computation
   - Causal masking support
   - Delta S support (per-block mean)
   - Scale factor support

3. **DLL Integration**
   - Windows DLL loading with dependency resolution
   - PyTorch integration with stream management
   - Error handling and validation

4. **Tensor Operations**
   - Quantize INT4
   - Dequantize INT4
   - Basic transpose quantization

### Missing Features ❌

1. **FP4 Quantization**
   - Currently uses INT4, not FP4
   - Missing FP8_E4M3FN scale storage (uses FP16)
   - No permute quantization variant

2. **Advanced Preprocessing**
   - No automatic padding to 128 boundaries
   - No K-tensor mean centering
   - No group mean kernel (Triton implementation)
   - Limited delta_s computation

3. **Optimizations**
   - No CUTLASS/composable_kernel equivalent
   - No TMA (Tensor Memory Accelerator) equivalent
   - No warpgroup specialization
   - Performance at ~10% of baseline (needs optimization)

4. **Python Package Structure**
   - No proper setup.py for pip installation
   - Missing module organization
   - No version management
   - Limited API compatibility with Blackwell version

## Implementation Gap Analysis

### Critical Gaps (High Priority)
1. **FP4 vs INT4**: Blackwell uses true FP4, ROCm7 uses INT4
2. **Preprocessing Pipeline**: Missing critical preprocessing steps
3. **Package Installation**: Not installable via pip
4. **API Compatibility**: Different function signatures and naming

### Performance Gaps (Medium Priority)
1. **Memory Access Patterns**: Not optimized for RDNA3
2. **Kernel Fusion**: No fused operations
3. **Block Size Optimization**: Fixed at 16, not tunable

### Feature Gaps (Low Priority)
1. **Triton Kernels**: Group mean computation
2. **Extended Data Type Support**: BF16 not fully tested
3. **Debugging Tools**: Limited profiling support

## Implementation Plan

### Phase 1: API Compatibility & Packaging
- Create unified API matching Blackwell interface
- Implement proper setup.py for pip installation
- Add preprocessing pipeline (padding, mean centering)

### Phase 2: FP4 Implementation
- Implement true FP4 quantization (if feasible on RDNA3)
- Add FP8_E4M3FN scale support (or equivalent)
- Implement permute/transpose variants

### Phase 3: Performance Optimization
- Add rocWMMA for matrix operations
- Optimize memory access patterns
- Implement kernel fusion where possible

### Phase 4: Full Feature Parity
- Port Triton kernels to HIP
- Add all preprocessing options
- Complete test coverage

## Recommendations

1. **Immediate Actions**:
   - Create proper Python package with setup.py
   - Implement preprocessing pipeline for compatibility
   - Add API wrapper to match Blackwell interface

2. **Short Term** (1-2 weeks):
   - Investigate FP4 feasibility on RDNA3
   - Optimize current INT4 implementation
   - Add missing quantization variants

3. **Long Term** (1 month):
   - Full performance optimization
   - Complete feature parity
   - Production-ready package

## Technical Notes

### Architecture Differences
- **Blackwell**: Uses CUTLASS, TMA, WGMMA instructions (SM 12.0a)
- **RDNA3**: Uses wavefront operations, different memory hierarchy
- **Quantization**: FP4 may need emulation on RDNA3

### Compatibility Considerations
- Blackwell targets CUDA 12.8+
- ROCm7 uses HIP with different intrinsics
- Memory layout optimizations differ between architectures