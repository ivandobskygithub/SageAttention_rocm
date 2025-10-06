# SageAttention ROCm Full Port Implementation Plan

## Project Overview
Complete ROCm port of SageAttention (v1, v2, v2++, v3) with focus on maintaining feature parity with CUDA implementation while optimizing for AMD GPU architectures.

## Directory Structure
```
sageattention_rocm_full/
├── setup.py                      # ROCm-specific build configuration
├── sageattention_rocm/           # Main Python package
│   ├── __init__.py
│   ├── core.py                   # Core attention implementations
│   ├── quant.py                  # Quantization utilities
│   ├── fa3_wrapper.py            # FlashAttention3 wrapper (if applicable)
│   └── triton/                   # Triton kernels (platform-agnostic)
│       ├── __init__.py
│       ├── attn_qk_int8_per_block.py
│       ├── attn_qk_int8_per_block_causal.py
│       ├── attn_qk_int8_per_block_causal_varlen.py
│       ├── attn_qk_int8_block_varlen.py
│       └── quant_per_*.py
├── csrc_rocm/                    # HIP/ROCm kernels
│   ├── qattn/
│   │   ├── hip_kernels/          # HIP implementations
│   │   ├── composable_kernel/    # CK-based implementations
│   │   └── pybind/               # Python bindings
│   └── fused/
│       ├── fused.hip             # Fused operations in HIP
│       └── pybind.cpp
└── tests/                        # ROCm-specific tests
```

## Implementation Phases

### Phase 1: Foundation Setup (Week 1)
- [x] Create directory structure
- [ ] Port build system (setup.py) for ROCm
- [ ] Set up HIP compilation flags
- [ ] Configure ROCm architecture detection (gfx908, gfx90a, gfx940, gfx1030, gfx1100)

### Phase 2: Core Infrastructure (Week 1-2)
- [ ] Convert utility headers to HIP:
  - `dispatch_utils.h` → Pure C++, minimal changes
  - `math.cuh` → `math.hip.h` (HIP math functions)
  - `utils.cuh` → `utils.hip.h`
  - `reduction_utils.cuh` → `reduction_utils.hip.h`
- [ ] Port numeric conversion utilities
- [ ] Adapt memory operations for ROCm

### Phase 3: Kernel Conversion Strategy

#### 3.1 SM80/86 Kernels → MI100/MI200 (Week 2-3)
**Target**: `qk_int_sv_f16_cuda_sm80.cu` → `qk_int_sv_f16_hip_gfx90a.hip`
- Replace CUDA intrinsics with HIP equivalents
- Use rocWMMA for tensor core operations
- Adapt shared memory patterns

#### 3.2 SM89 Kernels → MI300 (Week 3-4)
**Target**: Multiple FP8 kernels
- Port FP8 operations using ROCm FP8 support
- Replace WGMMA with MFMA instructions
- Implement instruction buffer mechanisms

#### 3.3 SM90 Kernels → Future MI300X optimizations (Week 4-5)
**Target**: `qk_int_sv_f8_cuda_sm90.cu`
- Requires TMA equivalent (DMA operations in ROCm)
- Complex memory access patterns

### Phase 4: Architecture-Specific Optimizations

#### AMD GPU Targets:
1. **CDNA2 (MI200 series - gfx90a)**
   - Full FP16 support
   - Matrix Core operations via MFMA
   - Priority: HIGH (datacenter focus)

2. **CDNA3 (MI300 series - gfx940/gfx941)**
   - FP8 support
   - Enhanced Matrix Core operations
   - Priority: HIGHEST (latest datacenter)

3. **RDNA3 (RX 7900 series - gfx1100)**
   - WMMA operations
   - Consumer GPU optimizations
   - Priority: MEDIUM

4. **RDNA2 (RX 6000 series - gfx1030)**
   - Basic support, no tensor cores
   - Priority: LOW

### Phase 5: Triton Kernels (Week 2, parallel)
- Triton kernels should work with minimal changes
- Verify ROCm Triton backend compatibility
- Test on both CDNA and RDNA architectures

### Phase 6: Python Integration (Week 5)
- Adapt `core.py` for ROCm backend detection
- Implement dynamic dispatch based on GPU architecture
- Maintain API compatibility with original SageAttention

### Phase 7: Testing & Validation (Week 5-6)
- Port existing tests to ROCm
- Create architecture-specific test suites
- Performance benchmarking vs CUDA implementation
- Accuracy validation

## Technical Challenges & Solutions

### 1. CUDA → HIP Conversion
**Challenge**: CUDA-specific intrinsics and memory operations
**Solution**:
- Use hipify-perl for initial conversion
- Manual optimization for performance-critical sections
- Reference existing ROCm ports (Flash Attention, xFormers)

### 2. WGMMA/TMA Instructions
**Challenge**: No direct ROCm equivalents
**Solution**:
- WGMMA → MFMA instructions on CDNA
- TMA → Async memory operations with HIP streams
- Consider Composable Kernel (CK) library for optimized implementations

### 3. FP8 Support
**Challenge**: Different FP8 formats between NVIDIA and AMD
**Solution**:
- Use ROCm's native FP8 support on MI300
- Implement conversion utilities if needed
- Fallback to FP16 for older architectures

### 4. Shared Memory Patterns
**Challenge**: Different shared memory architectures
**Solution**:
- Adapt permuted shared memory access patterns
- Use LDS (Local Data Share) optimally on AMD GPUs
- Profile and tune bank conflict avoidance

### 5. Build System
**Challenge**: Different compiler toolchains
**Solution**:
- Use hipcc instead of nvcc
- Adapt architecture detection for ROCm
- Support both cmake and setuptools builds

## Kernel Mapping Strategy

| CUDA Kernel | ROCm Implementation | Priority | Complexity |
|-------------|-------------------|----------|------------|
| qk_int_sv_f16_cuda_sm80 | qk_int_sv_f16_hip_gfx90a | HIGH | Medium |
| sm89_qk_int8_sv_f8_* (7 variants) | qk_int8_sv_f8_hip_gfx940 | HIGH | High |
| qk_int_sv_f8_cuda_sm90 | qk_int_sv_f8_hip_gfx940_tma | MEDIUM | Very High |
| fused operations | fused_hip | HIGH | Low |

## Dependencies
- ROCm 6.0+ (preferably 6.1 for FP8 support)
- hipBLAS
- rocWMMA (for WMMA operations)
- Composable Kernel (optional, for optimized implementations)
- PyTorch with ROCm support
- Triton with ROCm backend

## Performance Targets
- Achieve 80-90% of CUDA performance on equivalent hardware tiers
- Optimize for memory bandwidth utilization on AMD architectures
- Leverage AMD-specific features (larger register files, different cache hierarchies)

## Testing Matrix
| GPU | Architecture | Priority | Test Coverage |
|-----|--------------|----------|---------------|
| MI300X | gfx940/941 | Critical | Full |
| MI250X | gfx90a | High | Full |
| MI100 | gfx908 | Medium | Basic |
| RX 7900 XTX | gfx1100 | Medium | Triton-focused |
| RX 6900 XT | gfx1030 | Low | Minimal |

## Build Configuration
```python
# ROCm architectures to target
ROCM_ARCHS = {
    "gfx908": "MI100",
    "gfx90a": "MI200",
    "gfx940": "MI300A",
    "gfx941": "MI300X",
    "gfx1030": "RDNA2",
    "gfx1100": "RDNA3"
}

# Compiler flags
HIPCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-D__HIP_PLATFORM_AMD__",
    "-DROCM_VERSION=${ROCM_VERSION}",
    "--offload-arch=gfx90a",  # Multi-arch builds
    "--offload-arch=gfx940",
]
```

## Success Criteria
1. All Triton kernels working on ROCm
2. Core HIP kernels implemented for MI200/MI300
3. Performance within 20% of CUDA on comparable hardware
4. Pass all accuracy tests
5. Maintain API compatibility
6. Documentation and examples for ROCm usage

## Timeline Estimate
- **Total Duration**: 5-6 weeks for full implementation
- **MVP (Triton + Basic HIP)**: 2-3 weeks
- **Full Feature Parity**: 4-5 weeks
- **Optimization & Tuning**: Additional 1-2 weeks

## Notes
- Prioritize datacenter GPUs (CDNA) over consumer (RDNA)
- Consider using Composable Kernel library for complex operations
- Maintain separate code paths for different architectures
- Document ROCm-specific optimizations and limitations