# ROCm/HIP Implementation of SageAttention Kernels

This directory contains the ROCm/HIP equivalents of the CUDA kernels from `csrc/`.

## Strategy

Based on the analysis, we're taking a **Triton-first approach** with selective HIP porting:

1. **Primary Backend: Triton** - Handles 90% of functionality portably
2. **HIP Extensions: Fused Ops** - Port only essential preprocessing kernels
3. **Skip Complex CUDA Kernels** - No direct port of tensor core kernels

## Directory Structure

```
csrc_rocm/
├── README.md                   # This file
├── utils/                      # Utility headers (mostly reusable)
│   ├── hip_utils.h            # HIP equivalents of CUDA utilities
│   ├── dispatch_utils.h       # Reused from CUDA (portable)
│   ├── reduction_utils.hip.h  # HIP reduction primitives
│   └── math.hip.h             # HIP math intrinsics
├── fused/                      # Fused operations (HIP port)
│   ├── fused.hip              # HIP implementation
│   ├── fused.h                # Interface (reusable)
│   └── pybind.cpp             # Python bindings
└── qattn/                      # Attention kernels
    └── README.md              # Explains why we use Triton instead
```

## What We're NOT Porting

These CUDA components are too complex and Triton handles them better:

- `mma.cuh` - 723 lines of PTX tensor core assembly → Use Triton's `tl.dot()`
- `wgmma.cuh` - Hopper-specific, no AMD equivalent → Skip
- `cp_async.cuh` - NVIDIA async copy → Triton handles automatically
- `permuted_smem.cuh` - Bank conflict patterns → Triton handles automatically
- `qattn/*.cu` kernels - 1000+ lines each → Use Triton kernels

## What We ARE Porting

### 1. Utility Headers (Easy)
- Basic utilities that are mostly portable
- Reduction operations with HIP shuffle intrinsics
- Math functions using HIP equivalents

### 2. Fused Operations (Medium Priority)
These preprocessing kernels are simpler and worth porting:
- `quant_per_block_int8` - Block-wise INT8 quantization
- `sub_mean` - Mean subtraction
- `transpose_pad_permute` - Layout transformations

## Build Instructions

```bash
# For HIP compilation
export ROCM_PATH=/opt/rocm
hipcc -O3 -std=c++17 -fPIC -D__HIP_PLATFORM_AMD__ \
      --offload-arch=gfx90a,gfx940 \
      -c fused/fused.hip -o fused.o
```

## Performance Expectations

- **Triton kernels**: 80-90% of hand-tuned CUDA performance
- **HIP fused ops**: Should match CUDA performance
- **Overall**: Acceptable performance with 10x less development effort