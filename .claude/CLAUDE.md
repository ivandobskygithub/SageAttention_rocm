# SageAttention ROCm Port Project

## Project Overview
This repo contains the CUDA implementation of SageAttention (versions 1, 2, 2++, and 3). The goal is to port **SageAttention3 Blackwell** (`sageattention3_blackwell/`) to AMD ROCm 7.

## Current Architecture
- **Target for porting**: `sageattention3_blackwell/` - FP4 microscaling attention optimized for NVIDIA Blackwell (SM 12.0a)
- **CUDA requirements**: CUDA ≥12.8, uses CUTLASS library, targets `arch=compute_120a,code=sm_120a`
- **Key kernels**: Located in `sageattention3_blackwell/sageattn3/blackwell/` - includes TMA (Tensor Memory Accelerator), warpgroup specialization, FP4 quantization

## ROCm 7 Port Strategy
- Convert CUDA kernels to HIP
- Replace NVIDIA-specific features (TMA, WGMMA) with ROCm equivalents (composable_kernel or hipBLAS)
- Target AMD RDNA3/CDNA3 architectures
- Maintain FP4 quantization approach for inference acceleration

## Build System
- Current: PyTorch C++/CUDA extensions with `setup.py`
- Port will need: HIP/ROCm equivalents, hipify for automated conversion where applicable

## Key Technical Components
- FP4 quantization (`fp4_quantization_4d.cu`)
- Custom CUTLASS-based attention kernels (`api.cu`, `mainloop_tma_ws.h`, `kernel_ws.h`)
- Warpgroup specialization and tensor memory operations
- Block-scaled layouts for FP4 data

## Context7 Usage
Use context7 MCP to reference ROCm 7, HIP, and composable_kernel documentation when implementing the port.
