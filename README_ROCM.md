# SageAttention ROCm Port

## Overview

This repository contains a comprehensive ROCm port of SageAttention, enabling AMD GPU acceleration for quantized attention mechanisms. The implementation provides multiple optimization paths including AOTriton support, Flash Attention, and INT8 quantization for AMD GPUs.

## 🚀 Key Features

- **Full ROCm 7 Support**: Native AMD GPU acceleration
- **Flash Attention**: Leverages PyTorch's built-in Flash Attention for AMD GPUs
- **AOTriton Integration**: Support for AMD's optimized Triton (when available)
- **INT8 Quantization**: Memory-efficient attention with INT8 quantization
- **Automatic Fallback**: CPU fallback for environments without GPU
- **Validated Accuracy**: < 3% error compared to FP32 baseline

## 📋 System Requirements

### Hardware Requirements
- **AMD GPU**:
  - MI100/MI200/MI300 series (datacenter)
  - Radeon RX 7900 XTX/XT (consumer)
  - Radeon Pro W7900/W7800 (workstation)
  - Any RDNA3 or CDNA2/3 architecture GPU

### Software Requirements
- **Operating System**: Windows 10/11 or Linux (Ubuntu 20.04/22.04)
- **ROCm**: Version 6.0+ (7.0+ recommended)
- **Python**: 3.8-3.12
- **PyTorch**: 2.0+ with ROCm support

## 🛠️ Installation

### Step 1: Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate
```

### Step 2: Install ROCm Drivers

#### Windows:
1. Download AMD ROCm drivers from [AMD's official site](https://www.amd.com/en/developer/rocm-hub.html)
2. Install the drivers and reboot
3. Verify installation:
```bash
rocminfo
```

#### Linux:
```bash
# Ubuntu/Debian
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.2.60202-1_all.deb
sudo apt install ./amdgpu-install_6.2.60202-1_all.deb
sudo amdgpu-install --usecase=rocm

# Verify
rocm-smi
```

### Step 3: Install PyTorch with ROCm Support

```bash
# For ROCm 6.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# For ROCm 7.0+ (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0
```

### Step 4: Install SageAttention ROCm

```bash
# Clone the repository
git clone https://github.com/ivandobskygithub/SageAttention_rocm.git
cd SageAttention_rocm

# Checkout ROCm branch
git checkout rocm-full-port

# Install the package
cd sageattention_rocm_full
pip install -e .
```

### Step 5: (Optional) Enable AOTriton

```bash
# Set environment variable for experimental AOTriton support
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  # Linux
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1     # Windows
```

## 🏃 Quick Start

### Verify Installation

```bash
cd sageattention_rocm_full
python validation/test_flash_attention.py
```

Expected output:
```
Flash Attention: Available
Memory Efficient: Available
```

### Run Benchmarks

```bash
# Windows
run_benchmarks.bat

# Linux/Mac
python bench/benchmark_final.py
```

### Use in Your Code

```python
import torch
import torch.nn.functional as F

# Option 1: Use PyTorch's native attention (recommended for AMD GPUs)
output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

# Option 2: Use SageAttention ROCm implementation
from sageattention_rocm import sageattn

# With automatic optimization selection
output = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)

# Option 3: Use optimized implementation directly
from sageattention_rocm.core_optimized import sageattn_optimized

# Without INT8 quantization (faster on most AMD GPUs)
output = sageattn_optimized(q, k, v, use_int8=False)

# With INT8 quantization (for memory-constrained scenarios)
output = sageattn_optimized(q, k, v, use_int8=True)
```

## 📂 Project Structure

```
SageAttention/
├── sageattention_rocm_full/     # Main ROCm implementation
│   ├── sageattention_rocm/      # Core library
│   │   ├── core_optimized.py    # Optimized Flash Attention implementation
│   │   ├── core_triton_aotriton.py  # AOTriton integration
│   │   └── triton/              # Triton kernels and fallbacks
│   ├── bench/                   # Benchmarking scripts
│   ├── validation/              # Validation and tests
│   └── docs/                    # Documentation
├── csrc/                        # Original CUDA C++ kernels
└── sageattention/              # Original CUDA implementation
```

## 🔄 ROCm-Specific Changes

### 1. Architecture Mapping
- **CUDA → HIP**: Core compute kernels ported using HIP
- **Tensor Cores → Matrix Cores**: WMMA/MFMA instructions for AMD GPUs
- **CUTLASS → rocBLAS/Composable Kernel**: Optimized GEMM operations

### 2. Optimization Strategy
- **Primary**: Leverage PyTorch's built-in Flash Attention (already optimized for AMD)
- **Secondary**: Custom Triton kernels for specific operations
- **Fallback**: Pure PyTorch implementations for compatibility

### 3. Key Implementations

| Component | CUDA Version | ROCm Version | Status |
|-----------|--------------|--------------|--------|
| Flash Attention | Custom CUDA kernels | PyTorch built-in | ✅ Working |
| INT8 Quantization | CUDA kernels | Triton/PyTorch | ✅ Working |
| Triton Kernels | CUDA Triton | AMD Triton/AOTriton | ⚠️ Partial |
| Memory Optimization | CUTLASS | Composable Kernel | ✅ Working |

## 📊 Performance Characteristics

### On AMD Radeon 8060S (94GB):
- **Flash Attention**: ✅ Native support, optimal performance
- **Memory Efficient Attention**: ✅ Available and working
- **INT8 Quantization**: ⚠️ Shows overhead (0.45x speed)
- **Recommendation**: Use native Flash Attention for best performance

### Performance Comparison:
| Method | Relative Speed | Memory Usage | Use Case |
|--------|---------------|--------------|----------|
| Flash Attention | 1.0x (baseline) | Optimal | Default choice |
| Memory Efficient | ~0.95x | Lower | Large sequences |
| INT8 Quantized | ~0.45x | 50% reduction | Memory-constrained |
| Math Backend | ~0.2x | High | Fallback only |

## 🧪 Validation Results

- **Correctness**: < 3% error vs FP32 baseline ✅
- **Numerical Stability**: Validated across multiple scales ✅
- **Causal/Non-causal**: Both modes working ✅
- **Tensor Layouts**: HND and NHD supported ✅

## 🐛 Known Issues & Limitations

### RDNA 4.0 Support
- **Status**: Testing blocked due to lack of native RDNA 4.0 support in current ROCm drivers
- **Impact**: Unable to optimize or validate performance on next-generation AMD GPUs
- **Timeline**: Pending AMD's official RDNA 4.0 ROCm driver release

### AOTriton
- AOTriton functions exist in PyTorch but may not work correctly
- Error: "This operator should be overridden in python"
- **Solution**: Implementation automatically falls back to Flash Attention

### INT8 Performance
- INT8 quantization shows overhead rather than speedup on current AMD architectures
- Measured 0.45x speed (slower) compared to native Flash Attention
- **Solution**: Use `use_int8=False` for better performance

### Windows-Specific
- Triton may not be available on Windows
- **Solution**: Automatic fallback to PyTorch implementations

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run validation tests
4. Submit a pull request

## 📚 Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [AMD Triton](https://github.com/ROCmSoftwarePlatform/triton)
- [Original SageAttention Paper](https://arxiv.org/abs/2410.02367)

## 📜 License

This project maintains the same license as the original SageAttention implementation.

## 🙏 Acknowledgments

- Original SageAttention authors for the innovative quantization approach
- AMD ROCm team for driver and runtime support
- PyTorch team for ROCm integration and Flash Attention
- Community contributors for testing and feedback

## 📞 Support

For issues specific to the ROCm port:
- Open an issue on [GitHub](https://github.com/ivandobskygithub/SageAttention_rocm/issues)
- Check existing issues for solutions
- Provide GPU model, ROCm version, and PyTorch version when reporting

---

**Note**: This ROCm port is optimized for AMD GPUs. For NVIDIA GPUs, use the original CUDA implementation.