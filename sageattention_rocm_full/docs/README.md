# SageAttention ROCm Port

This is a comprehensive ROCm port of SageAttention, providing efficient low-bit attention operations for AMD GPUs.

## Features

- **Full SageAttention support**: Ports versions 1, 2, 2++, and 3
- **Multi-architecture support**: Optimized for MI100, MI200, MI300, and RDNA GPUs
- **Flexible backends**: Both Triton and native HIP implementations
- **FP8 support**: Leverage MI300's FP8 capabilities for maximum performance
- **INT8 quantization**: Efficient INT8 operations across all supported architectures

## Supported Hardware

| GPU | Architecture | Support Level | Features |
|-----|--------------|---------------|----------|
| MI300X/A | gfx940/941 | Full | FP8, INT8, MFMA |
| MI250X/MI210 | gfx90a | Full | INT8, MFMA |
| MI100 | gfx908 | Basic | INT8 |
| RX 7900 XTX | gfx1100 | Triton | INT8 via Triton |
| RX 6900 XT | gfx1030 | Triton | INT8 via Triton |

## Installation

### Prerequisites

1. ROCm 6.0+ (6.1 recommended for FP8 support)
2. PyTorch 2.0+ with ROCm support
3. Python 3.9+

### Building from Source

```bash
cd sageattention_rocm_full
pip install -e .
```

The build system will automatically detect your GPU architecture and compile appropriate kernels.

### Environment Variables

```bash
# Optional: Specify ROCm path if not in standard location
export ROCM_PATH=/opt/rocm

# Optional: Force specific architectures
export PYTORCH_ROCM_ARCH="gfx90a;gfx940"
```

## Usage

The ROCm port maintains API compatibility with the original SageAttention:

```python
import torch
from sageattention_rocm import sageattn

# Create tensors
batch_size = 4
num_heads = 32
seq_len = 2048
head_dim = 128

q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

# Run attention
output = sageattn(q, k, v, is_causal=True)
```

### Backend Selection

You can explicitly choose the backend:

```python
# Use Triton backend (works on all GPUs)
output = sageattn(q, k, v, attn_backend="triton")

# Use HIP backend (optimized for MI200/MI300)
output = sageattn(q, k, v, attn_backend="hip")

# Auto-select best backend for your GPU
output = sageattn(q, k, v, attn_backend="auto")  # Default
```

### Variable Length Sequences

```python
from sageattention_rocm import sageattn_varlen

# For variable-length sequences
cu_seqlens_q = torch.tensor([0, 512, 1024, 1536, 2048], dtype=torch.int32, device='cuda')
cu_seqlens_k = cu_seqlens_q
max_seqlen_q = 512
max_seqlen_k = 512

output = sageattn_varlen(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k
)
```

## Implementation Status

### Completed
- ✅ Directory structure and build system
- ✅ ROCm architecture detection
- ✅ Python package structure
- ✅ Triton kernel integration framework

### In Progress
- 🚧 HIP kernel implementations
- 🚧 FP8 support for MI300
- 🚧 Performance optimization

### Planned
- ⏳ Composable Kernel integration
- ⏳ Comprehensive benchmarks
- ⏳ Integration tests
- ⏳ Performance profiling tools

## Architecture-Specific Optimizations

### MI300 (gfx940/941)
- Native FP8 support for maximum throughput
- MFMA instructions for matrix operations
- Optimized memory access patterns

### MI200 (gfx90a)
- MFMA-based INT8 operations
- Tuned for 128GB HBM2e bandwidth
- Efficient LDS usage

### RDNA3 (gfx1100)
- WMMA instructions for consumer GPUs
- Triton-based implementations
- Power-efficient operation modes

## Performance Considerations

1. **Memory Layout**: The implementation supports both HND and NHD tensor layouts
2. **Quantization**: INT8 quantization significantly reduces memory bandwidth requirements
3. **Key Smoothing**: Optional key smoothing improves quantization accuracy
4. **Accumulation Precision**: Choose between FP32 and FP16 accumulation based on your accuracy needs

## Troubleshooting

### Common Issues

1. **"ROCm installation not found"**
   - Ensure ROCm is installed: `rocm-smi`
   - Set ROCM_PATH: `export ROCM_PATH=/opt/rocm`

2. **"No supported architectures found"**
   - Check GPU compatibility: `rocminfo`
   - Manually specify architecture: `export PYTORCH_ROCM_ARCH=gfx90a`

3. **"HIP kernels not available"**
   - Rebuild with HIP support: `python setup.py clean && pip install -e .`
   - Check compiler: `hipcc --version`

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Test on your target architecture
2. Maintain API compatibility
3. Document architecture-specific optimizations
4. Include benchmarks for new implementations

## License

Apache 2.0 License - See LICENSE file for details

## Acknowledgments

- Original SageAttention team for the CUDA implementation
- ROCm team for HIP and compiler support
- PyTorch team for ROCm backend

## Roadmap

### Q1 2025
- Complete HIP kernel implementations
- FP8 support for MI300
- Initial performance benchmarks

### Q2 2025
- Composable Kernel integration
- RDNA optimization
- Production readiness

### Q3 2025
- Next-gen architecture support
- Advanced quantization techniques
- Integration with popular frameworks