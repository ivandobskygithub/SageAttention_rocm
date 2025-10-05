# SageAttention3 ROCm7 Windows Build Guide

## Quick Start

### Prerequisites

1. **ROCm 7** installed in Python environment:
   ```bash
   pip install rocm-sdk-runtime rocm-sdk-compiler
   ```

2. **Visual Studio Build Tools** (for Windows SDK):
   - Windows 10 SDK (10.0.26100.0 or later)
   - C++ build tools

3. **PyTorch with ROCm** (optional, for Python bindings):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
   ```

### Build the DLL

```bash
cd sageattention3_rocm7
python build_dll.py
```

**Output**: `sage_attention_rocm7.dll` (approx. 162 KB)

### Test the DLL

```bash
python test_dll_with_deps.py
```

Expected output:
```
DLL LOADING TEST PASSED
✓ launch_sage_attention_forward
✓ launch_quantize_int4
✓ launch_dequantize_int4
✓ launch_quantize_int4_transpose
```

## Build Script Details

### What `build_dll.py` Does

1. **Compiles HIP source files** to object files (.o):
   - `hip/attention_forward_simple.hip.cpp`
   - `hip/quantization/int4_ops.hip.cpp`

2. **Links object files** into Windows DLL:
   - Adds Windows SDK library paths
   - Links against `amdhip64_7.dll`
   - Exports C functions with proper calling convention

3. **Verifies exports** using `dumpbin.exe`

### Key Build Flags

#### Compilation
```bash
--offload-arch=gfx1151      # Target RDNA3.5
-O3                         # Optimize
-std=c++17                  # C++ standard
-D__HIP_PLATFORM_AMD__      # Platform define
-DNOMINMAX                  # Windows compatibility
-D_CRT_SECURE_NO_WARNINGS   # Suppress CRT warnings
```

#### Linking
```bash
-shared                     # Build DLL
-Xlinker "/LIBPATH:..."    # Windows SDK paths (QUOTED!)
-Xlinker "/DEFAULTLIB:amdhip64.lib"   # HIP runtime
-Xlinker "/DEFAULTLIB:kernel32.lib"   # Windows API
```

### Common Issues & Solutions

#### Issue 1: "unsupported option '-fPIC'"
**Cause**: `-fPIC` is Linux-specific, not supported on Windows MSVC target

**Solution**: Remove `-fPIC` from compile flags (already done in `build_dll.py`)

#### Issue 2: "could not open 'kernel32.lib'"
**Cause**: Windows SDK library path not in linker search path

**Solution**: Use `-Xlinker "/LIBPATH:C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"`

**Important**: Quote the entire `/LIBPATH:...` argument to prevent path splitting on spaces

#### Issue 3: DLL won't load - "Could not find module"
**Cause**: `amdhip64_7.dll` not in DLL search path

**Solution**:
```python
import os
rocm_bin = r".venv\Lib\site-packages\_rocm_sdk_core\bin"
os.add_dll_directory(rocm_bin)  # Python 3.8+
```

#### Issue 4: "rocwmma/rocwmma.hpp not found"
**Cause**: rocWMMA not included in Windows ROCm SDK

**Solution**: Use simplified kernel (`attention_forward_simple.hip.cpp`) that doesn't require rocWMMA

## DLL API Reference

### C Function Signatures

```cpp
// Launch attention forward pass
extern "C" __declspec(dllexport) void launch_sage_attention_forward(
    const half* Q,              // Query tensor
    const half* K,              // Key tensor
    const half* V,              // Value tensor
    half* O,                    // Output tensor
    const float* delta_s,       // Mean subtraction (optional)
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    float scale,                // Softmax scale (1/sqrt(head_dim))
    bool is_causal,             // Apply causal mask?
    hipStream_t stream          // HIP stream
);

// Quantize tensor to INT4
extern "C" __declspec(dllexport) void launch_quantize_int4(
    const half* input,
    uint8_t* output,           // Packed INT4 (2 values per byte)
    half* scales,              // Per-block scale factors
    int batch_size,
    int num_tokens,
    int num_heads,
    int head_dim,
    // ... stride parameters ...
    hipStream_t stream
);

// Dequantize INT4 back to FP16
extern "C" __declspec(dllexport) void launch_dequantize_int4(
    const uint8_t* input,      // Packed INT4
    const half* scales,
    half* output,
    int batch_size,
    int num_tokens,
    int num_heads,
    int head_dim,
    // ... stride parameters ...
    hipStream_t stream
);

// Quantize with transpose (for K matrix)
extern "C" __declspec(dllexport) void launch_quantize_int4_transpose(
    const half* input,
    uint8_t* output,           // Transposed and packed
    half* scales,
    int batch_size,
    int num_tokens,
    int num_heads,
    int head_dim,
    // ... stride parameters ...
    hipStream_t stream
);
```

## Python Integration

### Loading with ctypes

```python
import ctypes
import os
from pathlib import Path

# Setup DLL search path
rocm_bin = Path(".venv/Lib/site-packages/_rocm_sdk_core/bin")
os.add_dll_directory(str(rocm_bin))

# Load DLL
dll_path = Path("sage_attention_rocm7.dll")
lib = ctypes.CDLL(str(dll_path))

# Access functions
launch_attention = lib.launch_sage_attention_forward
```

### PyTorch Binding (TODO)

Create `sage_attention_rocm7/__init__.py`:

```python
import torch
import ctypes
import os

# Load DLL with proper setup
_lib = None

def _load_dll():
    global _lib
    if _lib is None:
        # Add ROCm to PATH
        rocm_bin = Path(__file__).parent.parent / ".venv" / "Lib" / "site-packages" / "_rocm_sdk_core" / "bin"
        os.add_dll_directory(str(rocm_bin))

        # Load DLL
        dll_path = Path(__file__).parent / "sage_attention_rocm7.dll"
        _lib = ctypes.CDLL(str(dll_path))
    return _lib

def sage_attention_forward(Q, K, V, scale=None, is_causal=False):
    """
    Compute attention: softmax(Q @ K^T / scale) @ V

    Args:
        Q: Query tensor [batch, num_heads, seq_len_q, head_dim]
        K: Key tensor [batch, num_heads, seq_len_k, head_dim]
        V: Value tensor [batch, num_heads, seq_len_k, head_dim]
        scale: Softmax scale (default: 1/sqrt(head_dim))
        is_causal: Apply causal mask (default: False)

    Returns:
        Output tensor [batch, num_heads, seq_len_q, head_dim]
    """
    lib = _load_dll()

    # Validate inputs
    assert Q.dtype == torch.float16, "Q must be FP16"
    assert K.dtype == torch.float16, "K must be FP16"
    assert V.dtype == torch.float16, "V must be FP16"
    assert Q.is_contiguous(), "Q must be contiguous"
    assert K.is_contiguous(), "K must be contiguous"
    assert V.is_contiguous(), "V must be contiguous"

    # Get dimensions
    batch, num_heads, seq_len_q, head_dim = Q.shape
    _, _, seq_len_k, _ = K.shape

    # Default scale
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # Allocate output
    O = torch.empty_like(Q)

    # Get HIP stream
    stream = torch.hip.current_stream().hip_stream

    # Call kernel
    lib.launch_sage_attention_forward(
        ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(K.data_ptr()),
        ctypes.c_void_p(V.data_ptr()),
        ctypes.c_void_p(O.data_ptr()),
        None,  # delta_s (optional)
        batch,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        ctypes.c_float(scale),
        ctypes.c_bool(is_causal),
        stream
    )

    return O
```

## Performance Notes

### Current Implementation

- **Kernel**: Simplified thread-level implementation
- **No rocWMMA**: Manual matrix operations
- **No rocBLAS**: Direct GEMM computation
- **INT4 Quantization**: Block-scaled symmetric quantization

### Expected Performance

This is a **functional baseline**. Performance is not yet optimized.

**Estimated vs. PyTorch native attention**:
- Current: 0.5-0.8x (slower due to non-optimized GEMM)
- With rocBLAS: 1.5-2x (faster)
- With full optimization: 2-3x (faster, matching CUDA version)

### Optimization Roadmap

1. **Add rocBLAS calls** for Q@K^T and S@V
2. **Optimize shared memory usage** (tile sizes)
3. **Implement wave-level reductions** for softmax
4. **Add async memory copies** (if applicable)
5. **Benchmark and profile** using rocprof

## Troubleshooting

### Build Failures

**Q: Compilation fails with "rocwmma/rocwmma.hpp not found"**

A: You're using the wrong source file. Use `attention_forward_simple.hip.cpp`, not `attention_forward.hip.cpp`.

**Q: Linking fails with "could not open 'xxx.lib'"**

A: Check that `find_windows_sdk()` in `build_dll.py` finds your Windows SDK. Update the SDK path if necessary.

**Q: Object files are empty or not created**

A: Check compilation output for errors. The build script now properly detects this.

### Runtime Failures

**Q: DLL won't load - "Could not find module"**

A: Ensure you call `os.add_dll_directory()` before loading:
```python
os.add_dll_directory(".venv/Lib/site-packages/_rocm_sdk_core/bin")
```

**Q: Kernel launches but crashes**

A: Check tensor types, shapes, and contiguity:
```python
assert Q.dtype == torch.float16
assert Q.is_contiguous()
```

**Q: Incorrect results**

A: Verify tensor layout is `[batch, num_heads, seq_len, head_dim]`. If using PyTorch's native layout, you may need to transpose.

## Directory Structure

```
sageattention3_rocm7/
├── hip/
│   ├── attention_forward_simple.hip.cpp   # Working kernel ✅
│   ├── attention_forward.hip.cpp          # Original (needs rocWMMA) ⚠️
│   ├── quantization/
│   │   └── int4_ops.hip.cpp              # Quantization kernels ✅
│   └── include/
│       └── common.h                       # Shared definitions
│
├── build_dll.py                           # Build script ✅
├── test_dll_with_deps.py                 # Test DLL loading ✅
├── check_dll_deps.py                     # Check dependencies
│
├── sage_attention_rocm7.dll              # Output DLL ✅
│
├── BUILD_SUMMARY.md                       # Build report
├── WINDOWS_BUILD_GUIDE.md                # This file
│
└── sage_attention_rocm7/                 # Python package (TODO)
    ├── __init__.py                        # PyTorch wrapper
    └── binding.cpp                        # C++ extension (optional)
```

## Contributing

### Testing Changes

After modifying kernels:

1. Rebuild:
   ```bash
   python build_dll.py
   ```

2. Test loading:
   ```bash
   python test_dll_with_deps.py
   ```

3. Test with PyTorch (once bindings are ready):
   ```bash
   python -m pytest tests/
   ```

### Code Style

- **HIP/CUDA style**: Follow existing kernel patterns
- **Python**: PEP 8
- **C++**: Kernel argument docs, clear variable names
- **Comments**: Explain *why*, not *what*

## License

Same as parent SageAttention project (Apache 2.0)

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [SageAttention Paper](https://arxiv.org/abs/2410.02367)
- [Original CUDA Implementation](https://github.com/thu-ml/SageAttention)

---

**Last Updated**: October 5, 2025
**Build Status**: ✅ Working
**Target GPU**: AMD Radeon 890M (gfx1151)
