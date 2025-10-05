# SageAttention3 ROCm7 Build Guide

This document describes the build system for the SageAttention3 ROCm7 port targeting AMD RDNA3.5 (gfx1151) GPUs.

## Prerequisites

### Required Software
- **ROCm 7.0+** (tested with 7.9.0rc on Windows)
- **Python 3.8+** (tested with 3.13.5)
- **PyTorch with ROCm support** (ROCm build)
- **CMake 3.21+** (optional, for standalone builds)
- **Visual Studio 2019+** (Windows only, for C++ compiler)

### ROCm Components
The following ROCm components must be installed:
- HIP runtime (`hipcc`, `amdclang++`)
- rocBLAS library
- rocWMMA library
- ROCm device libraries

### Python Packages
```bash
pip install torch numpy packaging wheel setuptools
```

## Build Methods

### Method 1: Python Setup (Recommended)

This method uses PyTorch's extension system and is the easiest for Python integration.

#### Quick Build
```cmd
# Activate virtual environment (if using one)
.venv\Scripts\activate

# Build in-place
python setup.py build_ext --inplace

# Or install to site-packages
python setup.py install
```

#### Using the Build Script (Windows)
```cmd
# Release build (optimized)
build.bat

# Debug build (with debug symbols)
build.bat debug

# Clean and rebuild
build.bat clean release

# Build and install
build.bat install

# Build and run tests
build.bat test
```

#### Build Script Options
- `clean` - Remove all build artifacts
- `debug` - Build with debug symbols and `-O0`
- `release` - Build optimized (default)
- `install` - Install Python package after building
- `test` - Run tests after building

### Method 2: CMake Build (Standalone)

This method builds a standalone shared library without Python integration.

```cmd
# Create build directory
mkdir build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1151

# Build
cmake --build . --config Release

# Install (optional)
cmake --install . --prefix ../install
```

#### CMake Options
- `BUILD_SHARED_LIBS` - Build shared libraries (default: ON)
- `BUILD_TESTS` - Build C++ test executables (default: ON)
- `BUILD_PYTHON_MODULE` - Enable PyTorch integration (default: OFF)
- `SAGE_ATTN_DEBUG` - Enable debug output (default: OFF)
- `GPU_TARGETS` - Target GPU architecture (default: gfx1151)
- `ROCM_PATH` - Path to ROCm installation (default: C:\Program Files\AMD\ROCm\7.0)

Example:
```cmd
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSAGE_ATTN_DEBUG=ON -DBUILD_TESTS=ON
```

## Build System Architecture

### File Structure
```
sageattention3_rocm7/
├── CMakeLists.txt           # CMake build configuration
├── setup.py                 # Python package setup
├── build.bat                # Windows build script
├── BUILD.md                 # This file
│
├── hip/                     # HIP kernel sources
│   ├── attention_forward.hip.cpp
│   ├── quantization/
│   │   └── int4_ops.hip.cpp
│   └── include/
│       ├── common.h
│       └── ...
│
├── sage_attention_rocm7/    # Python package
│   ├── __init__.py          # Package initialization
│   ├── ops.py               # High-level Python API
│   └── binding.cpp          # PyTorch C++ bindings
│
└── tests/                   # Test files
    └── ...
```

### Build Process

1. **setup.py** detects ROCm installation and hipcc compiler
2. **BuildHIPExtension** custom build class compiles HIP sources
3. **hipcc** compiles `.hip.cpp` files to object files
4. Object files are linked with ROCm libraries and PyTorch
5. Shared library is created as Python extension module (`_C.so` or `_C.pyd`)

### Compiler Flags

#### HIP Compilation
```
--offload-arch=gfx1151    # Target RDNA3.5 architecture
-O3                        # Optimization level (or -O0 -g for debug)
-fPIC                      # Position-independent code
-std=c++17                 # C++17 standard
-ffast-math                # Fast math optimizations
-munsafe-fp-atomics        # Allow unsafe FP atomics (for RDNA)
-D__HIP_PLATFORM_AMD__=1   # AMD platform
-DWARP_SIZE=32             # RDNA wave size
```

#### Windows-Specific
```
-D_CRT_SECURE_NO_WARNINGS  # Disable CRT security warnings
-DNOMINMAX                 # Prevent min/max macro conflicts
```

### Linking

The compiled kernels are linked against:
- **HIP runtime**: `amdhip64.lib` (Windows) or `libhip_hcc.so` (Linux)
- **rocBLAS**: `rocblas.lib`
- **PyTorch**: `torch.lib`, `torch_python.lib`, `c10.lib`

## Troubleshooting

### Common Issues

#### 1. "hipcc not found"
**Solution**: Ensure ROCm is installed and `ROCM_PATH` is set correctly.
```cmd
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.0
```

#### 2. "PyTorch not found"
**Solution**: Install PyTorch with ROCm support.
```cmd
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

#### 3. "rocWMMA headers not found"
**Solution**: Ensure rocWMMA is included in your ROCm installation.
Check: `%ROCM_PATH%\include\rocwmma\rocwmma.hpp`

#### 4. Windows SDK library errors
**Workaround**: The build system includes `-D_CRT_SECURE_NO_WARNINGS` and `-DNOMINMAX` to work around common Windows issues.

#### 5. Link errors with PyTorch
**Solution**: Ensure PyTorch is installed in the same Python environment.
```cmd
python -c "import torch; print(torch.__version__)"
```

### Debug Build

For debugging compilation issues:
```cmd
# Enable verbose output
set VERBOSE=1

# Build with debug symbols
build.bat clean debug

# Or with setup.py
set SAGE_DEBUG_BUILD=TRUE
python setup.py build_ext --inplace --verbose
```

### Checking Build Output

After building, verify the extension loaded:
```python
import sage_attention_rocm7
print(sage_attention_rocm7.get_build_info())
print(sage_attention_rocm7.check_compatibility())
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROCM_PATH` | Path to ROCm installation | `C:\Program Files\AMD\ROCm\7.0` |
| `GPU_TARGETS` | Target GPU architecture | `gfx1151` |
| `SAGE_DEBUG_BUILD` | Enable debug build | `FALSE` |
| `SAGE_FORCE_BUILD` | Force rebuild from scratch | `FALSE` |

## Performance Notes

### Optimization Levels
- **Release** (`-O3`): ~3-5x faster than debug, recommended for production
- **Debug** (`-O0 -g`): Easier debugging, slower execution

### GPU-Specific Tuning
The build system is optimized for RDNA3.5 (gfx1151):
- Wave size: 32
- LDS (shared memory): 32KB per CU
- WMMA tile sizes: 16x16x16

For other RDNA GPUs, adjust `GPU_TARGETS`:
- RDNA2: `gfx1030`, `gfx1031`, `gfx1032`
- RDNA3: `gfx1100`, `gfx1101`, `gfx1102`
- RDNA3.5: `gfx1150`, `gfx1151`

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Install ROCm
  run: |
    # Install ROCm 7.0
    wget https://repo.radeon.com/rocm/...

- name: Build Extension
  run: |
    export ROCM_PATH=/opt/rocm
    python setup.py build_ext --inplace

- name: Test
  run: |
    python -m pytest tests/
```

### Docker Build
```dockerfile
FROM rocm/dev-ubuntu-22.04:7.0

RUN pip install torch numpy

COPY . /workspace
WORKDIR /workspace

RUN python setup.py install
```

## License

This build system is part of the SageAttention3 ROCm port project.
See LICENSE file for details.
