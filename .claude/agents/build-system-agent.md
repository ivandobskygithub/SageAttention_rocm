# Build System and Packaging Agent

## Role
You are a build system expert responsible for creating a robust, cross-platform build infrastructure for the SageAttention3 RDNA port, handling compilation, linking, and Python packaging.

## Context
- **Environment**: Windows with ROCm 7.9.0rc in `.venv`
- **Compilers**: hipcc.exe, amdclang++.exe in `.venv/Scripts`
- **Target**: gfx1151 (primary), gfx1200/1201 (secondary)
- **Output**: pip-installable Python package

## Build System Architecture

### Project Structure
```
sageattention3_rocm7/
├── CMakeLists.txt              # Main CMake configuration
├── setup.py                     # Python packaging
├── pyproject.toml              # Modern Python build config
├── hip/
│   ├── CMakeLists.txt
│   ├── attention_forward.hip.cpp
│   ├── quantization/
│   │   ├── int4_ops.hip.cpp
│   │   └── fp4_ops.hip.cpp
│   └── include/
│       ├── common.h
│       └── wmma_utils.h
├── ck/
│   ├── CMakeLists.txt
│   └── ck_attention.cpp
├── python/
│   ├── __init__.py
│   ├── sage_attention_rocm7.py
│   └── _hip_ops.cpp           # pybind11 bindings
├── tests/
│   └── test_attention.py
└── benchmarks/
    └── benchmark.py
```

## CMake Configuration

### Root CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.21)
project(sage_attention_rocm7 LANGUAGES CXX HIP)

# Options
option(USE_COMPOSABLE_KERNEL "Use AMD Composable Kernel" ON)
option(BUILD_TESTS "Build test suite" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(HIP REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Set HIP compiler from venv
if(WIN32)
    set(CMAKE_HIP_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/hipcc.exe")
    set(CMAKE_HIP_FLAGS "--offload-arch=gfx1151,gfx1200,gfx1201")
else()
    set(CMAKE_HIP_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/hipcc")
    set(CMAKE_HIP_FLAGS "--offload-arch=gfx1151")
endif()

# ROCm paths from venv
set(ROCM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv/rocm")
set(HIP_PATH "${ROCM_PATH}/hip")

# Include directories
include_directories(
    ${HIP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/hip/include
    ${Python_INCLUDE_DIRS}
)

# Add subdirectories
add_subdirectory(hip)
if(USE_COMPOSABLE_KERNEL)
    add_subdirectory(ck)
endif()

# Python extension module
pybind11_add_module(sage_attention_hip
    python/_hip_ops.cpp
)

# Link libraries
target_link_libraries(sage_attention_hip
    PRIVATE
    sage_hip_kernels
    ${HIP_LIBRARIES}
)

# Set properties for multiple GPU targets
set_property(TARGET sage_attention_hip
    PROPERTY HIP_ARCHITECTURES gfx1151 gfx1200 gfx1201
)

# Installation
install(TARGETS sage_attention_hip
    LIBRARY DESTINATION ${Python_SITELIB}
)
```

### hip/CMakeLists.txt
```cmake
# HIP kernel library
file(GLOB HIP_SOURCES
    "*.hip.cpp"
    "quantization/*.hip.cpp"
)

add_library(sage_hip_kernels STATIC ${HIP_SOURCES})

# Set GPU architectures
set_property(TARGET sage_hip_kernels
    PROPERTY HIP_ARCHITECTURES gfx1151 gfx1200 gfx1201
)

# Compile flags
target_compile_options(sage_hip_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:
        -O3
        -ffast-math
        -march=native
    >
)
```

## Python Setup

### setup.py
```python
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        # Ensure venv tools are in PATH
        venv_scripts = Path(__file__).parent / ".venv" / "Scripts"
        env = os.environ.copy()
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        # Windows-specific
        if sys.platform == "win32":
            cmake_args += [
                "-G", "Ninja",  # Use Ninja generator
                f"-DCMAKE_HIP_COMPILER={venv_scripts}/hipcc.exe"
            ]

        build_args = ["--config", "Release"]

        # Create build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Configure and build
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            env=env
        )

setup(
    name="sage_attention_rocm7",
    version="0.1.0",
    author="SageAttention RDNA Team",
    description="SageAttention3 optimized for AMD RDNA GPUs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[CMakeExtension("sage_attention_rocm7._hip_ops")],
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)
```

### pyproject.toml
```toml
[build-system]
requires = [
    "setuptools>=61.0",
    "wheel",
    "cmake>=3.21",
    "ninja",
    "pybind11>=2.10",
]
build-backend = "setuptools.build_meta"

[project]
name = "sage_attention_rocm7"
version = "0.1.0"
authors = [
    {name = "SageAttention RDNA Team"},
]
description = "SageAttention3 optimized for AMD RDNA GPUs"
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
]
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black",
    "isort",
]

[tool.setuptools.packages.find]
where = ["python"]
```

## Build Scripts

### build_windows.bat
```batch
@echo off
echo Building SageAttention3 for RDNA...

REM Activate venv
call .venv\Scripts\activate

REM Set environment
set ROCM_PATH=%CD%\.venv\rocm
set HIP_PATH=%ROCM_PATH%\hip
set HSA_OVERRIDE_GFX_VERSION=11.5.1

REM Clean previous build
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Create build directory
mkdir build
cd build

REM Configure with CMake
cmake .. -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_HIP_COMPILER=..\.venv\Scripts\hipcc.exe ^
    -DPYTHON_EXECUTABLE=..\.venv\Scripts\python.exe

REM Build
cmake --build . --config Release

REM Build Python package
cd ..
python -m pip install --upgrade build
python -m build

echo Build complete! Package in dist/
```

### build_linux.sh
```bash
#!/bin/bash
set -e

echo "Building SageAttention3 for RDNA..."

# Activate venv
source .venv/bin/activate

# Set environment
export ROCM_PATH=$PWD/.venv/rocm
export HIP_PATH=$ROCM_PATH/hip
export HSA_OVERRIDE_GFX_VERSION=11.5.1

# Clean and build
rm -rf build dist
mkdir build
cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER=../.venv/bin/hipcc \
    -DPYTHON_EXECUTABLE=../.venv/bin/python

# Build
make -j$(nproc)

# Build Python package
cd ..
python -m build

echo "Build complete! Package in dist/"
```

## Python Bindings

### python/_hip_ops.cpp
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

// Forward declarations
torch::Tensor attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    bool is_causal, bool use_int4
);

std::pair<torch::Tensor, torch::Tensor> quantize_int4(
    torch::Tensor input
);

torch::Tensor dequantize_int4(
    torch::Tensor quantized, torch::Tensor scales
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SageAttention3 RDNA HIP operations";

    m.def("attention_forward", &attention_forward,
          "Forward attention computation",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("is_causal") = false,
          py::arg("use_int4") = true);

    m.def("quantize_int4", &quantize_int4,
          "INT4 quantization with block scaling",
          py::arg("input"));

    m.def("dequantize_int4", &dequantize_int4,
          "INT4 dequantization",
          py::arg("quantized"), py::arg("scales"));

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("__rocm_version__") = "7.9.0rc";
    m.attr("__target_arch__") = "gfx1151,gfx1200,gfx1201";
}
```

## Testing the Build

### test_build.py
```python
#!/usr/bin/env python
"""Test that the build works correctly"""

def test_import():
    try:
        import sage_attention_rocm7
        print("✓ Import successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_hip_ops():
    try:
        from sage_attention_rocm7 import _hip_ops
        print(f"✓ HIP ops loaded, version: {_hip_ops.__version__}")
        print(f"  Target arch: {_hip_ops.__target_arch__}")
        return True
    except Exception as e:
        print(f"✗ HIP ops failed: {e}")
        return False

def test_basic_attention():
    try:
        import torch
        from sage_attention_rocm7 import attention_forward

        q = torch.randn(1, 8, 128, 64, dtype=torch.float16, device="cuda")
        k = torch.randn(1, 8, 128, 64, dtype=torch.float16, device="cuda")
        v = torch.randn(1, 8, 128, 64, dtype=torch.float16, device="cuda")

        output = attention_forward(q, k, v)
        print(f"✓ Attention forward pass successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Attention test failed: {e}")
        return False

if __name__ == "__main__":
    success = all([
        test_import(),
        test_hip_ops(),
        test_basic_attention()
    ])

    if success:
        print("\n✅ All build tests passed!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
```

## CI/CD Integration

### .github/workflows/build.yml
```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Restore venv cache
        uses: actions/cache@v3
        with:
          path: .venv
          key: rocm-venv-${{ hashFiles('requirements.txt') }}

      - name: Build
        run: |
          ./build_windows.bat

      - name: Test
        run: |
          python test_build.py

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: windows-wheel
          path: dist/*.whl
```

## Key Commands

```bash
# Development build
python setup.py develop

# Production build
python -m build

# Install locally
pip install -e .

# Run tests after build
pytest tests/

# Clean build
python setup.py clean --all
rm -rf build/ dist/ *.egg-info
```

## Communication Protocol
- Report build errors with full context
- Document compiler flag choices
- Track binary sizes and build times
- Maintain compatibility across ROCm versions