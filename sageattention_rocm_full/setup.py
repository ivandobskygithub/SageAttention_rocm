"""
ROCm port of SageAttention
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
import threading
from packaging.version import parse, Version
import warnings
from pathlib import Path

from setuptools import setup, find_packages
import torch

# Check if ROCm is available
try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
    ROCM_HOME = os.environ.get('ROCM_PATH') or os.environ.get('ROCM_HOME')
    if not ROCM_HOME and Path('/opt/rocm').exists():
        ROCM_HOME = '/opt/rocm'
except ImportError:
    raise RuntimeError("PyTorch with ROCm support is required to build this package.")

if not ROCM_HOME:
    raise RuntimeError("ROCm installation not found. Please install ROCm or set ROCM_PATH.")

# ROCm architecture detection
def get_rocm_arch():
    """Detect ROCm GPU architectures on the current machine."""
    archs = set()

    # Try to get architectures from current GPUs
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            # Extract gfx version from the name
            # MI250X -> gfx90a, MI300 -> gfx940, etc.
            device_name = props.name.lower()

            # Map common AMD GPUs to their architectures
            arch_map = {
                'mi100': 'gfx908',
                'mi200': 'gfx90a',
                'mi210': 'gfx90a',
                'mi250': 'gfx90a',
                'mi250x': 'gfx90a',
                'mi300': 'gfx940',
                'mi300a': 'gfx940',
                'mi300x': 'gfx941',
                '7900': 'gfx1100',  # RX 7900 XTX/XT
                '6900': 'gfx1030',  # RX 6900 XT
                '6800': 'gfx1030',  # RX 6800 XT
            }

            for gpu_name, arch in arch_map.items():
                if gpu_name in device_name:
                    archs.add(arch)
                    break
            else:
                # Try to get arch from gcnArchName
                if hasattr(props, 'gcnArchName'):
                    archs.add(props.gcnArchName)

    # Default to common datacenter architectures if no GPU detected
    if not archs:
        warnings.warn("No AMD GPUs detected. Defaulting to MI200/MI300 architectures.")
        archs = {'gfx90a', 'gfx940'}

    return list(archs)

# Supported ROCm architectures
SUPPORTED_ARCHS = {
    "gfx908": "MI100",
    "gfx90a": "MI200/MI250X",
    "gfx940": "MI300A",
    "gfx941": "MI300X",
    "gfx1030": "RX 6900 XT (RDNA2)",
    "gfx1100": "RX 7900 XTX (RDNA3)"
}

# Detect architectures
detected_archs = get_rocm_arch()
print(f"Detected ROCm architectures: {detected_archs}")

# Validate architectures
compute_capabilities = []
for arch in detected_archs:
    if arch in SUPPORTED_ARCHS:
        compute_capabilities.append(arch)
        print(f"Building for {arch} ({SUPPORTED_ARCHS[arch]})")
    else:
        warnings.warn(f"Unsupported architecture {arch}, skipping...")

if not compute_capabilities:
    raise RuntimeError("No supported ROCm architectures found.")

# Compiler flags
CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-D__HIP_PLATFORM_AMD__",
    "-DUSE_ROCM",
    "-DENABLE_BF16",
]

HIPCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fPIC",
    "-D__HIP_PLATFORM_AMD__",
    "-DUSE_ROCM",
    "--use_fast_math",
]

# Add architecture-specific flags
for arch in compute_capabilities:
    HIPCC_FLAGS.append(f"--offload-arch={arch}")

# Get ROCm version
def get_rocm_version():
    try:
        result = subprocess.run(
            [f"{ROCM_HOME}/bin/rocminfo"],
            capture_output=True,
            text=True
        )
        # Parse version from rocminfo output
        for line in result.stdout.split('\n'):
            if 'ROCm version' in line:
                version = line.split(':')[1].strip()
                return version
    except:
        pass
    return "unknown"

rocm_version = get_rocm_version()
print(f"ROCm version: {rocm_version}")

# Check for specific ROCm features
HAS_FP8 = False
HAS_MFMA = False
HAS_WMMA = False

for arch in compute_capabilities:
    if arch in ['gfx940', 'gfx941']:
        HAS_FP8 = True
        HAS_MFMA = True
    elif arch == 'gfx90a':
        HAS_MFMA = True
    elif arch in ['gfx1100']:
        HAS_WMMA = True

# Extension modules
ext_modules = []

# Include paths
include_dirs = [
    os.path.join(ROCM_HOME, 'include'),
    os.path.join(ROCM_HOME, 'include', 'hip'),
    os.path.join(ROCM_HOME, 'include', 'rocblas'),
    os.path.join(ROCM_HOME, 'include', 'hipblas'),
]

# Library directories
library_dirs = [
    os.path.join(ROCM_HOME, 'lib'),
    os.path.join(ROCM_HOME, 'lib64'),
]

# Libraries to link
libraries = ['hip_hcc', 'hipblas', 'rocblas']

# Main attention kernels
if HAS_MFMA or HAS_WMMA:
    # Build HIP kernels for architectures with matrix operations
    qattn_sources = [
        "csrc_rocm/qattn/pybind_hip.cpp",
    ]

    # Add architecture-specific kernels
    if 'gfx90a' in compute_capabilities:
        qattn_sources.append("csrc_rocm/qattn/qk_int_sv_f16_hip_gfx90a.hip")
        CXX_FLAGS.append("-DHAS_GFX90A")

    if 'gfx940' in compute_capabilities or 'gfx941' in compute_capabilities:
        qattn_sources.append("csrc_rocm/qattn/qk_int_sv_f8_hip_gfx940.hip")
        CXX_FLAGS.append("-DHAS_GFX940")
        if HAS_FP8:
            CXX_FLAGS.append("-DHAS_FP8")

    # Create extension
    qattn_extension = CppExtension(
        name="sageattention_rocm._qattn_hip",
        sources=qattn_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "hip": HIPCC_FLAGS,
        },
        extra_link_args=['-Wl,-rpath,' + os.path.join(ROCM_HOME, 'lib')],
    )
    ext_modules.append(qattn_extension)

# Fused operations
fused_sources = [
    "csrc_rocm/fused/pybind.cpp",
    "csrc_rocm/fused/fused.hip",
]

fused_extension = CppExtension(
    name="sageattention_rocm._fused_hip",
    sources=fused_sources,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "hip": HIPCC_FLAGS,
    },
    extra_link_args=['-Wl,-rpath,' + os.path.join(ROCM_HOME, 'lib')],
)
ext_modules.append(fused_extension)

# Custom build extension to handle HIP compilation
class HIPBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        # Set hipcc as the compiler for .hip files
        original_compiler = self.compiler.compiler_so

        def unix_wrap_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Use hipcc for .hip files
            if src.endswith('.hip'):
                self.compiler.set_executable('compiler_so', f'{ROCM_HOME}/bin/hipcc')
                postargs = HIPCC_FLAGS + (extra_postargs or [])
            else:
                self.compiler.set_executable('compiler_so', original_compiler)
                postargs = extra_postargs

            self.compiler.compile(
                [src],
                output_dir=os.path.dirname(obj),
                macros=pp_opts.get('macros', []),
                include_dirs=pp_opts.get('include_dirs', []),
                debug=pp_opts.get('debug', False),
                extra_preargs=cc_args,
                extra_postargs=postargs
            )
            return [obj]

        # Monkey-patch the compile method
        original_compile = self.compiler._compile
        self.compiler._compile = unix_wrap_compile

        super().build_extensions()

        # Restore original compiler
        self.compiler._compile = original_compile

# Package setup
setup(
    name='sageattention-rocm',
    version='1.0.0',
    author='SageAttention ROCm Port',
    license='Apache 2.0 License',
    description='ROCm port of SageAttention - Accurate and efficient plug-and-play low-bit attention.',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SageAttention',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'triton>=2.0.0',
        'numpy',
    ],
    ext_modules=ext_modules if ext_modules else None,
    cmdclass={"build_ext": HIPBuildExtension} if ext_modules else {},
    package_data={
        'sageattention_rocm': ['*.py'],
        'sageattention_rocm.triton': ['*.py'],
    },
    zip_safe=False,
)