"""
Setup for SageAttention ROCm with optional HIP kernel compilation
Supports both Triton-only and Triton+HIP builds
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import os
import subprocess
from pathlib import Path

# Configuration
BUILD_HIP_KERNELS = os.environ.get('SAGE_BUILD_HIP', '0') == '1'
ROCM_HOME = os.environ.get('ROCM_PATH') or os.environ.get('ROCM_HOME') or '/opt/rocm'

def get_rocm_arch():
    """Detect ROCm GPU architectures."""
    archs = []

    if torch.cuda.is_available():
        # Try to detect current GPU
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_name = props.name.lower()

            if 'mi300' in device_name:
                archs.append('gfx940')
            elif 'mi250' in device_name or 'mi200' in device_name:
                archs.append('gfx90a')
            elif 'mi100' in device_name:
                archs.append('gfx908')
            elif '7900' in device_name:
                archs.append('gfx1100')

    # Default to common architectures if none detected
    if not archs:
        archs = ['gfx90a', 'gfx940']

    return list(set(archs))  # Remove duplicates

# Extension modules
ext_modules = []

if BUILD_HIP_KERNELS and Path(ROCM_HOME).exists():
    print(f"Building with HIP kernels (ROCm path: {ROCM_HOME})")

    # Detect architectures
    archs = get_rocm_arch()
    print(f"Target architectures: {archs}")

    # HIP compiler flags
    hip_cflags = [
        '-O3',
        '-std=c++17',
        '-fPIC',
        '-D__HIP_PLATFORM_AMD__',
        '-DUSE_ROCM',
    ]

    # Add architecture flags
    for arch in archs:
        hip_cflags.append(f'--offload-arch={arch}')

    # Include directories
    include_dirs = [
        os.path.join(ROCM_HOME, 'include'),
        os.path.join(ROCM_HOME, 'include', 'hip'),
        'csrc_rocm',
        'csrc_rocm/utils',
    ]

    # Library directories
    library_dirs = [
        os.path.join(ROCM_HOME, 'lib'),
        os.path.join(ROCM_HOME, 'lib64'),
    ]

    # Fused operations extension
    fused_ext = CppExtension(
        name='sageattention_rocm._fused_hip',
        sources=['csrc_rocm/fused/fused.hip'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['amdhip64' if os.name == 'nt' else 'hip_hcc'],
        extra_compile_args={
            'cxx': hip_cflags,
            'hip': hip_cflags,
        },
        extra_link_args=['-Wl,-rpath,' + os.path.join(ROCM_HOME, 'lib')] if os.name != 'nt' else [],
    )
    ext_modules.append(fused_ext)

    # Custom build extension for HIP files
    class HIPBuildExtension(BuildExtension):
        def build_extensions(self):
            # Override compiler for .hip files
            original_compiler = self.compiler.compiler_so
            hipcc_path = os.path.join(ROCM_HOME, 'bin', 'hipcc')

            def compile_hip(sources, *args, **kwargs):
                hip_sources = [s for s in sources if s.endswith('.hip')]
                other_sources = [s for s in sources if not s.endswith('.hip')]

                # Compile HIP files with hipcc
                if hip_sources:
                    self.compiler.set_executable('compiler_so', hipcc_path)
                    self.compiler.compile(hip_sources, *args, **kwargs)

                # Compile other files with regular compiler
                if other_sources:
                    self.compiler.set_executable('compiler_so', original_compiler)
                    self.compiler.compile(other_sources, *args, **kwargs)

            # Monkey-patch compile method
            original_compile = self.compiler.compile
            self.compiler.compile = compile_hip

            super().build_extensions()

            # Restore original
            self.compiler.compile = original_compile

    build_ext_class = HIPBuildExtension
else:
    print("Building Triton-only version (no HIP kernels)")
    build_ext_class = BuildExtension

# Package setup
setup(
    name='sageattention-rocm',
    version='1.0.0',
    author='SageAttention ROCm Port',
    license='Apache 2.0 License',
    description='ROCm port of SageAttention with Triton and optional HIP kernels',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/ivandobskygithub/SageAttention_rocm',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'triton>=2.0.0',
        'numpy',
    ],
    ext_modules=ext_modules if ext_modules else None,
    cmdclass={'build_ext': build_ext_class} if ext_modules else {},
    package_data={
        'sageattention_rocm': ['*.py'],
        'sageattention_rocm.triton': ['*.py'],
    },
    zip_safe=False,
)

# Post-installation message
print("\n" + "="*60)
print("SageAttention ROCm Installation Complete")
print("="*60)
if BUILD_HIP_KERNELS:
    print("✓ Triton kernels installed")
    print("✓ HIP fused operations installed")
    print("\nTo use HIP kernels:")
    print("  from sageattention_rocm import fused_ops")
else:
    print("✓ Triton kernels installed")
    print("✗ HIP kernels not built (set SAGE_BUILD_HIP=1 to enable)")
    print("\nTo build with HIP kernels:")
    print("  SAGE_BUILD_HIP=1 pip install -e .")
print("\nUsage:")
print("  from sageattention_rocm import sageattn")
print("  output = sageattn(q, k, v)")
print("="*60)