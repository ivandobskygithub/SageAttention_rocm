"""
Setup script for SageAttention3 ROCm7
"""
import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read version from __init__.py
version = "0.1.0"

# Get long description from README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "SageAttention3 ROCm7 - High-performance attention with INT4 quantization for AMD GPUs"

# Package metadata
setup(
    name="sageattention3-rocm7",
    version=version,
    author="SageAttention Team",
    description="SageAttention3 ROCm7 - High-performance attention with INT4 quantization for AMD GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sageattn/sageattention",

    # Package structure
    packages=find_packages(exclude=["tests", "validation", "hip", "build"]),
    package_data={
        "sageattention3_rocm7": [
            "sage_attention_rocm7.dll",  # Include the compiled DLL
            "*.pyd",  # Include any Python extension modules
        ],
    },
    include_package_data=True,

    # Dependencies
    install_requires=[
        "torch>=2.0.0",  # PyTorch with ROCm support
        "numpy>=1.19.0",
    ],

    # Python version requirement
    python_requires=">=3.8",

    # Entry points for convenience
    entry_points={
        "console_scripts": [
            "sage-rocm7-test=sageattention3_rocm7.test:main",
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: Microsoft :: Windows",
    ],

    # Additional metadata
    keywords="attention transformer quantization rocm amd gpu int4 fp4",
    project_urls={
        "Bug Reports": "https://github.com/sageattn/sageattention/issues",
        "Source": "https://github.com/sageattn/sageattention",
    },
)