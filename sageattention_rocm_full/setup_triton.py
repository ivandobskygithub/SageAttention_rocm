"""
Minimal setup for SageAttention ROCm Triton-only implementation
Targets ROCm 7 with Triton kernels only (no compiled extensions)
"""

from setuptools import setup, find_packages
import os

# Pure Python package - no compilation needed for Triton
setup(
    name='sageattention-rocm-triton',
    version='1.0.0',
    author='SageAttention ROCm Port',
    license='Apache 2.0 License',
    description='ROCm Triton port of SageAttention - Pure Python/Triton implementation',
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
    package_data={
        'sageattention_rocm': ['*.py'],
        'sageattention_rocm.triton': ['*.py'],
    },
    zip_safe=False,
)