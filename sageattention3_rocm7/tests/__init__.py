"""
Test suite for SageAttention3 ROCm7 port.

This package contains comprehensive tests for:
- Attention forward pass
- INT4 quantization/dequantization
- Numerical accuracy
- Edge cases
- Performance validation
"""

import torch
import pytest

# Test configuration
TEST_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_DTYPE = torch.float16

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "accuracy: mark test as testing numerical accuracy")
    config.addinivalue_line("markers", "performance: mark test as testing performance")

def requires_gpu(test_func):
    """Decorator to skip tests if GPU is not available"""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Test requires GPU"
    )(pytest.mark.gpu(test_func))

def get_test_device():
    """Get device for testing"""
    return TEST_DEVICE

def is_rocm_available():
    """Check if ROCm backend is available"""
    if not torch.cuda.is_available():
        return False

    try:
        # Check if using AMD GPU
        device_name = torch.cuda.get_device_name(0)
        return 'AMD' in device_name or 'Radeon' in device_name
    except:
        return False
