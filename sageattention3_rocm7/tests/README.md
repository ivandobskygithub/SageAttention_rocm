# SageAttention3 ROCm - Test Suite

## Overview

Comprehensive test suite for validating the SageAttention3 ROCm port. Tests cover functionality, accuracy, and integration scenarios.

## Test Structure

```
tests/
├── __init__.py              # Test utilities and fixtures
├── conftest.py              # Pytest configuration
├── test_attention.py        # Attention forward pass tests
├── test_quantization.py     # INT4 quantization tests
└── test_integration.py      # Integration and real-world tests
```

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py -v

# GPU tests only (requires CUDA/ROCm)
python run_tests.py --gpu-only

# Accuracy tests only
python run_tests.py --accuracy-only

# Quick tests (exclude slow)
python run_tests.py --quick
```

### Using pytest directly

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_attention.py -v

# Specific test
pytest tests/test_attention.py::TestAttentionBasic::test_attention_shapes -v

# With markers
pytest -m gpu -v           # GPU tests only
pytest -m accuracy -v      # Accuracy tests only
pytest -m "not slow" -v    # Exclude slow tests
```

### With Coverage

```bash
python run_tests.py --coverage
# Or
pytest tests/ --cov=sage_attention_rocm7 --cov-report=html --cov-report=term
```

## Test Categories

### 1. Attention Tests (test_attention.py)

**TestAttentionBasic** - Basic functionality
- Shape validation for various configurations
- Dtype and device checks
- Contiguous tensor requirements

**TestAttentionCausal** - Causal masking
- Causal vs non-causal differences
- Accuracy vs PyTorch baseline

**TestAttentionAccuracy** - Numerical accuracy
- Error metrics vs PyTorch
- Scale parameter validation
- Various sequence lengths

**TestAttentionEdgeCases** - Edge cases
- Different Q/K/V sequence lengths
- Single head, single batch
- Zero initialization

### 2. Quantization Tests (test_quantization.py)

**TestQuantizationBasic** - Basic functionality
- Quantize/dequantize shapes
- Dtype and device validation
- Contiguous tensor requirements

**TestQuantizationAccuracy** - Accuracy validation
- Error bounds (SNR, MSE, relative error)
- Block scaling effectiveness
- Uniform value handling

**TestQuantizationMemory** - Memory efficiency
- Compression ratio validation
- 4D tensor quantization
- Memory savings calculation

**TestQuantizationEdgeCases** - Edge cases
- Zero values
- Large/small magnitude values
- Sign preservation

### 3. Integration Tests (test_integration.py)

**TestInt4AttentionIntegration** - INT4 integration
- INT4 vs FP16 accuracy
- Memory savings validation

**TestRealWorldPatterns** - Real-world usage
- LLM inference patterns
- Incremental generation
- Multi-query attention

**TestEndToEnd** - End-to-end workflows
- Full pipeline validation
- Batch processing
- Gradient checkpointing compatibility

## Test Markers

- `@pytest.mark.gpu` - Requires GPU (auto-skipped on CPU)
- `@pytest.mark.accuracy` - Numerical accuracy test
- `@pytest.mark.slow` - Long-running test
- `@pytest.mark.performance` - Performance benchmark

## Expected Pass Criteria

### Attention Forward
- ✅ Output shape matches input
- ✅ No NaN or Inf values
- ✅ Max relative error < 10% vs PyTorch
- ✅ Mean absolute error < 0.1

### INT4 Quantization
- ✅ Compression ratio > 3.0x
- ✅ SNR > 20 dB
- ✅ Mean relative error < 10%
- ✅ Sign preservation > 95%

### Integration
- ✅ INT4 attention error < 20% vs FP16
- ✅ Memory savings validated
- ✅ Works across batch sizes

## Troubleshooting

### GPU Not Available
Tests marked with `@pytest.mark.gpu` will be automatically skipped if GPU is not available.

### Import Errors
If `sage_attention_rocm7` cannot be imported, ensure it's built:
```bash
cd sageattention3_rocm7
python setup.py build_ext --inplace
```

### Test Failures
1. Check GPU compatibility: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
2. Verify ROCm installation: `rocminfo`
3. Check build logs for compilation errors
4. Run with verbose output: `pytest -vv -s`

### Accuracy Issues
- Verify PyTorch baseline works correctly
- Check tensor dtypes (must be float16)
- Ensure tensors are contiguous
- Compare with CPU fallback

## Adding New Tests

1. Create test function in appropriate test file
2. Use `@requires_gpu` decorator if GPU needed
3. Add appropriate markers (`@pytest.mark.accuracy`, etc.)
4. Follow existing test patterns
5. Include docstring describing test purpose

Example:
```python
@requires_gpu
@pytest.mark.accuracy
def test_new_feature():
    """Test description"""
    # Test code here
    pass
```

## Continuous Integration

For CI/CD pipelines:
```bash
# Install dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest tests/ --cov=sage_attention_rocm7 --cov-report=xml

# Exit with error if tests fail
pytest tests/ || exit 1
```

## Performance Testing

For performance-focused testing, use the benchmark suite instead:
```bash
python benchmarks/bench_attention.py --suite
python benchmarks/bench_quantization.py --suite
```

## Contact

For issues with tests:
1. Check test output and error messages
2. Review validation report (VALIDATION_REPORT.md)
3. Verify environment setup
4. Report issues with full logs
