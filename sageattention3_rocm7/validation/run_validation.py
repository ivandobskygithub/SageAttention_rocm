"""
Main validation runner script.

This script runs both correctness tests and performance benchmarks,
then generates a comprehensive validation report.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Check that the environment is properly configured."""
    print("Checking environment...")

    # Check Python version
    print(f"  Python: {sys.version}")

    # Check PyTorch and CUDA
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
    except ImportError:
        print("  ERROR: PyTorch not installed!")
        return False

    # Check DLL exists
    dll_path = Path(__file__).parent.parent / "sage_attention_rocm7.dll"
    if not dll_path.exists():
        print(f"  ERROR: DLL not found at {dll_path}")
        return False
    print(f"  DLL found: {dll_path}")

    # Check ROCm dependencies
    venv_path = Path(os.environ.get("VIRTUAL_ENV", ".venv"))
    rocm_dll_path = venv_path / "Lib" / "site-packages" / "_rocm_sdk_core" / "lib" / "llvm" / "bin" / "amdhip64_7.dll"
    if not rocm_dll_path.exists():
        print(f"  WARNING: ROCm DLL not found at {rocm_dll_path}")
    else:
        print(f"  ROCm DLL found: {rocm_dll_path}")

    print("Environment check passed!")
    return True


def run_correctness_tests():
    """Run correctness validation tests."""
    print("\n" + "=" * 80)
    print("Running Correctness Tests")
    print("=" * 80)

    test_script = Path(__file__).parent / "test_gpu_kernels.py"

    # Run pytest
    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=False
    )

    return result.returncode == 0


def run_benchmarks():
    """Run performance benchmarks."""
    print("\n" + "=" * 80)
    print("Running Performance Benchmarks")
    print("=" * 80)

    benchmark_script = Path(__file__).parent / "benchmark_gpu.py"

    # Run benchmark
    result = subprocess.run(
        [sys.executable, str(benchmark_script)],
        capture_output=False
    )

    return result.returncode == 0


def generate_report(tests_passed: bool, benchmarks_passed: bool):
    """Generate validation report."""
    print("\n" + "=" * 80)
    print("Generating Validation Report")
    print("=" * 80)

    report_path = Path(__file__).parent / "VALIDATION_REPORT.md"

    # Read benchmark results if available
    results_path = Path(__file__).parent / "benchmark_results.json"
    benchmark_summary = "Benchmarks not run"

    if results_path.exists():
        import json
        with open(results_path) as f:
            results = json.load(f)

        # Extract summary stats
        if "attention" in results:
            attn_results = results["attention"]
            avg_tflops = sum(r["tflops"] for r in attn_results) / len(attn_results)
            avg_speedup = sum(r["speedup_vs_baseline"] for r in attn_results) / len(attn_results)
            benchmark_summary = f"Average: {avg_tflops:.2f} TFLOPS, {avg_speedup:.2f}x speedup vs PyTorch"

    # Generate report
    report_content = f"""# SageAttention3 ROCm Validation Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

- **Correctness Tests**: {'PASSED' if tests_passed else 'FAILED'}
- **Performance Benchmarks**: {'PASSED' if benchmarks_passed else 'FAILED'}

## Environment

- **GPU**: {get_gpu_info()}
- **PyTorch**: {get_pytorch_version()}
- **DLL**: sage_attention_rocm7.dll

## Test Results

### Correctness Tests

The validation suite includes:

1. **INT4 Quantization Tests**
   - Quantize/dequantize roundtrip accuracy
   - Quantization symmetry
   - Zero input handling
   - Memory savings verification (target: ~4x compression)
   - Range value handling

2. **Attention Forward Pass Tests**
   - Basic attention computation vs PyTorch reference
   - Causal masking support
   - Different sequence lengths
   - Single token edge case
   - Numerical stability with extreme values

3. **Memory Efficiency Tests**
   - Large batch handling
   - INT4 vs FP16 memory comparison

4. **Edge Case Tests**
   - Minimum dimensions
   - Power-of-2 dimensions

Status: {'PASSED' if tests_passed else 'FAILED'}

### Performance Benchmarks

{benchmark_summary}

See `benchmark_results.json` for detailed results.

## Validation Status

{'✓ All validations passed! The DLL is ready for integration.' if tests_passed and benchmarks_passed else '✗ Some validations failed. Please review the errors above.'}

## Next Steps

{'1. Integrate into ComfyUI workflow\n2. Test with real diffusion models\n3. Optimize for production use' if tests_passed and benchmarks_passed else '1. Fix failing tests\n2. Re-run validation\n3. Check kernel implementation'}

"""

    with open(report_path, 'w') as f:
        f.write(report_content)

    print(f"\nValidation report saved to: {report_path}")


def get_gpu_info():
    """Get GPU information string."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except:
        pass
    return "Unknown"


def get_pytorch_version():
    """Get PyTorch version string."""
    try:
        import torch
        return torch.__version__
    except:
        return "Unknown"


def main():
    """Main validation runner."""
    print("=" * 80)
    print("SageAttention3 ROCm Validation Suite")
    print("=" * 80)

    # Check environment
    if not check_environment():
        print("\nEnvironment check failed!")
        return 1

    # Run correctness tests
    tests_passed = run_correctness_tests()

    # Run benchmarks
    benchmarks_passed = run_benchmarks()

    # Generate report
    generate_report(tests_passed, benchmarks_passed)

    # Final status
    print("\n" + "=" * 80)
    if tests_passed and benchmarks_passed:
        print("✓ VALIDATION SUCCESSFUL")
        print("=" * 80)
        print("\nThe DLL is working correctly and ready for integration!")
        return 0
    else:
        print("✗ VALIDATION FAILED")
        print("=" * 80)
        print("\nPlease review the errors and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
