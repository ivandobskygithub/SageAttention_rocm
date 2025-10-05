"""
Comprehensive validation suite for SageAttention ROCm implementation
Tests correctness against PyTorch baseline and benchmarks performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import ROCm implementation
from sageattention_rocm import sageattn, get_rocm_device_info
from sageattention_rocm.core_triton_v2 import sageattn_optimized
from sageattention_rocm.fused_ops import get_backend_info

# For baseline comparison
def pytorch_attention_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    dropout_p: float = 0.0
) -> torch.Tensor:
    """PyTorch SDPA baseline for comparison."""
    return F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        is_causal=is_causal
    )


@dataclass
class ValidationResult:
    """Results from validation test."""
    test_name: str
    passed: bool
    max_error: float
    mean_error: float
    relative_error: float
    time_ms: float
    details: str = ""


@dataclass
class BenchmarkResult:
    """Results from performance benchmark."""
    config: str
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    time_ms: float
    tflops: float
    memory_gb: float


class SageAttentionValidator:
    """Comprehensive validator for SageAttention ROCm."""

    def __init__(self, device: str = 'cuda', dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.validation_results: List[ValidationResult] = []
        self.benchmark_results: List[BenchmarkResult] = []

        # Get device info
        self.device_info = get_rocm_device_info() if torch.cuda.is_available() else None
        self.backend_info = get_backend_info()

    def print_system_info(self):
        """Print system and backend information."""
        print("=" * 80)
        print("SageAttention ROCm Validation Suite")
        print("=" * 80)

        if self.device_info:
            print(f"Device: {self.device_info['name']}")
            print(f"Architecture: {self.device_info.get('arch', 'unknown')}")
            print(f"Memory: {self.device_info['total_memory'] / 1e9:.1f} GB")

        print(f"\nBackends available:")
        print(f"  Triton: {'✓' if self.backend_info['triton'] else '✗'}")
        print(f"  HIP Fused: {'✓' if self.backend_info.get('hip_fused', False) else '✗'}")
        print(f"  HIP Validated: {'✓' if self.backend_info.get('hip_validated', False) else '✗'}")

        print(f"\nPyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print("=" * 80)

    def validate_correctness(
        self,
        batch_sizes: List[int] = [1, 2, 4],
        head_counts: List[int] = [8, 16, 32],
        seq_lens: List[int] = [128, 512, 1024, 2048],
        head_dims: List[int] = [64, 128],
        tolerance: float = 1e-2
    ) -> bool:
        """Validate correctness against PyTorch baseline."""
        print("\n" + "=" * 80)
        print("Correctness Validation")
        print("=" * 80)

        all_passed = True

        for batch in batch_sizes:
            for heads in head_counts:
                for seq_len in seq_lens:
                    for head_dim in head_dims:
                        # Skip very large configurations
                        if batch * heads * seq_len * head_dim > 1e9:
                            continue

                        # Test non-causal
                        result = self._validate_single_config(
                            batch, heads, seq_len, head_dim,
                            is_causal=False, tolerance=tolerance
                        )
                        self.validation_results.append(result)
                        all_passed &= result.passed

                        # Test causal
                        result = self._validate_single_config(
                            batch, heads, seq_len, head_dim,
                            is_causal=True, tolerance=tolerance
                        )
                        self.validation_results.append(result)
                        all_passed &= result.passed

        # Print summary
        self._print_validation_summary()

        return all_passed

    def _validate_single_config(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        head_dim: int,
        is_causal: bool,
        tolerance: float
    ) -> ValidationResult:
        """Validate a single configuration."""
        config_name = f"B{batch}_H{heads}_L{seq_len}_D{head_dim}{'_causal' if is_causal else ''}"

        try:
            # Create test tensors
            q = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=self.dtype, device=self.device)
            k = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=self.dtype, device=self.device)
            v = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=self.dtype, device=self.device)

            # Compute baseline
            with torch.no_grad():
                baseline = pytorch_attention_baseline(q, k, v, is_causal=is_causal)

            # Compute SageAttention ROCm
            start = time.perf_counter()
            with torch.no_grad():
                output = sageattn(q, k, v, tensor_layout="NHD", is_causal=is_causal)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

            # Compute errors
            diff = (output - baseline).abs()
            max_error = diff.max().item()
            mean_error = diff.mean().item()
            relative_error = (diff / (baseline.abs() + 1e-6)).mean().item()

            # Check if passed
            passed = max_error < tolerance

            return ValidationResult(
                test_name=config_name,
                passed=passed,
                max_error=max_error,
                mean_error=mean_error,
                relative_error=relative_error,
                time_ms=elapsed,
                details=f"{'PASS' if passed else 'FAIL'}: max_err={max_error:.6f}"
            )

        except Exception as e:
            return ValidationResult(
                test_name=config_name,
                passed=False,
                max_error=float('inf'),
                mean_error=float('inf'),
                relative_error=float('inf'),
                time_ms=0,
                details=f"ERROR: {str(e)}"
            )

    def _print_validation_summary(self):
        """Print validation results summary."""
        print("\nValidation Results Summary:")
        print("-" * 80)

        passed = sum(1 for r in self.validation_results if r.passed)
        total = len(self.validation_results)

        print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")

        if passed < total:
            print("\nFailed tests:")
            for result in self.validation_results:
                if not result.passed:
                    print(f"  {result.test_name}: {result.details}")

        # Statistics
        errors = [r.max_error for r in self.validation_results if r.passed]
        if errors:
            print(f"\nError statistics (passed tests):")
            print(f"  Max error: {max(errors):.6f}")
            print(f"  Mean max error: {np.mean(errors):.6f}")
            print(f"  Median max error: {np.median(errors):.6f}")

    def benchmark_performance(
        self,
        seq_lens: List[int] = [1024, 2048, 4096, 8192],
        batch_size: int = 4,
        num_heads: int = 32,
        head_dim: int = 128,
        configs: List[str] = None
    ):
        """Benchmark performance of different configurations."""
        print("\n" + "=" * 80)
        print("Performance Benchmarks")
        print("=" * 80)
        print(f"Config: Batch={batch_size}, Heads={num_heads}, HeadDim={head_dim}")
        print("-" * 80)

        if configs is None:
            configs = [
                "pytorch_baseline",
                "sageattn_triton",
                "sageattn_optimized",
            ]

        for config in configs:
            print(f"\n{config}:")
            for is_causal in [False, True]:
                print(f"  {'Causal' if is_causal else 'Non-causal'}:")
                for seq_len in seq_lens:
                    result = self._benchmark_single(
                        config, batch_size, num_heads, seq_len, head_dim, is_causal
                    )
                    if result:
                        self.benchmark_results.append(result)
                        print(f"    L={seq_len:5d}: {result.time_ms:7.2f} ms, "
                              f"{result.tflops:6.2f} TFLOPS")

    def _benchmark_single(
        self,
        config: str,
        batch: int,
        heads: int,
        seq_len: int,
        head_dim: int,
        is_causal: bool,
        num_warmup: int = 5,
        num_iters: int = 50
    ) -> BenchmarkResult:
        """Benchmark a single configuration."""
        try:
            # Create test tensors
            q = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=self.dtype, device=self.device)
            k = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=self.dtype, device=self.device)
            v = torch.randn(batch, heads, seq_len, head_dim,
                          dtype=self.dtype, device=self.device)

            # Select function based on config
            if config == "pytorch_baseline":
                fn = lambda: pytorch_attention_baseline(q, k, v, is_causal)
            elif config == "sageattn_triton":
                fn = lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=is_causal)
            elif config == "sageattn_optimized":
                fn = lambda: sageattn_optimized(q, k, v, is_causal=is_causal, use_fused_ops=True)
            else:
                return None

            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = fn()
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iters):
                with torch.no_grad():
                    _ = fn()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            # Calculate metrics
            time_ms = (elapsed / num_iters) * 1000
            flops = 4 * batch * heads * seq_len * seq_len * head_dim
            if is_causal:
                flops = flops // 2
            tflops = flops / (time_ms / 1000) / 1e12

            # Memory usage (approximate)
            memory_gb = (batch * heads * seq_len * head_dim * 2 * 3) / 1e9  # Q, K, V

            return BenchmarkResult(
                config=f"{config}{'_causal' if is_causal else ''}",
                batch_size=batch,
                num_heads=heads,
                seq_len=seq_len,
                head_dim=head_dim,
                time_ms=time_ms,
                tflops=tflops,
                memory_gb=memory_gb
            )

        except Exception as e:
            print(f"      ERROR: {e}")
            return None

    def compare_with_baseline(self):
        """Compare ROCm implementation with baseline benchmarks."""
        print("\n" + "=" * 80)
        print("Comparison with Baseline")
        print("=" * 80)

        # Standard configurations from bench_baseline.py
        batch = 4
        heads = 32
        head_dim = 128

        print(f"\nConfiguration: B={batch}, H={heads}, D={head_dim}")
        print("-" * 60)

        seq_lens = [1024, 2048, 4096, 8192, 16384]

        results = {
            'pytorch': {'causal': {}, 'non_causal': {}},
            'sageattn': {'causal': {}, 'non_causal': {}}
        }

        for is_causal in [False, True]:
            causal_key = 'causal' if is_causal else 'non_causal'
            print(f"\n{'Causal' if is_causal else 'Non-causal'} Attention:")
            print(f"{'SeqLen':>8} {'PyTorch':>12} {'SageAttn':>12} {'Speedup':>10}")
            print("-" * 45)

            for seq_len in seq_lens:
                # Skip very large sequences if memory limited
                if seq_len > 8192 and self.device_info and self.device_info['total_memory'] < 16e9:
                    continue

                # Benchmark PyTorch
                pytorch_result = self._benchmark_single(
                    "pytorch_baseline", batch, heads, seq_len, head_dim, is_causal
                )

                # Benchmark SageAttention
                sage_result = self._benchmark_single(
                    "sageattn_optimized", batch, heads, seq_len, head_dim, is_causal
                )

                if pytorch_result and sage_result:
                    speedup = pytorch_result.time_ms / sage_result.time_ms
                    results['pytorch'][causal_key][seq_len] = pytorch_result.tflops
                    results['sageattn'][causal_key][seq_len] = sage_result.tflops

                    print(f"{seq_len:8d} {pytorch_result.tflops:10.2f}TF {sage_result.tflops:10.2f}TF "
                          f"{speedup:8.2f}x")

        return results

    def run_stress_test(self, max_seq_len: int = 32768):
        """Run stress tests with large sequences."""
        print("\n" + "=" * 80)
        print("Stress Testing")
        print("=" * 80)

        configs = [
            (1, 8, 16384, 128),   # Large sequence
            (2, 16, 8192, 128),   # Medium batch/sequence
            (8, 32, 2048, 128),   # Large batch
            (1, 64, 4096, 64),    # Many heads
        ]

        if max_seq_len <= 16384:
            configs = [(b, h, min(s, max_seq_len), d) for b, h, s, d in configs]

        for batch, heads, seq_len, head_dim in configs:
            print(f"\nTesting B={batch}, H={heads}, L={seq_len}, D={head_dim}")

            try:
                q = torch.randn(batch, heads, seq_len, head_dim,
                              dtype=self.dtype, device=self.device)
                k = torch.randn(batch, heads, seq_len, head_dim,
                              dtype=self.dtype, device=self.device)
                v = torch.randn(batch, heads, seq_len, head_dim,
                              dtype=self.dtype, device=self.device)

                # Test with different configurations
                configs_to_test = [
                    ("Triton", lambda: sageattn(q, k, v, tensor_layout="NHD")),
                    ("Triton+Smooth", lambda: sageattn(q, k, v, smooth_k=True)),
                    ("Optimized", lambda: sageattn_optimized(q, k, v, use_fused_ops=True)),
                ]

                for name, fn in configs_to_test:
                    try:
                        start = time.perf_counter()
                        with torch.no_grad():
                            output = fn()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000

                        # Check for NaN/Inf
                        has_nan = output.isnan().any().item()
                        has_inf = output.isinf().any().item()

                        status = "✓"
                        if has_nan:
                            status = "✗ (NaN)"
                        elif has_inf:
                            status = "✗ (Inf)"

                        print(f"  {name:15s}: {elapsed:7.2f} ms {status}")

                    except Exception as e:
                        print(f"  {name:15s}: ERROR - {str(e)[:50]}")

            except Exception as e:
                print(f"  Failed to allocate tensors: {e}")

    def export_results(self, filename: str = "validation_results.txt"):
        """Export results to file."""
        with open(filename, 'w') as f:
            f.write("SageAttention ROCm Validation Results\n")
            f.write("=" * 80 + "\n\n")

            # System info
            f.write("System Information:\n")
            if self.device_info:
                f.write(f"  Device: {self.device_info['name']}\n")
                f.write(f"  Architecture: {self.device_info.get('arch', 'unknown')}\n")

            # Validation results
            f.write("\nValidation Results:\n")
            passed = sum(1 for r in self.validation_results if r.passed)
            total = len(self.validation_results)
            f.write(f"  Passed: {passed}/{total}\n")

            # Benchmark results
            if self.benchmark_results:
                f.write("\nBenchmark Results (TFLOPS):\n")
                for result in self.benchmark_results:
                    f.write(f"  {result.config} L={result.seq_len}: {result.tflops:.2f}\n")

        print(f"\nResults exported to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Validate SageAttention ROCm implementation')
    parser.add_argument('--quick', action='store_true', help='Run quick validation only')
    parser.add_argument('--full', action='store_true', help='Run full validation suite')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--stress', action='store_true', help='Run stress tests')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for benchmarks')
    parser.add_argument('--num-heads', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--head-dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--export', type=str, help='Export results to file')

    args = parser.parse_args()

    # Create validator
    validator = SageAttentionValidator()
    validator.print_system_info()

    # Default to quick test if nothing specified
    if not any([args.quick, args.full, args.benchmark, args.stress]):
        args.quick = True

    # Run validation
    if args.quick:
        print("\nRunning quick validation...")
        validator.validate_correctness(
            batch_sizes=[1, 2],
            head_counts=[8, 16],
            seq_lens=[128, 512, 1024],
            head_dims=[64, 128]
        )

    if args.full:
        print("\nRunning full validation...")
        validator.validate_correctness(
            batch_sizes=[1, 2, 4, 8],
            head_counts=[8, 16, 32, 64],
            seq_lens=[128, 256, 512, 1024, 2048, 4096],
            head_dims=[64, 128, 256]
        )

    if args.benchmark:
        print("\nRunning benchmarks...")
        validator.benchmark_performance(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim
        )
        validator.compare_with_baseline()

    if args.stress:
        print("\nRunning stress tests...")
        validator.run_stress_test()

    # Export results if requested
    if args.export:
        validator.export_results(args.export)

    # Final summary
    print("\n" + "=" * 80)
    print("Validation Complete")
    print("=" * 80)

    if validator.validation_results:
        passed = sum(1 for r in validator.validation_results if r.passed)
        total = len(validator.validation_results)
        print(f"Correctness: {passed}/{total} tests passed ({100*passed/total:.1f}%)")

    if validator.benchmark_results:
        avg_tflops = np.mean([r.tflops for r in validator.benchmark_results])
        print(f"Performance: Average {avg_tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()