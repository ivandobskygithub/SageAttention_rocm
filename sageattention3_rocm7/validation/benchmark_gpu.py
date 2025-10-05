"""
GPU Performance Benchmarking for SageAttention3 ROCm

This benchmark suite measures:
- Attention kernel performance (TFLOPS)
- INT4 quantization/dequantization throughput
- Memory bandwidth utilization
- Comparison with PyTorch baseline
- Scalability with different batch sizes and sequence lengths

Provides detailed performance reports for RDNA3.5 (gfx1151) GPU.
"""

import sys
import os
from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np

from torch_integration import SageAttentionROCm


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    time_ms: float
    memory_mb: float
    tflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "seq_len": self.seq_len,
            "head_dim": self.head_dim,
            "time_ms": self.time_ms,
            "memory_mb": self.memory_mb,
            "tflops": self.tflops,
            "bandwidth_gbps": self.bandwidth_gbps,
            "speedup_vs_baseline": self.speedup_vs_baseline,
        }


class AttentionBenchmark:
    """Benchmark suite for attention kernels."""

    def __init__(self, device: str = "cuda:0", warmup_iters: int = 5, bench_iters: int = 20):
        """
        Initialize benchmark.

        Args:
            device: CUDA device
            warmup_iters: Number of warmup iterations
            bench_iters: Number of benchmark iterations
        """
        self.device = torch.device(device)
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.sage = SageAttentionROCm()
        self.results: List[BenchmarkResult] = []

        # Get GPU properties
        self.gpu_props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {self.gpu_props.name}")
        print(f"Total memory: {self.gpu_props.total_memory / 1e9:.2f} GB")
        print(f"SM count: {self.gpu_props.multi_processor_count}")

    def benchmark_function(self, func, *args, **kwargs) -> tuple:
        """
        Benchmark a function.

        Args:
            func: Function to benchmark
            *args, **kwargs: Arguments to function

        Returns:
            (time_ms, peak_memory_mb)
        """
        # Warmup
        for _ in range(self.warmup_iters):
            func(*args, **kwargs)
        torch.cuda.synchronize()

        # Clear memory stats
        torch.cuda.reset_peak_memory_stats()

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(self.bench_iters):
            func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()

        # Get metrics
        time_ms = start_event.elapsed_time(end_event) / self.bench_iters
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB

        return time_ms, peak_memory

    def calculate_attention_flops(
        self, batch: int, num_heads: int, seq_len_q: int, seq_len_k: int, head_dim: int
    ) -> float:
        """
        Calculate FLOPs for attention computation.

        Attention: O = softmax(Q @ K^T / sqrt(d)) @ V

        Operations:
        1. Q @ K^T: batch * num_heads * seq_len_q * seq_len_k * head_dim * 2 (MAC)
        2. Softmax: batch * num_heads * seq_len_q * seq_len_k * 5 (approx)
        3. @ V: batch * num_heads * seq_len_q * seq_len_k * head_dim * 2 (MAC)

        Returns:
            Total FLOPs
        """
        qk_flops = batch * num_heads * seq_len_q * seq_len_k * head_dim * 2
        softmax_flops = batch * num_heads * seq_len_q * seq_len_k * 5
        v_flops = batch * num_heads * seq_len_q * seq_len_k * head_dim * 2

        return qk_flops + softmax_flops + v_flops

    def pytorch_attention_baseline(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float, is_causal: bool = False
    ) -> torch.Tensor:
        """PyTorch reference implementation."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if is_causal:
            seq_len_q, seq_len_k = Q.size(2), K.size(2)
            mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=Q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output

    def benchmark_attention_configs(self, configs: List[tuple]) -> List[BenchmarkResult]:
        """
        Benchmark attention with multiple configurations.

        Args:
            configs: List of (batch, num_heads, seq_len, head_dim) tuples

        Returns:
            List of benchmark results
        """
        print("\n" + "=" * 80)
        print("Attention Forward Pass Benchmark")
        print("=" * 80)

        results = []

        for batch, num_heads, seq_len, head_dim in configs:
            print(f"\nConfig: batch={batch}, heads={num_heads}, seq_len={seq_len}, dim={head_dim}")

            # Create inputs
            Q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=self.device)
            K = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=self.device)
            V = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device=self.device)
            scale = 1.0 / (head_dim ** 0.5)

            # Benchmark ROCm kernel
            time_rocm, mem_rocm = self.benchmark_function(
                self.sage.attention_forward, Q, K, V, scale=scale, is_causal=False
            )

            # Benchmark PyTorch baseline
            time_pytorch, mem_pytorch = self.benchmark_function(
                self.pytorch_attention_baseline, Q, K, V, scale, is_causal=False
            )

            # Calculate metrics
            flops = self.calculate_attention_flops(batch, num_heads, seq_len, seq_len, head_dim)
            tflops_rocm = (flops / 1e12) / (time_rocm / 1000)
            tflops_pytorch = (flops / 1e12) / (time_pytorch / 1000)
            speedup = time_pytorch / time_rocm

            # Print results
            print(f"  ROCm kernel:     {time_rocm:7.3f} ms, {tflops_rocm:6.2f} TFLOPS, {mem_rocm:7.2f} MB")
            print(f"  PyTorch baseline: {time_pytorch:7.3f} ms, {tflops_pytorch:6.2f} TFLOPS, {mem_pytorch:7.2f} MB")
            print(f"  Speedup: {speedup:.2f}x")

            # Store result
            result = BenchmarkResult(
                name="attention_forward",
                batch_size=batch,
                num_heads=num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                time_ms=time_rocm,
                memory_mb=mem_rocm,
                tflops=tflops_rocm,
                speedup_vs_baseline=speedup,
            )
            results.append(result)

        return results

    def benchmark_quantization(self, configs: List[tuple]) -> List[BenchmarkResult]:
        """
        Benchmark INT4 quantization/dequantization.

        Args:
            configs: List of (batch, tokens, heads, dim) tuples

        Returns:
            List of benchmark results
        """
        print("\n" + "=" * 80)
        print("INT4 Quantization/Dequantization Benchmark")
        print("=" * 80)

        results = []

        for batch, tokens, heads, dim in configs:
            print(f"\nConfig: batch={batch}, tokens={tokens}, heads={heads}, dim={dim}")

            # Create input
            x = torch.randn(batch, tokens, heads, dim, dtype=torch.float16, device=self.device)

            # Benchmark quantization
            time_quant, mem_quant = self.benchmark_function(
                self.sage.quantize_int4, x, block_size=16
            )

            # Benchmark full roundtrip
            def roundtrip():
                x_q, scales = self.sage.quantize_int4(x, block_size=16)
                x_recon = self.sage.dequantize_int4(x_q, scales, head_dim=dim)
                return x_recon

            time_roundtrip, mem_roundtrip = self.benchmark_function(roundtrip)

            # Calculate throughput
            num_elements = batch * tokens * heads * dim
            throughput_quant = num_elements / (time_quant / 1000) / 1e9  # Billion elements/sec
            throughput_roundtrip = num_elements / (time_roundtrip / 1000) / 1e9

            # Calculate bandwidth (rough estimate)
            bytes_read = num_elements * 2  # FP16 input
            bytes_write = num_elements // 2  # INT4 output + scales
            bandwidth = (bytes_read + bytes_write) / (time_quant / 1000) / 1e9  # GB/s

            print(f"  Quantize only:   {time_quant:7.3f} ms, {throughput_quant:6.2f} B elem/s, {mem_quant:7.2f} MB")
            print(f"  Full roundtrip:  {time_roundtrip:7.3f} ms, {throughput_roundtrip:6.2f} B elem/s, {mem_roundtrip:7.2f} MB")
            print(f"  Bandwidth:       {bandwidth:.2f} GB/s")

            result = BenchmarkResult(
                name="quantize_int4",
                batch_size=batch,
                num_heads=heads,
                seq_len=tokens,
                head_dim=dim,
                time_ms=time_quant,
                memory_mb=mem_quant,
                bandwidth_gbps=bandwidth,
            )
            results.append(result)

        return results

    def benchmark_scaling(self):
        """Benchmark performance scaling with different parameters."""
        print("\n" + "=" * 80)
        print("Scaling Benchmark")
        print("=" * 80)

        # Test sequence length scaling
        print("\nSequence Length Scaling (fixed batch=2, heads=8, dim=64):")
        seq_lens = [64, 128, 256, 512, 1024]
        seq_results = []

        for seq_len in seq_lens:
            Q = torch.randn(2, 8, seq_len, 64, dtype=torch.float16, device=self.device)
            K = torch.randn(2, 8, seq_len, 64, dtype=torch.float16, device=self.device)
            V = torch.randn(2, 8, seq_len, 64, dtype=torch.float16, device=self.device)

            time_ms, _ = self.benchmark_function(
                self.sage.attention_forward, Q, K, V, scale=1.0 / 8.0, is_causal=False
            )

            flops = self.calculate_attention_flops(2, 8, seq_len, seq_len, 64)
            tflops = (flops / 1e12) / (time_ms / 1000)

            print(f"  seq_len={seq_len:4d}: {time_ms:7.3f} ms, {tflops:6.2f} TFLOPS")
            seq_results.append((seq_len, time_ms, tflops))

        # Test batch size scaling
        print("\nBatch Size Scaling (fixed heads=8, seq_len=256, dim=64):")
        batch_sizes = [1, 2, 4, 8, 16]
        batch_results = []

        for batch in batch_sizes:
            Q = torch.randn(batch, 8, 256, 64, dtype=torch.float16, device=self.device)
            K = torch.randn(batch, 8, 256, 64, dtype=torch.float16, device=self.device)
            V = torch.randn(batch, 8, 256, 64, dtype=torch.float16, device=self.device)

            time_ms, _ = self.benchmark_function(
                self.sage.attention_forward, Q, K, V, scale=1.0 / 8.0, is_causal=False
            )

            flops = self.calculate_attention_flops(batch, 8, 256, 256, 64)
            tflops = (flops / 1e12) / (time_ms / 1000)

            print(f"  batch={batch:2d}: {time_ms:7.3f} ms, {tflops:6.2f} TFLOPS")
            batch_results.append((batch, time_ms, tflops))

        return seq_results, batch_results

    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark suite."""
        print("\n" + "=" * 80)
        print("SageAttention3 ROCm Comprehensive Performance Benchmark")
        print("GPU:", self.gpu_props.name)
        print("=" * 80)

        all_results = {}

        # Standard attention configurations
        attention_configs = [
            # (batch, heads, seq_len, dim)
            (1, 8, 128, 64),      # Small
            (2, 8, 256, 64),      # Medium
            (4, 8, 512, 64),      # Large
            (2, 16, 512, 128),    # Large heads
            (8, 8, 256, 64),      # Large batch
        ]

        all_results["attention"] = self.benchmark_attention_configs(attention_configs)

        # Quantization configurations
        quant_configs = [
            # (batch, tokens, heads, dim)
            (1, 512, 8, 64),
            (2, 1024, 8, 64),
            (4, 512, 16, 128),
            (8, 256, 8, 64),
        ]

        all_results["quantization"] = self.benchmark_quantization(quant_configs)

        # Scaling tests
        seq_scaling, batch_scaling = self.benchmark_scaling()
        all_results["seq_scaling"] = seq_scaling
        all_results["batch_scaling"] = batch_scaling

        return all_results

    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        output_path = Path(__file__).parent / filename

        # Convert results to serializable format
        serializable = {}
        for key, value in results.items():
            if isinstance(value, list):
                if value and isinstance(value[0], BenchmarkResult):
                    serializable[key] = [r.to_dict() for r in value]
                else:
                    serializable[key] = value
            else:
                serializable[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    def print_summary(self, results: Dict):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("Benchmark Summary")
        print("=" * 80)

        if "attention" in results:
            print("\nAttention Performance:")
            avg_tflops = np.mean([r.tflops for r in results["attention"]])
            avg_speedup = np.mean([r.speedup_vs_baseline for r in results["attention"]])
            print(f"  Average TFLOPS: {avg_tflops:.2f}")
            print(f"  Average speedup vs PyTorch: {avg_speedup:.2f}x")

        if "quantization" in results:
            print("\nQuantization Performance:")
            avg_bandwidth = np.mean([r.bandwidth_gbps for r in results["quantization"]])
            print(f"  Average bandwidth: {avg_bandwidth:.2f} GB/s")

        print("\n" + "=" * 80)


def main():
    """Main benchmark entry point."""
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available!")
        return 1

    # Create benchmark
    benchmark = AttentionBenchmark(warmup_iters=5, bench_iters=20)

    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()

    # Print summary
    benchmark.print_summary(results)

    # Save results
    benchmark.save_results(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
