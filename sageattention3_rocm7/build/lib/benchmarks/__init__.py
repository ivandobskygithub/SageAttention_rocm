"""
Benchmark suite for SageAttention3 ROCm port.

Provides performance benchmarking utilities and test cases for:
- Throughput (TFLOPS)
- Memory bandwidth
- Latency
- Comparison with PyTorch baseline
"""

import torch
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np


class BenchmarkTimer:
    """GPU-aware timer for accurate benchmarking"""

    def __init__(self, device='cuda', warmup_iters=5, num_iters=100):
        self.device = device
        self.warmup_iters = warmup_iters
        self.num_iters = num_iters
        self.use_cuda = device == 'cuda' and torch.cuda.is_available()

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0

    def get_elapsed_ms(self):
        """Get elapsed time in milliseconds"""
        return self.elapsed_ms


def benchmark_function(func, *args, warmup_iters=5, num_iters=100, device='cuda', **kwargs):
    """
    Benchmark a function with warmup and multiple iterations.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        warmup_iters: Number of warmup iterations
        num_iters: Number of benchmark iterations
        device: Device to use ('cuda' or 'cpu')
        **kwargs: Keyword arguments to pass to function

    Returns:
        dict: Benchmark results including mean, std, min, max times
    """
    use_cuda = device == 'cuda' and torch.cuda.is_available()

    # Warmup
    for _ in range(warmup_iters):
        func(*args, **kwargs)
        if use_cuda:
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iters):
        if use_cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            func(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        else:
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    times = np.array(times)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'num_iters': num_iters,
    }


def compute_attention_flops(batch_size: int, num_heads: int, seq_len_q: int,
                            seq_len_k: int, head_dim: int) -> int:
    """
    Compute theoretical FLOPs for attention.

    Attention FLOPs:
    - Q @ K^T: batch * heads * seq_q * seq_k * head_dim * 2 (matmul)
    - softmax: batch * heads * seq_q * seq_k * 5 (exp, sum, div, etc.)
    - attn @ V: batch * heads * seq_q * seq_k * head_dim * 2 (matmul)

    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length
        head_dim: Head dimension

    Returns:
        Total FLOPs
    """
    # Q @ K^T
    qk_flops = batch_size * num_heads * seq_len_q * seq_len_k * head_dim * 2

    # Softmax (approximate)
    softmax_flops = batch_size * num_heads * seq_len_q * seq_len_k * 5

    # Attn @ V
    av_flops = batch_size * num_heads * seq_len_q * seq_len_k * head_dim * 2

    return qk_flops + softmax_flops + av_flops


def compute_memory_bandwidth(data_bytes: int, time_ms: float) -> float:
    """
    Compute memory bandwidth in GB/s.

    Args:
        data_bytes: Total data transferred in bytes
        time_ms: Time taken in milliseconds

    Returns:
        Bandwidth in GB/s
    """
    return (data_bytes / 1e9) / (time_ms / 1000.0)


def format_benchmark_results(results: Dict[str, Any]) -> str:
    """Format benchmark results for printing"""
    lines = []
    lines.append("Benchmark Results:")
    lines.append("-" * 60)

    for key, value in results.items():
        if isinstance(value, float):
            if 'tflops' in key.lower():
                lines.append(f"  {key}: {value:.2f} TFLOPS")
            elif 'gb' in key.lower() or 'bandwidth' in key.lower():
                lines.append(f"  {key}: {value:.2f} GB/s")
            elif 'ms' in key.lower() or 'time' in key.lower():
                lines.append(f"  {key}: {value:.3f} ms")
            else:
                lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")

    lines.append("-" * 60)
    return "\n".join(lines)
