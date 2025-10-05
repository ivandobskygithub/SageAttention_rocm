"""
Benchmark SageAttention ROCm Triton implementation
Following the same pattern as bench_qk_int8_pv_fp16_triton.py
"""

import torch
import argparse
import sys
import os
import time
from typing import Dict, List

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import ROCm implementation
from sageattention_rocm import sageattn, get_rocm_device_info

# Try to import flash_attn for consistent benchmarking
try:
    from flash_attn.utils.benchmark import benchmark_forward
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False
    print("flash_attn not available, using custom benchmark")

def custom_benchmark_forward(fn, *args, repeats=100, warmup=10, desc='', **kwargs):
    """Custom benchmark if flash_attn not available."""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Time
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = fn(*args, **kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    import numpy as np
    times = np.array(times) * 1000  # Convert to ms

    class Result:
        def __init__(self, times):
            self.mean = np.mean(times)
            self.std = np.std(times)
            self.min = np.min(times)
            self.max = np.max(times)

    return None, Result(times)


def run_benchmark(args):
    """Run benchmark with specified arguments."""

    # Use custom benchmark if flash_attn not available
    benchmark_fn = benchmark_forward if HAS_FLASH else custom_benchmark_forward

    head = args.num_heads
    batch = args.batch_size
    headdim = args.head_dim

    print(f"SageAttention ROCm Triton Benchmark")
    print("=" * 60)

    # Print device info
    device_info = get_rocm_device_info()
    if device_info:
        print(f"Device: {device_info['name']}")
        print(f"Architecture: {device_info.get('arch', 'unknown')}")
        print(f"Memory: {device_info['total_memory'] / 1e9:.1f} GB")

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch}")
    print(f"  Num heads: {head}")
    print(f"  Head dim: {headdim}")
    print(f"  Quantization granularity: {args.quant_gran}")
    print(f"  PV accumulation dtype: {args.pv_accum_dtype}")
    print(f"  Smooth K: {args.smooth_k}")
    print("=" * 60)

    # Test different sequence lengths
    seq_lens = [1024, 2048, 4096, 8192, 16384, 32768]

    # Adjust for available memory
    if device_info and device_info['total_memory'] < 16e9:  # Less than 16GB
        seq_lens = [s for s in seq_lens if s <= 8192]
        print(f"Note: Limiting sequence length to 8192 due to memory constraints")

    def run_sage_attention(q, k, v, is_causal):
        """Wrapper for SageAttention with specified options."""
        # Map quantization granularity to parameters
        # Note: Current Triton implementation uses per-block by default
        # This is a placeholder for future granularity options

        # Map accumulation dtype
        if args.pv_accum_dtype == "fp32":
            pv_dtype = "fp32"
        elif args.pv_accum_dtype == "fp16":
            pv_dtype = "fp16"
        elif args.pv_accum_dtype == "fp32+fp16":
            # This is SageAttention2++
            pv_dtype = "fp32"  # Use fp32 as primary
        else:
            pv_dtype = "fp32"

        return sageattn(
            q, k, v,
            tensor_layout="NHD",
            is_causal=is_causal,
            smooth_k=args.smooth_k,
            pv_accum_dtype=pv_dtype
        )

    # Non-causal benchmark
    print("\nNon-causal attention:")
    print(f"{'SeqLen':>8} {'Time(ms)':>12} {'TFLOPS':>12} {'Memory(GB)':>12}")
    print("-" * 50)

    is_causal = False
    for seq_len in seq_lens:
        try:
            flops = 4 * head * batch * headdim * seq_len * seq_len
            memory_gb = (batch * head * seq_len * headdim * 2 * 3) / 1e9

            # Create tensors
            q = torch.randn(batch, head, seq_len, headdim,
                          dtype=torch.float16, device="cuda")
            k = torch.randn(batch, head, seq_len, headdim,
                          dtype=torch.float16, device="cuda")
            v = torch.randn(batch, head, seq_len, headdim,
                          dtype=torch.float16, device="cuda")

            # Warmup
            for _ in range(5):
                run_sage_attention(q, k, v, is_causal)
            torch.cuda.synchronize()

            # Benchmark
            _, time = benchmark_fn(
                run_sage_attention, q, k, v, is_causal,
                repeats=args.repeats, desc='SageAttn ROCm'
            )

            tflops = flops / time.mean * 1e-15  # Convert to TFLOPS
            print(f"{seq_len:8d} {time.mean:12.3f} {tflops:12.2f} {memory_gb:12.2f}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:8d} {'OOM':>12} {'-':>12} {memory_gb:12.2f}")
                break
            else:
                raise

    # Causal benchmark
    print("\nCausal attention:")
    print(f"{'SeqLen':>8} {'Time(ms)':>12} {'TFLOPS':>12} {'Memory(GB)':>12}")
    print("-" * 50)

    is_causal = True
    for seq_len in seq_lens:
        try:
            flops = 4 * head * batch * headdim * seq_len * seq_len // 2  # Causal is half
            memory_gb = (batch * head * seq_len * headdim * 2 * 3) / 1e9

            q = torch.randn(batch, head, seq_len, headdim,
                          dtype=torch.float16, device="cuda")
            k = torch.randn(batch, head, seq_len, headdim,
                          dtype=torch.float16, device="cuda")
            v = torch.randn(batch, head, seq_len, headdim,
                          dtype=torch.float16, device="cuda")

            # Warmup
            for _ in range(5):
                run_sage_attention(q, k, v, is_causal)
            torch.cuda.synchronize()

            # Benchmark
            _, time = benchmark_fn(
                run_sage_attention, q, k, v, is_causal,
                repeats=args.repeats, desc='SageAttn ROCm'
            )

            tflops = flops / time.mean * 1e-15
            print(f"{seq_len:8d} {time.mean:12.3f} {tflops:12.2f} {memory_gb:12.2f}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:8d} {'OOM':>12} {'-':>12} {memory_gb:12.2f}")
                break
            else:
                raise

    # Compare different configurations if requested
    if args.compare:
        print("\n" + "=" * 60)
        print("Configuration Comparison (SeqLen=4096)")
        print("=" * 60)

        seq_len = 4096
        configs = [
            ("No smoothing", False, "fp32"),
            ("With smoothing", True, "fp32"),
            ("FP16 accumulation", True, "fp16"),
        ]

        q = torch.randn(batch, head, seq_len, headdim,
                      dtype=torch.float16, device="cuda")
        k = torch.randn(batch, head, seq_len, headdim,
                      dtype=torch.float16, device="cuda")
        v = torch.randn(batch, head, seq_len, headdim,
                      dtype=torch.float16, device="cuda")

        print(f"{'Config':>20} {'Time(ms)':>12} {'TFLOPS':>12}")
        print("-" * 45)

        for name, smooth, accum_dtype in configs:
            def fn(q, k, v, is_causal):
                return sageattn(q, k, v, tensor_layout="NHD",
                              is_causal=is_causal,
                              smooth_k=smooth,
                              pv_accum_dtype=accum_dtype)

            _, time = benchmark_fn(fn, q, k, v, False, repeats=args.repeats)
            flops = 4 * head * batch * headdim * seq_len * seq_len
            tflops = flops / time.mean * 1e-15

            print(f"{name:>20} {time.mean:12.3f} {tflops:12.2f}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark SageAttention ROCm Triton')

    # Model configuration
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size')
    parser.add_argument('--num_heads', type=int, default=32,
                      help='Number of attention heads')
    parser.add_argument('--head_dim', type=int, default=128,
                      help='Head dimension')

    # SageAttention configuration
    parser.add_argument('--quant_gran', type=str, default='per_block',
                      choices=['per_block', 'per_warp', 'per_thread'],
                      help='Quantization granularity (Note: ROCm currently uses per_block)')
    parser.add_argument('--pv_accum_dtype', type=str, default='fp32',
                      choices=['fp32', 'fp16', 'fp32+fp16', 'fp32+fp32'],
                      help='PV accumulation dtype')
    parser.add_argument('--smooth_k', action='store_true', default=True,
                      help='Apply key smoothing')
    parser.add_argument('--no_smooth_k', dest='smooth_k', action='store_false',
                      help='Disable key smoothing')

    # Benchmark configuration
    parser.add_argument('--repeats', type=int, default=100,
                      help='Number of benchmark iterations')
    parser.add_argument('--compare', action='store_true',
                      help='Compare different configurations')

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA/ROCm not available")
        return 1

    # Run benchmark
    run_benchmark(args)

    return 0


if __name__ == "__main__":
    exit(main())