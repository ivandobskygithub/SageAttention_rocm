"""
Attention performance benchmarks.

Benchmarks attention forward pass with various configurations:
- Different sequence lengths
- Different batch sizes
- Different head dimensions
- FP16 vs INT4 quantization
- Comparison with PyTorch baseline
"""

import torch
import argparse
import sys
from pathlib import Path
from typing import Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks import (
    benchmark_function,
    compute_attention_flops,
    compute_memory_bandwidth,
    format_benchmark_results,
)

try:
    import sage_attention_rocm7 as sage_attn
    HAS_SAGE_ATTN = True
except ImportError:
    HAS_SAGE_ATTN = False
    print("WARNING: sage_attention_rocm7 not available, only PyTorch baseline will be benchmarked")


def pytorch_attention(Q, K, V, scale=None):
    """PyTorch reference implementation"""
    if scale is None:
        scale = 1.0 / (Q.shape[-1] ** 0.5)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output


def benchmark_attention(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: str = 'cuda',
    use_causal: bool = False,
    warmup_iters: int = 10,
    num_iters: int = 100,
) -> Dict:
    """Benchmark attention with given configuration"""

    results = {
        'config': {
            'batch_size': batch_size,
            'num_heads': num_heads,
            'seq_len': seq_len,
            'head_dim': head_dim,
            'use_causal': use_causal,
        }
    }

    # Create input tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

    # Compute theoretical FLOPs
    total_flops = compute_attention_flops(batch_size, num_heads, seq_len, seq_len, head_dim)
    results['theoretical_flops'] = total_flops

    # Memory footprint
    input_bytes = (Q.numel() + K.numel() + V.numel()) * Q.element_size()
    output_bytes = Q.numel() * Q.element_size()
    total_memory_bytes = input_bytes + output_bytes
    results['memory_bytes'] = total_memory_bytes

    # Benchmark PyTorch baseline
    print(f"Benchmarking PyTorch baseline...")
    bench_pytorch = benchmark_function(
        pytorch_attention,
        Q, K, V,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        device=device,
    )

    pytorch_tflops = (total_flops / 1e12) / (bench_pytorch['mean_ms'] / 1000.0)
    pytorch_bandwidth = compute_memory_bandwidth(total_memory_bytes, bench_pytorch['mean_ms'])

    results['pytorch'] = {
        **bench_pytorch,
        'tflops': pytorch_tflops,
        'bandwidth_gb_s': pytorch_bandwidth,
    }

    # Benchmark SageAttention FP16
    if HAS_SAGE_ATTN:
        print(f"Benchmarking SageAttention FP16...")
        bench_sage_fp16 = benchmark_function(
            sage_attn.attention_forward,
            Q, K, V,
            is_causal=use_causal,
            use_int4_quantization=False,
            warmup_iters=warmup_iters,
            num_iters=num_iters,
            device=device,
        )

        sage_fp16_tflops = (total_flops / 1e12) / (bench_sage_fp16['mean_ms'] / 1000.0)
        sage_fp16_bandwidth = compute_memory_bandwidth(total_memory_bytes, bench_sage_fp16['mean_ms'])

        results['sage_fp16'] = {
            **bench_sage_fp16,
            'tflops': sage_fp16_tflops,
            'bandwidth_gb_s': sage_fp16_bandwidth,
            'speedup_vs_pytorch': bench_pytorch['mean_ms'] / bench_sage_fp16['mean_ms'],
        }

        # Benchmark SageAttention INT4
        print(f"Benchmarking SageAttention INT4...")
        bench_sage_int4 = benchmark_function(
            sage_attn.attention_forward,
            Q, K, V,
            is_causal=use_causal,
            use_int4_quantization=True,
            warmup_iters=warmup_iters,
            num_iters=num_iters,
            device=device,
        )

        # INT4 uses less memory bandwidth (K, V are quantized)
        # Approximate: K and V are ~4x smaller
        int4_memory_bytes = input_bytes // 2 + output_bytes  # Rough estimate

        sage_int4_tflops = (total_flops / 1e12) / (bench_sage_int4['mean_ms'] / 1000.0)
        sage_int4_bandwidth = compute_memory_bandwidth(int4_memory_bytes, bench_sage_int4['mean_ms'])

        results['sage_int4'] = {
            **bench_sage_int4,
            'tflops': sage_int4_tflops,
            'bandwidth_gb_s': sage_int4_bandwidth,
            'speedup_vs_pytorch': bench_pytorch['mean_ms'] / bench_sage_int4['mean_ms'],
            'speedup_vs_fp16': bench_sage_fp16['mean_ms'] / bench_sage_int4['mean_ms'],
        }

    return results


def print_results(results: Dict):
    """Print benchmark results in a formatted table"""
    config = results['config']

    print("\n" + "="*80)
    print(f"Attention Benchmark Results")
    print("="*80)
    print(f"Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Num heads: {config['num_heads']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Head dimension: {config['head_dim']}")
    print(f"  Causal: {config['use_causal']}")
    print(f"  Theoretical FLOPs: {results['theoretical_flops'] / 1e12:.2f} TFLOPs")
    print(f"  Memory footprint: {results['memory_bytes'] / 1e6:.2f} MB")
    print("-"*80)

    # PyTorch baseline
    pytorch = results['pytorch']
    print(f"\nPyTorch Baseline:")
    print(f"  Time: {pytorch['mean_ms']:.3f} ± {pytorch['std_ms']:.3f} ms")
    print(f"  Throughput: {pytorch['tflops']:.2f} TFLOPS")
    print(f"  Bandwidth: {pytorch['bandwidth_gb_s']:.2f} GB/s")

    # SageAttention FP16
    if 'sage_fp16' in results:
        sage_fp16 = results['sage_fp16']
        print(f"\nSageAttention FP16:")
        print(f"  Time: {sage_fp16['mean_ms']:.3f} ± {sage_fp16['std_ms']:.3f} ms")
        print(f"  Throughput: {sage_fp16['tflops']:.2f} TFLOPS")
        print(f"  Bandwidth: {sage_fp16['bandwidth_gb_s']:.2f} GB/s")
        print(f"  Speedup vs PyTorch: {sage_fp16['speedup_vs_pytorch']:.2f}x")

    # SageAttention INT4
    if 'sage_int4' in results:
        sage_int4 = results['sage_int4']
        print(f"\nSageAttention INT4:")
        print(f"  Time: {sage_int4['mean_ms']:.3f} ± {sage_int4['std_ms']:.3f} ms")
        print(f"  Throughput: {sage_int4['tflops']:.2f} TFLOPS")
        print(f"  Bandwidth: {sage_int4['bandwidth_gb_s']:.2f} GB/s")
        print(f"  Speedup vs PyTorch: {sage_int4['speedup_vs_pytorch']:.2f}x")
        print(f"  Speedup vs FP16: {sage_int4['speedup_vs_fp16']:.2f}x")

    print("="*80 + "\n")


def run_benchmark_suite(device: str = 'cuda'):
    """Run comprehensive benchmark suite"""

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available, cannot run benchmarks")
        return

    print(f"\nRunning on device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}\n")

    # Benchmark configurations
    configs = [
        # Small (mobile/edge inference)
        {'batch_size': 1, 'num_heads': 8, 'seq_len': 128, 'head_dim': 64},
        {'batch_size': 1, 'num_heads': 8, 'seq_len': 512, 'head_dim': 64},

        # Medium (typical transformer)
        {'batch_size': 2, 'num_heads': 12, 'seq_len': 512, 'head_dim': 64},
        {'batch_size': 4, 'num_heads': 16, 'seq_len': 1024, 'head_dim': 64},

        # Large (LLM inference)
        {'batch_size': 1, 'num_heads': 32, 'seq_len': 2048, 'head_dim': 128},
        {'batch_size': 1, 'num_heads': 40, 'seq_len': 4096, 'head_dim': 128},
    ]

    all_results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config}")
        print(f"{'='*80}")

        try:
            results = benchmark_attention(
                device=device,
                warmup_iters=10,
                num_iters=100,
                **config
            )
            print_results(results)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR: Benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    if HAS_SAGE_ATTN and all_results:
        print("\nSpeedup Summary (vs PyTorch):")
        print("-"*80)
        print(f"{'Config':<40} {'FP16 Speedup':>15} {'INT4 Speedup':>15}")
        print("-"*80)

        for r in all_results:
            cfg = r['config']
            config_str = f"B={cfg['batch_size']}, H={cfg['num_heads']}, S={cfg['seq_len']}, D={cfg['head_dim']}"

            fp16_speedup = r.get('sage_fp16', {}).get('speedup_vs_pytorch', 0)
            int4_speedup = r.get('sage_int4', {}).get('speedup_vs_pytorch', 0)

            print(f"{config_str:<40} {fp16_speedup:>14.2f}x {int4_speedup:>14.2f}x")

        print("-"*80)

        # Average speedups
        avg_fp16 = sum(r.get('sage_fp16', {}).get('speedup_vs_pytorch', 0) for r in all_results) / len(all_results)
        avg_int4 = sum(r.get('sage_int4', {}).get('speedup_vs_pytorch', 0) for r in all_results) / len(all_results)

        print(f"{'Average':<40} {avg_fp16:>14.2f}x {avg_int4:>14.2f}x")
        print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark SageAttention3 ROCm')

    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=None, help='Number of heads')
    parser.add_argument('--seq-len', type=int, default=None, help='Sequence length')
    parser.add_argument('--head-dim', type=int, default=None, help='Head dimension')
    parser.add_argument('--causal', action='store_true', help='Use causal masking')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--suite', action='store_true', help='Run full benchmark suite')

    args = parser.parse_args()

    if args.suite:
        run_benchmark_suite(device=args.device)
    elif all([args.batch_size, args.num_heads, args.seq_len, args.head_dim]):
        results = benchmark_attention(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            device=args.device,
            use_causal=args.causal,
            warmup_iters=args.warmup,
            num_iters=args.iters,
        )
        print_results(results)
    else:
        print("ERROR: Either use --suite or provide all of: --batch-size, --num-heads, --seq-len, --head-dim")
        parser.print_help()
        sys.exit(1)
