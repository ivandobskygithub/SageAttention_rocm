"""
Quantization performance benchmarks.

Benchmarks INT4 quantization/dequantization:
- Throughput
- Memory bandwidth
- Different tensor sizes
- Comparison with FP16
"""

import torch
import argparse
import sys
from pathlib import Path
from typing import Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks import (
    benchmark_function,
    compute_memory_bandwidth,
    format_benchmark_results,
)

try:
    import sage_attention_rocm7 as sage_attn
    HAS_SAGE_ATTN = True
except ImportError:
    HAS_SAGE_ATTN = False
    print("ERROR: sage_attention_rocm7 not available")
    sys.exit(1)


def benchmark_quantization(
    shape: tuple,
    block_size: int = 16,
    device: str = 'cuda',
    warmup_iters: int = 10,
    num_iters: int = 100,
) -> Dict:
    """Benchmark quantization with given configuration"""

    results = {
        'config': {
            'shape': shape,
            'block_size': block_size,
        }
    }

    # Create input tensor
    x = torch.randn(*shape, dtype=torch.float16, device=device)

    # Memory sizes
    input_bytes = x.numel() * x.element_size()
    results['input_bytes'] = input_bytes

    # Benchmark quantization
    print(f"Benchmarking quantization...")
    bench_quant = benchmark_function(
        sage_attn.quantize_int4,
        x,
        block_size=block_size,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        device=device,
    )

    quant_bandwidth = compute_memory_bandwidth(input_bytes, bench_quant['mean_ms'])

    results['quantization'] = {
        **bench_quant,
        'bandwidth_gb_s': quant_bandwidth,
        'throughput_gb_s': (input_bytes / 1e9) / (bench_quant['mean_ms'] / 1000.0),
    }

    # Get quantized tensors for dequantization benchmark
    quantized, scales = sage_attn.quantize_int4(x, block_size=block_size)

    quantized_bytes = quantized.numel() * quantized.element_size()
    scales_bytes = scales.numel() * scales.element_size()
    total_quantized_bytes = quantized_bytes + scales_bytes

    results['quantized_bytes'] = total_quantized_bytes
    results['compression_ratio'] = input_bytes / total_quantized_bytes

    # Benchmark dequantization
    print(f"Benchmarking dequantization...")
    bench_dequant = benchmark_function(
        sage_attn.dequantize_int4,
        quantized, scales, x.shape,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        device=device,
    )

    dequant_bandwidth = compute_memory_bandwidth(total_quantized_bytes + input_bytes, bench_dequant['mean_ms'])

    results['dequantization'] = {
        **bench_dequant,
        'bandwidth_gb_s': dequant_bandwidth,
        'throughput_gb_s': (input_bytes / 1e9) / (bench_dequant['mean_ms'] / 1000.0),
    }

    # Combined quantization + dequantization
    def quant_dequant(x, block_size):
        q, s = sage_attn.quantize_int4(x, block_size=block_size)
        return sage_attn.dequantize_int4(q, s, x.shape)

    print(f"Benchmarking combined quant+dequant...")
    bench_combined = benchmark_function(
        quant_dequant,
        x, block_size,
        warmup_iters=warmup_iters,
        num_iters=num_iters,
        device=device,
    )

    results['combined'] = {
        **bench_combined,
        'bandwidth_gb_s': compute_memory_bandwidth(input_bytes * 2, bench_combined['mean_ms']),
    }

    return results


def print_results(results: Dict):
    """Print benchmark results"""
    config = results['config']

    print("\n" + "="*80)
    print(f"Quantization Benchmark Results")
    print("="*80)
    print(f"Configuration:")
    print(f"  Shape: {config['shape']}")
    print(f"  Block size: {config['block_size']}")
    print(f"  Input size: {results['input_bytes'] / 1e6:.2f} MB")
    print(f"  Quantized size: {results['quantized_bytes'] / 1e6:.2f} MB")
    print(f"  Compression ratio: {results['compression_ratio']:.2f}x")
    print("-"*80)

    # Quantization
    quant = results['quantization']
    print(f"\nQuantization:")
    print(f"  Time: {quant['mean_ms']:.3f} ± {quant['std_ms']:.3f} ms")
    print(f"  Bandwidth: {quant['bandwidth_gb_s']:.2f} GB/s")
    print(f"  Throughput: {quant['throughput_gb_s']:.2f} GB/s")

    # Dequantization
    dequant = results['dequantization']
    print(f"\nDequantization:")
    print(f"  Time: {dequant['mean_ms']:.3f} ± {dequant['std_ms']:.3f} ms")
    print(f"  Bandwidth: {dequant['bandwidth_gb_s']:.2f} GB/s")
    print(f"  Throughput: {dequant['throughput_gb_s']:.2f} GB/s")

    # Combined
    combined = results['combined']
    print(f"\nCombined (Quant + Dequant):")
    print(f"  Time: {combined['mean_ms']:.3f} ± {combined['std_ms']:.3f} ms")
    print(f"  Bandwidth: {combined['bandwidth_gb_s']:.2f} GB/s")

    print("="*80 + "\n")


def run_benchmark_suite(device: str = 'cuda'):
    """Run comprehensive quantization benchmark suite"""

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\nRunning on device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}\n")

    # Benchmark configurations
    configs = [
        # Small
        {'shape': (1024, 64), 'block_size': 16},
        {'shape': (2048, 128), 'block_size': 16},

        # Medium
        {'shape': (8192, 256), 'block_size': 16},
        {'shape': (16384, 512), 'block_size': 16},

        # Large (typical K/V in attention)
        {'shape': (32768, 128), 'block_size': 16},  # ~4K seq_len, 128 head_dim
        {'shape': (65536, 128), 'block_size': 16},  # ~8K seq_len, 128 head_dim

        # Different block sizes
        {'shape': (16384, 256), 'block_size': 32},
        {'shape': (16384, 256), 'block_size': 64},
    ]

    all_results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config}")
        print(f"{'='*80}")

        try:
            results = benchmark_quantization(
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
    print("QUANTIZATION BENCHMARK SUMMARY")
    print("="*80)

    if all_results:
        print("\nPerformance Summary:")
        print("-"*80)
        print(f"{'Shape':<25} {'Block':>8} {'Quant(ms)':>12} {'Dequant(ms)':>12} {'Compression':>12}")
        print("-"*80)

        for r in all_results:
            cfg = r['config']
            shape_str = f"{cfg['shape'][0]}x{cfg['shape'][1]}"
            block_size = cfg['block_size']

            quant_time = r['quantization']['mean_ms']
            dequant_time = r['dequantization']['mean_ms']
            compression = r['compression_ratio']

            print(f"{shape_str:<25} {block_size:>8} {quant_time:>11.3f} {dequant_time:>11.3f} {compression:>11.2f}x")

        print("-"*80)

        # Average metrics
        avg_quant_bw = sum(r['quantization']['bandwidth_gb_s'] for r in all_results) / len(all_results)
        avg_dequant_bw = sum(r['dequantization']['bandwidth_gb_s'] for r in all_results) / len(all_results)
        avg_compression = sum(r['compression_ratio'] for r in all_results) / len(all_results)

        print(f"\nAverage Quantization Bandwidth: {avg_quant_bw:.2f} GB/s")
        print(f"Average Dequantization Bandwidth: {avg_dequant_bw:.2f} GB/s")
        print(f"Average Compression Ratio: {avg_compression:.2f}x")

        print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark INT4 Quantization')

    parser.add_argument('--rows', type=int, default=None, help='Number of rows')
    parser.add_argument('--cols', type=int, default=None, help='Number of columns')
    parser.add_argument('--block-size', type=int, default=16, help='Block size for quantization')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--suite', action='store_true', help='Run full benchmark suite')

    args = parser.parse_args()

    if not HAS_SAGE_ATTN:
        print("ERROR: sage_attention_rocm7 not available")
        sys.exit(1)

    if args.suite:
        run_benchmark_suite(device=args.device)
    elif args.rows and args.cols:
        results = benchmark_quantization(
            shape=(args.rows, args.cols),
            block_size=args.block_size,
            device=args.device,
            warmup_iters=args.warmup,
            num_iters=args.iters,
        )
        print_results(results)
    else:
        print("ERROR: Either use --suite or provide both --rows and --cols")
        parser.print_help()
        sys.exit(1)
