# SageAttention3 ROCm - Benchmark Suite

## Overview

Performance benchmarking suite for the SageAttention3 ROCm port. Measures throughput, latency, memory bandwidth, and compares with PyTorch baseline.

## Benchmark Structure

```
benchmarks/
├── __init__.py              # Benchmark utilities
├── bench_attention.py       # Attention performance benchmarks
├── bench_quantization.py    # Quantization performance benchmarks
└── README.md               # This file
```

## Quick Start

### Run Full Benchmark Suite

```bash
# Windows
run_benchmarks.bat

# Linux/WSL
bash run_benchmarks.sh

# Or manually
python benchmarks/bench_attention.py --suite
python benchmarks/bench_quantization.py --suite
```

### Custom Benchmarks

```bash
# Attention benchmark with specific config
python benchmarks/bench_attention.py \
    --batch-size 2 \
    --num-heads 8 \
    --seq-len 1024 \
    --head-dim 64 \
    --causal \
    --iters 100

# Quantization benchmark
python benchmarks/bench_quantization.py \
    --rows 8192 \
    --cols 256 \
    --block-size 16 \
    --iters 100
```

## Benchmark Configurations

### Attention Benchmarks

| Config | Batch | Heads | Seq Len | Head Dim | Use Case |
|--------|-------|-------|---------|----------|----------|
| Small  | 1     | 8     | 128     | 64       | Mobile/Edge |
| Small  | 1     | 8     | 512     | 64       | Mobile/Edge |
| Medium | 2     | 12    | 512     | 64       | Standard |
| Medium | 4     | 16    | 1024    | 64       | Standard |
| Large  | 1     | 32    | 2048    | 128      | LLM |
| Large  | 1     | 40    | 4096    | 128      | LLM |

### Quantization Benchmarks

| Shape | Block Size | Description |
|-------|------------|-------------|
| 1024×64 | 16 | Small tensors |
| 8192×256 | 16 | Medium tensors |
| 32768×128 | 16 | 4K seq_len |
| 65536×128 | 16 | 8K seq_len |
| 16384×256 | 32, 64 | Block size study |

## Metrics Reported

### Attention Benchmarks

**Latency:**
- Mean execution time (ms)
- Standard deviation
- Min/Max/Median times

**Throughput:**
- TFLOPS (tera floating-point operations per second)
- Effective memory bandwidth (GB/s)
- Tokens per second

**Comparison:**
- PyTorch baseline
- SageAttention FP16
- SageAttention INT4
- Speedup ratios

**Example Output:**
```
Configuration: B=2, H=8, S=1024, D=64
  Theoretical FLOPs: 2.15 TFLOPs
  Memory footprint: 16.78 MB

PyTorch Baseline:
  Time: 5.234 ± 0.123 ms
  Throughput: 2.45 TFLOPS
  Bandwidth: 128.3 GB/s

SageAttention FP16:
  Time: 4.891 ± 0.098 ms
  Throughput: 2.62 TFLOPS
  Bandwidth: 137.5 GB/s
  Speedup vs PyTorch: 1.07x

SageAttention INT4:
  Time: 3.456 ± 0.087 ms
  Throughput: 3.71 TFLOPS
  Bandwidth: 194.8 GB/s
  Speedup vs PyTorch: 1.51x
  Speedup vs FP16: 1.42x
```

### Quantization Benchmarks

**Performance:**
- Quantization time (ms)
- Dequantization time (ms)
- Memory bandwidth (GB/s)
- Throughput (GB/s)

**Compression:**
- Original size vs quantized size
- Compression ratio
- Overhead from scales

**Example Output:**
```
Shape: 8192×256, Block size: 16
  Input size: 4.19 MB
  Quantized size: 1.07 MB
  Compression ratio: 3.92x

Quantization:
  Time: 0.123 ± 0.008 ms
  Bandwidth: 136.5 GB/s
  Throughput: 34.1 GB/s

Dequantization:
  Time: 0.098 ± 0.006 ms
  Bandwidth: 171.3 GB/s
  Throughput: 42.8 GB/s
```

## Performance Targets (RDNA3.5)

### Attention
- **FP16 Throughput:** 2-5 TFLOPS
- **INT4 Speedup:** 1.3-2.0x vs FP16
- **Memory Bandwidth:** 50-150 GB/s (peak ~200 GB/s on iGPU)
- **Latency (2K seq):** 5-15 ms (batch=1, 32 heads)

### Quantization
- **Compression Ratio:** 3.5-4.0x
- **Quant Bandwidth:** 80-150 GB/s
- **Dequant Bandwidth:** 60-120 GB/s
- **SNR:** > 25 dB

## Interpreting Results

### Good Performance Indicators
- ✅ INT4 speedup > 1.3x vs PyTorch
- ✅ Memory bandwidth > 50% of peak
- ✅ Compression ratio > 3.5x
- ✅ Low variance (std < 10% of mean)

### Performance Issues
- ⚠️ Speedup < 1.0x (slower than baseline)
- ⚠️ High variance (std > 20% of mean)
- ⚠️ Compression ratio < 3.0x
- ⚠️ Bandwidth < 30% of peak

### Common Bottlenecks
1. **Memory Bandwidth** - Most common, especially for attention
2. **Kernel Launch Overhead** - Affects small batches
3. **Synchronization** - Can impact multi-kernel workflows
4. **Suboptimal Tiling** - Reduces occupancy

## Profiling

For detailed profiling:

```bash
# ROCm profiler
rocprof python benchmarks/bench_attention.py --batch-size 2 --num-heads 8 --seq-len 1024 --head-dim 64

# PyTorch profiler
python -c "
import torch
from benchmarks.bench_attention import benchmark_attention
with torch.profiler.profile() as prof:
    benchmark_attention(2, 8, 1024, 64)
print(prof.key_averages().table())
"
```

## Benchmark Best Practices

1. **Warmup:** Always use warmup iterations (default: 10)
2. **Iterations:** Use enough iterations for stable results (default: 100)
3. **GPU State:** Clear GPU memory between runs
4. **System Load:** Run on idle system for consistent results
5. **Power Mode:** Ensure GPU is in performance mode

## Adding Custom Benchmarks

Example custom benchmark:

```python
from benchmarks import benchmark_function, compute_attention_flops

def my_attention_function(Q, K, V):
    # Your implementation
    return output

# Benchmark it
results = benchmark_function(
    my_attention_function,
    Q, K, V,
    warmup_iters=10,
    num_iters=100,
    device='cuda'
)

print(f"Mean time: {results['mean_ms']:.3f} ms")
print(f"Throughput: {compute_attention_flops(...) / results['mean_ms']:.2f} GFLOPS")
```

## Environment Variables

```bash
# Force specific GPU
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_attention.py --suite

# Disable GPU (CPU benchmark)
CUDA_VISIBLE_DEVICES="" python benchmarks/bench_attention.py --suite

# ROCm debugging
export AMD_LOG_LEVEL=3
export HSA_ENABLE_SDMA=0
```

## Continuous Benchmarking

For CI/CD:

```bash
# Run benchmarks and save results
python benchmarks/bench_attention.py --suite > attention_results.txt
python benchmarks/bench_quantization.py --suite > quantization_results.txt

# Compare with baseline
python compare_benchmarks.py baseline.txt attention_results.txt
```

## Troubleshooting

### Low Performance
1. Check GPU is being used: `nvidia-smi` or `rocm-smi`
2. Verify correct architecture: `hipconfig --platform`
3. Check memory bandwidth: Compare with theoretical peak
4. Profile kernels: Use rocprof or PyTorch profiler

### High Variance
1. Increase iterations: `--iters 500`
2. Increase warmup: `--warmup 20`
3. Check system load: `top` or `htop`
4. Disable power management

### OOM Errors
1. Reduce batch size
2. Reduce sequence length
3. Clear GPU memory: `torch.cuda.empty_cache()`
4. Check available memory: `torch.cuda.mem_get_info()`

## Reporting Results

When reporting benchmark results, include:

1. **Hardware:** GPU model, memory, architecture
2. **Software:** ROCm version, PyTorch version, Python version
3. **Configuration:** Batch size, sequence length, etc.
4. **Results:** Mean ± std, throughput, bandwidth
5. **Comparison:** Speedup vs baseline
6. **Environment:** System load, power mode, etc.

Example:
```
Hardware: AMD Radeon 890M (gfx1151), 16GB shared
Software: ROCm 7.9.0rc, PyTorch 2.2.0+rocm, Python 3.11
Config: B=2, H=8, S=1024, D=64, Causal=False
Results: 4.891 ± 0.098 ms, 2.62 TFLOPS, 137.5 GB/s
Speedup: 1.07x vs PyTorch
```

## Contact

For benchmark issues:
1. Check hardware compatibility
2. Verify software versions
3. Review profiling results
4. Report with full system info
