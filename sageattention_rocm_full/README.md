# SageAttention ROCm Implementation

## Quick Start - What to Run

### 🚀 MAIN BENCHMARKS TO RUN:

```bash
# 1. FIRST - Test your environment
python validation/test_flash_attention.py

# 2. MAIN BENCHMARK - Run this for performance testing
python bench/benchmark_final.py

# 3. OPTIONAL - Test with INT8 quantization
python bench/benchmark_optimized.py
```

## Directory Structure

```
sageattention_rocm_full/
├── README.md                    # THIS FILE - Start here!
├── setup.py                     # Package installation
│
├── sageattention_rocm/          # Main implementation
│   ├── __init__.py
│   ├── core_optimized.py        # BEST: Optimized implementation using Flash Attention
│   ├── core_triton_aotriton.py  # AOTriton attempt (not working properly)
│   └── triton/                  # Triton kernels and fallbacks
│
├── bench/                       # Benchmark scripts
│   ├── benchmark_final.py       # ⭐ RUN THIS - Best benchmark showing Flash Attention
│   ├── benchmark_optimized.py   # Alternative benchmark with INT8
│   └── benchmark_aotriton.py    # Original AOTriton benchmark (shows issues)
│
├── validation/                  # Validation and testing
│   ├── test_flash_attention.py  # ⭐ RUN THIS FIRST - Tests available backends
│   ├── validate_cpu.py          # CPU validation (already passed ✅)
│   └── validate_simple.py       # Simple validation test
│
└── docs/                        # Documentation
    ├── BENCHMARKS_SUMMARY.md    # Performance results summary
    └── AOTRITON_STATUS_REPORT.md # AOTriton integration status
```

## Installation

```bash
# Navigate to the directory
cd sageattention_rocm_full

# Install the package (optional)
pip install -e .
```

## Required Environment

✅ **Your current setup is ready:**
- PyTorch 2.10.0a0+rocm7.9
- AMD Radeon 8060S Graphics
- Flash Attention: Available
- Memory Efficient Attention: Available

## Step-by-Step Usage

### Step 1: Verify Your Environment

```bash
python validation/test_flash_attention.py
```

Expected output:
- Should show "Flash Attention: Available"
- Should show "Memory Efficient: Available"

### Step 2: Run Main Benchmark

```bash
python bench/benchmark_final.py
```

This will:
- Test all available attention backends
- Compare Flash, Efficient, Math, and Auto modes
- Show memory usage
- Recommend best approach for your GPU

### Step 3: (Optional) Test INT8 Quantization

```bash
python bench/benchmark_optimized.py
```

Note: INT8 shows overhead on your GPU, but you can verify this yourself.

## Key Findings for Your System

1. **Flash Attention is working** ✅
2. **Best performance**: Use PyTorch's native `scaled_dot_product_attention`
3. **INT8 quantization**: Adds overhead (0.45x slower)
4. **AOTriton**: Functions exist but not working properly

## API Usage

### Best Approach (Using Flash Attention):

```python
import torch.nn.functional as F

# This automatically uses Flash Attention on your GPU
output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

### Using SageAttention Implementation:

```python
from sageattention_rocm.core_optimized import sageattn_optimized

# Without INT8 (faster on your GPU)
output = sageattn_optimized(q, k, v, use_int8=False)

# With INT8 (slower but uses less memory)
output = sageattn_optimized(q, k, v, use_int8=True)
```

## Troubleshooting

### If you see "CUDA not available":
You're using CPU-only PyTorch. Activate your ROCm environment:
```bash
# Your environment shows (.venv) when active
.venv\Scripts\activate  # or however you activate it
```

### If benchmarks are slow:
- Flash Attention is already optimal
- INT8 adds overhead on AMD GPUs
- Use larger batch sizes for better GPU utilization

## Performance Summary

| Method | Relative Speed | When to Use |
|--------|---------------|-------------|
| Flash Attention (Auto) | 1.0x (baseline) | **Always - Best choice** |
| Memory Efficient | ~1.0x | Automatically selected |
| INT8 Quantized | 0.45x (slower) | Only if memory constrained |
| Math Backend | 0.2-0.3x | Fallback only |

## Files to Ignore

These files are archived/redundant - don't use them:
- Any loose .py files not in directories
- Files in archive/ directory
- Multiple test_aotriton variants

---

**Bottom Line**: Run `python bench/benchmark_final.py` to see your GPU's performance!