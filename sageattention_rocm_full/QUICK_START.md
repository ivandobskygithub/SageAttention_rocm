# 🚀 QUICK START - SageAttention ROCm

## One Command to Run Everything:

```bash
run_benchmarks.bat
```

Or run individually:

## The 2 Main Commands You Need:

### 1️⃣ Test Your Setup (5 seconds)
```bash
python validation\test_flash_attention.py
```
✅ Should show: Flash Attention Available

### 2️⃣ Run Performance Benchmark (1 minute)
```bash
python bench\benchmark_final.py
```
📊 Shows: Speed comparison of all backends

---

## That's It!

Everything else is optional. The benchmark will tell you:
- Which attention backend is fastest (Flash vs Efficient vs Math)
- Memory usage for different configurations
- Best approach for your GPU

## Results on Your AMD Radeon 8060S:
- **Flash Attention**: ✅ Working and fastest
- **INT8 Quantization**: ❌ Slower (0.45x)
- **Best approach**: Use PyTorch's native attention

---

*Full documentation in README.md*