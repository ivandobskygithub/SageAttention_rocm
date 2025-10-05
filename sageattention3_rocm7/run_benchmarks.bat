@echo off
REM Run all benchmarks for SageAttention3 ROCm port (Windows)

echo ==========================================
echo SageAttention3 ROCm Benchmark Suite
echo ==========================================
echo.

REM Check if CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || (
    echo ERROR: CUDA/ROCm not available
    exit /b 1
)

REM Display GPU info
echo GPU Information:
python -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}'); print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
echo.

REM Run attention benchmarks
echo ==========================================
echo Running Attention Benchmarks...
echo ==========================================
python benchmarks\bench_attention.py --suite

REM Run quantization benchmarks
echo.
echo ==========================================
echo Running Quantization Benchmarks...
echo ==========================================
python benchmarks\bench_quantization.py --suite

echo.
echo ==========================================
echo Benchmark suite completed!
echo ==========================================
