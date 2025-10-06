@echo off
REM Main benchmark runner for SageAttention ROCm

echo ========================================================================
echo SageAttention ROCm Benchmarks
echo ========================================================================
echo.

REM Set environment variable for AOTriton (even though it's not working)
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

echo Step 1: Testing your environment...
echo ----------------------------------------
python validation\test_flash_attention.py

echo.
echo ========================================================================
echo Step 2: Running main performance benchmark...
echo ========================================================================
python bench\benchmark_final.py

echo.
echo ========================================================================
echo Optional: Run these for additional testing:
echo   - python bench\benchmark_optimized.py  (INT8 quantization test)
echo   - python validation\validate_cpu.py    (CPU validation)
echo ========================================================================

pause