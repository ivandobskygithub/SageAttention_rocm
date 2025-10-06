@echo off
REM Run benchmark with currently activated Python environment

echo Using current Python environment...
python --version

echo.
echo Setting AOTriton environment variable...
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

echo.
echo Testing Flash Attention backends first...
python test_flash_attention.py

echo.
echo Running full benchmarks...
python benchmark_aotriton.py

pause