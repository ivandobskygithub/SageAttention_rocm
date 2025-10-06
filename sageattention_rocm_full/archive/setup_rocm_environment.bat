@echo off
REM Setup script for ROCm PyTorch environment

echo ======================================================================
echo SageAttention ROCm Environment Setup
echo ======================================================================
echo.

REM Check current PyTorch
echo Checking current PyTorch installation...
python -c "import torch; print(f'Current PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
echo.

echo Options:
echo 1. Install PyTorch with ROCm 6.2 support (recommended)
echo 2. Check for existing ROCm environments
echo 3. Run benchmarks with current environment
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Installing PyTorch with ROCm 6.2 support...
    echo.

    REM Uninstall current PyTorch
    pip uninstall -y torch torchvision torchaudio

    REM Install ROCm version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

    echo.
    echo Installation complete! Verifying...
    python -c "import torch; print(f'New PyTorch: {torch.__version__}')"
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

    echo.
    echo Now running benchmarks...
    python benchmark_aotriton.py

) else if "%choice%"=="2" (
    echo.
    echo Checking for conda environments with ROCm PyTorch...
    echo.

    REM Check conda environments
    where conda >nul 2>&1
    if %errorlevel%==0 (
        conda env list
        echo.
        echo To activate a ROCm environment, use:
        echo   conda activate [environment_name]
        echo.
        echo Then run:
        echo   set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
        echo   python benchmark_aotriton.py
    ) else (
        echo Conda not found. Checking for Python virtual environments...
        echo.
        dir /b /ad ..\..\*env* 2>nul
        echo.
        echo To activate a virtual environment, use:
        echo   ..\..\[env_name]\Scripts\activate
        echo.
        echo Then run:
        echo   set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
        echo   python benchmark_aotriton.py
    )

) else if "%choice%"=="3" (
    echo.
    echo Running benchmarks with current environment...
    set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    python benchmark_aotriton.py

) else (
    echo Invalid choice!
)

echo.
pause