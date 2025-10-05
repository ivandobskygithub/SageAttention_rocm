@echo off
REM ============================================================================
REM SageAttention3 ROCm7 Build Script for Windows
REM
REM This script builds the HIP kernels for AMD RDNA3.5 (gfx1151) using ROCm 7
REM on Windows. It handles environment setup and invokes the build system.
REM
REM Usage:
REM   build.bat [clean|debug|release|install|test]
REM
REM Options:
REM   clean   - Clean build artifacts
REM   debug   - Build with debug symbols
REM   release - Build optimized (default)
REM   install - Install Python package
REM   test    - Run tests after building
REM ============================================================================

setlocal enabledelayedexpansion

REM =========================
REM Configuration
REM =========================

REM Default ROCm installation path
if not defined ROCM_PATH (
    set "ROCM_PATH=C:\Program Files\AMD\ROCm\7.0"
)

REM Default GPU target
if not defined GPU_TARGETS (
    set GPU_TARGETS=gfx1151
)

REM Build type (Release or Debug)
set BUILD_TYPE=Release
set SAGE_DEBUG_BUILD=FALSE

REM Python executable (use virtual environment if active)
if defined VIRTUAL_ENV (
    set PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe
) else (
    set PYTHON_EXE=python
)

REM Build directory
set BUILD_DIR=%~dp0build
set INSTALL_DIR=%~dp0install

REM =========================
REM Parse Arguments
REM =========================

set DO_CLEAN=0
set DO_BUILD=1
set DO_INSTALL=0
set DO_TEST=0

:parse_args
if "%~1"=="" goto end_parse_args
if /i "%~1"=="clean" (
    set DO_CLEAN=1
    shift
    goto parse_args
)
if /i "%~1"=="debug" (
    set BUILD_TYPE=Debug
    set SAGE_DEBUG_BUILD=TRUE
    shift
    goto parse_args
)
if /i "%~1"=="release" (
    set BUILD_TYPE=Release
    set SAGE_DEBUG_BUILD=FALSE
    shift
    goto parse_args
)
if /i "%~1"=="install" (
    set DO_INSTALL=1
    shift
    goto parse_args
)
if /i "%~1"=="test" (
    set DO_TEST=1
    shift
    goto parse_args
)
echo Unknown option: %~1
echo Usage: build.bat [clean^|debug^|release^|install^|test]
exit /b 1

:end_parse_args

REM =========================
REM Print Configuration
REM =========================

echo.
echo ============================================================
echo SageAttention3 ROCm7 Build Configuration
echo ============================================================
echo ROCm Path:      %ROCM_PATH%
echo GPU Targets:    %GPU_TARGETS%
echo Build Type:     %BUILD_TYPE%
echo Python:         %PYTHON_EXE%
echo Build Dir:      %BUILD_DIR%
echo Debug Build:    %SAGE_DEBUG_BUILD%
echo ============================================================
echo.

REM =========================
REM Environment Setup
REM =========================

echo [1/4] Setting up environment...

REM Check if ROCm is installed
if not exist "%ROCM_PATH%" (
    echo ERROR: ROCm not found at %ROCM_PATH%
    echo Please install ROCm 7.0+ or set ROCM_PATH environment variable
    exit /b 1
)

REM Add ROCm to PATH
set "PATH=%ROCM_PATH%\bin;%PATH%"
set "PATH=%ROCM_PATH%\lib;%PATH%"

REM If using virtual environment, check for hipcc in Scripts
if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\hipcc.exe" (
        echo Using hipcc from virtual environment
        set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
    )
)

REM Verify hipcc is available
where hipcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: hipcc not found in PATH
    echo Please ensure ROCm is properly installed
    exit /b 1
)

REM Print hipcc version
echo.
hipcc --version
echo.

REM Check Python
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found at %PYTHON_EXE%
    exit /b 1
)

echo Python version:
%PYTHON_EXE% --version
echo.

REM =========================
REM Clean (if requested)
REM =========================

if %DO_CLEAN%==1 (
    echo [2/4] Cleaning build artifacts...
    if exist "%BUILD_DIR%" (
        echo Removing %BUILD_DIR%
        rmdir /s /q "%BUILD_DIR%"
    )
    if exist "%INSTALL_DIR%" (
        echo Removing %INSTALL_DIR%
        rmdir /s /q "%INSTALL_DIR%"
    )
    if exist "dist" (
        echo Removing dist
        rmdir /s /q "dist"
    )
    if exist "build" (
        echo Removing build
        rmdir /s /q "build"
    )
    if exist "*.egg-info" (
        echo Removing egg-info
        for /d %%i in (*.egg-info) do rmdir /s /q "%%i"
    )
    echo Clean complete.
    if %DO_BUILD%==0 (
        exit /b 0
    )
)

REM =========================
REM Build with setup.py
REM =========================

if %DO_BUILD%==1 (
    echo [3/4] Building Python extension...

    REM Export environment variables for setup.py
    set ROCM_PATH=%ROCM_PATH%
    set GPU_TARGETS=%GPU_TARGETS%
    set SAGE_DEBUG_BUILD=%SAGE_DEBUG_BUILD%

    REM Build the extension
    echo Running: %PYTHON_EXE% setup.py build_ext --inplace
    %PYTHON_EXE% setup.py build_ext --inplace

    if errorlevel 1 (
        echo.
        echo ERROR: Build failed!
        echo.
        echo Troubleshooting tips:
        echo   1. Ensure ROCm 7.0+ is properly installed
        echo   2. Check that hipcc is in PATH
        echo   3. Verify PyTorch is installed: python -c "import torch; print(torch.__version__)"
        echo   4. Try running with debug: build.bat clean debug
        echo.
        exit /b 1
    )

    echo.
    echo Build successful!
    echo.
)

REM =========================
REM Install (if requested)
REM =========================

if %DO_INSTALL%==1 (
    echo [4/4] Installing Python package...

    REM Install in development mode
    echo Running: %PYTHON_EXE% -m pip install -e .
    %PYTHON_EXE% -m pip install -e .

    if errorlevel 1 (
        echo ERROR: Installation failed!
        exit /b 1
    )

    echo.
    echo Installation successful!
    echo.

    REM Verify installation
    echo Verifying installation...
    %PYTHON_EXE% -c "import sage_attention_rocm7; print('sage_attention_rocm7 imported successfully')"

    if errorlevel 1 (
        echo WARNING: Package import failed
        exit /b 1
    )
)

REM =========================
REM Test (if requested)
REM =========================

if %DO_TEST%==1 (
    echo [5/5] Running tests...

    REM Build and run C++ tests
    if exist "%BUILD_DIR%\test_int4_quantization.exe" (
        echo Running C++ tests...
        "%BUILD_DIR%\test_int4_quantization.exe"
        if errorlevel 1 (
            echo ERROR: C++ tests failed!
            exit /b 1
        )
    ) else (
        echo WARNING: C++ test executable not found
    )

    REM Run Python tests if they exist
    if exist "tests" (
        echo Running Python tests...
        %PYTHON_EXE% -m pytest tests -v
        if errorlevel 1 (
            echo ERROR: Python tests failed!
            exit /b 1
        )
    )

    echo.
    echo All tests passed!
    echo.
)

REM =========================
REM Completion
REM =========================

echo.
echo ============================================================
echo Build Complete!
echo ============================================================
echo.
echo To use the package:
echo   1. Activate your Python environment
echo   2. Import: python -c "import sage_attention_rocm7"
echo.
echo To install: build.bat install
echo To test:    build.bat test
echo To clean:   build.bat clean
echo ============================================================
echo.

endlocal
exit /b 0
