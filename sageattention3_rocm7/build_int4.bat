@echo off
REM Build script for INT4 quantization kernels on Windows

echo ===================================
echo Building INT4 Quantization Kernels
echo ===================================
echo.

set HIPCC=D:\development\SageAttention\.venv\Scripts\hipcc.exe
set BUILD_DIR=build_int4
set ARCH=gfx1151

echo Using HIP compiler: %HIPCC%
echo Target architecture: %ARCH%
echo.

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM Compile INT4 operations library
echo Compiling int4_ops.hip.cpp...
%HIPCC% ^
    --offload-arch=%ARCH% ^
    -O3 ^
    -ffast-math ^
    -munsafe-fp-atomics ^
    -I. ^
    -c hip/quantization/int4_ops.hip.cpp ^
    -o %BUILD_DIR%/int4_ops.o

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to compile int4_ops.hip.cpp
    exit /b 1
)

echo Successfully compiled int4_ops.hip.cpp
echo.

REM Compile test program
echo Compiling test_int4_quantization.hip.cpp...
%HIPCC% ^
    --offload-arch=%ARCH% ^
    -O3 ^
    -ffast-math ^
    -I. ^
    %BUILD_DIR%/int4_ops.o ^
    test_int4_quantization.hip.cpp ^
    -o %BUILD_DIR%/test_int4_quantization.exe

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to compile test_int4_quantization.hip.cpp
    exit /b 1
)

echo Successfully compiled test_int4_quantization.hip.cpp
echo.

echo ===================================
echo Build completed successfully!
echo ===================================
echo.
echo To run the test:
echo   cd %BUILD_DIR%
echo   test_int4_quantization.exe
echo.
