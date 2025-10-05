@echo off
REM Syntax check for INT4 quantization kernels

echo ========================================
echo Syntax Check: INT4 Quantization Kernels
echo ========================================
echo.

set HIPCC=D:\development\SageAttention\.venv\Scripts\hipcc.exe
set ARCH=gfx1151

echo Checking hip/quantization/int4_ops.hip.cpp...
%HIPCC% ^
    --offload-arch=%ARCH% ^
    -fsyntax-only ^
    -I. ^
    hip/quantization/int4_ops.hip.cpp

if %ERRORLEVEL% neq 0 (
    echo ERROR: Syntax errors found in int4_ops.hip.cpp
    exit /b 1
)

echo [OK] int4_ops.hip.cpp - No syntax errors
echo.

echo Checking test_int4_quantization.hip.cpp...
%HIPCC% ^
    --offload-arch=%ARCH% ^
    -fsyntax-only ^
    -I. ^
    test_int4_quantization.hip.cpp

if %ERRORLEVEL% neq 0 (
    echo ERROR: Syntax errors found in test_int4_quantization.hip.cpp
    exit /b 1
)

echo [OK] test_int4_quantization.hip.cpp - No syntax errors
echo.

echo ========================================
echo All syntax checks passed!
echo ========================================
