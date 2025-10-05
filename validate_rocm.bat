@echo off
call .venv\Scripts\activate
echo.
echo ========================================
echo Validating ROCm Environment
echo ========================================
echo.

echo Python executable: %VIRTUAL_ENV%\Scripts\python.exe
python --version
echo.

echo Installed ROCm packages:
pip list | findstr /i "rocm hip torch"
echo.

echo Testing ROCm Python module:
python test_rocm_env.py
echo.

echo Checking HIP compiler:
where hipcc 2>nul
if %errorlevel%==0 (
    echo HIP compiler found in PATH
    hipcc --version
) else (
    echo Trying venv HIP compiler...
    if exist ".venv\Scripts\hipcc.exe" (
        .venv\Scripts\hipcc.exe --version
    ) else (
        echo HIP compiler not found
    )
)
echo.

echo Checking AMD compiler:
where amdclang 2>nul
if %errorlevel%==0 (
    echo AMD Clang found in PATH
    amdclang --version
) else (
    echo Trying venv AMD Clang...
    if exist ".venv\Scripts\amdclang.exe" (
        .venv\Scripts\amdclang.exe --version
    ) else (
        echo AMD Clang not found
    )
)
echo.

echo Environment variables:
echo ROCM_HOME=%ROCM_HOME%
echo ROCM_PATH=%ROCM_PATH%
echo HIP_PATH=%HIP_PATH%
echo HSA_OVERRIDE_GFX_VERSION=%HSA_OVERRIDE_GFX_VERSION%
echo.

pause