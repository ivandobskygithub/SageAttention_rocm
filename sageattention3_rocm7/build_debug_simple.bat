@echo off
REM Build debug version of sage_attention_rocm7_debug.dll

set HIPCC=D:\development\SageAttention\.venv\Scripts\hipcc.exe
set WINSDK=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0
set ROCM_INC=D:\development\SageAttention\.venv\Lib\site-packages\_rocm_sdk_core\include
set ROCM_LIB=D:\development\SageAttention\.venv\Lib\site-packages\_rocm_sdk_core\lib

echo Building debug DLL...

%HIPCC% -std=c++17 -Dmax=fmax -Dmin=fmin ^
  -g -O0 -DSAGE_DEBUG_BUILD=1 -D__HIP_PLATFORM_AMD__ ^
  --offload-arch=gfx1151 ^
  -shared -o sage_attention_rocm7_debug.dll ^
  hip/attention_forward_simple.hip.cpp ^
  -I"%ROCM_INC%" ^
  -L"%ROCM_LIB%" ^
  -Xlinker /libpath:"%WINSDK%\um\x64" ^
  -Xlinker /libpath:"%WINSDK%\ucrt\x64" ^
  -lamdhip64

if %ERRORLEVEL% EQU 0 (
  echo Build successful!
  dir sage_attention_rocm7_debug.dll
) else (
  echo Build failed with error code %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)
