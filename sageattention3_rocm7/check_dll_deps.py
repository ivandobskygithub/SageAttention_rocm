"""
Check DLL dependencies using subprocess
"""

import subprocess
from pathlib import Path

dll_path = Path(__file__).parent / "sage_attention_rocm7.dll"
dumpbin = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe"

print("Checking DLL dependencies...")
print(f"DLL: {dll_path}")
print()

result = subprocess.run(
    [dumpbin, '/DEPENDENTS', str(dll_path)],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
