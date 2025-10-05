"""
Build DLL for SageAttention3 ROCm7 on Windows
Uses simplified attention kernel without rocWMMA dependency
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent
VENV_PATH = PROJECT_ROOT.parent / ".venv"
HIPCC = VENV_PATH / "Scripts" / "hipcc.exe"
GPU_ARCH = "gfx1151"

# Source files (using simplified version without rocWMMA)
HIP_SOURCES = [
    "hip/attention_forward_simple.hip.cpp",
    "hip/quantization/int4_ops.hip.cpp",
]

# Include directories
INCLUDE_DIRS = [
    "hip/include",
]

# Add PyTorch includes if available
torch_include = VENV_PATH / "Lib" / "site-packages" / "torch" / "include"
if torch_include.exists():
    INCLUDE_DIRS.append(str(torch_include))
    torch_csrc = torch_include / "torch" / "csrc" / "api" / "include"
    if torch_csrc.exists():
        INCLUDE_DIRS.append(str(torch_csrc))

def compile_hip_file(src_path, obj_path):
    """Compile a single HIP source file"""
    # Windows-compatible compile flags
    cmd = [
        str(HIPCC),
        "-c",
        str(src_path),
        "-o", str(obj_path),
        f"--offload-arch={GPU_ARCH}",
        "-O3",
        "-std=c++17",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-DNOMINMAX",
        "-D__HIP_PLATFORM_AMD__",
    ]

    # Add include directories
    for inc_dir in INCLUDE_DIRS:
        inc_path = PROJECT_ROOT / inc_dir if not Path(inc_dir).is_absolute() else Path(inc_dir)
        if inc_path.exists():
            cmd.extend(["-I", str(inc_path)])

    print(f"Compiling {src_path.name}...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check result
    if result.returncode != 0:
        print(f"  [ERROR] Compilation failed with return code {result.returncode}")
        if result.stderr:
            print(f"  STDERR:\n{result.stderr}")
        return False

    if not obj_path.exists():
        print(f"  [ERROR] Object file was not created: {obj_path}")
        if result.stderr:
            print(f"  STDERR:\n{result.stderr}")
        return False

    size = obj_path.stat().st_size
    if size == 0:
        print(f"  [ERROR] Object file is empty")
        return False

    print(f"  [OK] Created {obj_path.name} ({size:,} bytes)")
    if result.stderr and "warning" in result.stderr.lower():
        print(f"  Warnings:\n{result.stderr}")

    return True

def find_windows_sdk():
    """Find Windows SDK library paths"""
    sdk_bases = [
        Path("C:/Program Files (x86)/Windows Kits/10/Lib"),
        Path("C:/Program Files/Windows Kits/10/Lib"),
    ]

    for sdk_base in sdk_bases:
        if sdk_base.exists():
            # Find latest version
            for ver in sorted(sdk_base.iterdir(), reverse=True):
                if ver.is_dir():
                    um_path = ver / "um" / "x64"
                    ucrt_path = ver / "ucrt" / "x64"
                    if um_path.exists() and ucrt_path.exists():
                        return [str(um_path), str(ucrt_path)]

    return []

def link_dll(obj_files, dll_path):
    """Link object files into DLL"""
    # Delete old DLL if exists
    if dll_path.exists():
        dll_path.unlink()

    # Link command
    cmd = [
        str(HIPCC),
        "-shared",
        "-o", str(dll_path),
        f"--offload-arch={GPU_ARCH}",
    ]

    # Add object files
    cmd.extend([str(f) for f in obj_files])

    # Add ROCm library path
    rocm_lib = VENV_PATH / "Lib" / "site-packages" / "_rocm_sdk_core" / "lib"
    if rocm_lib.exists():
        cmd.extend(["-L", str(rocm_lib)])

    # Add Windows SDK library paths using /LIBPATH for lld-link
    sdk_paths = find_windows_sdk()
    if sdk_paths:
        print(f"  Found Windows SDK:")
        for p in sdk_paths:
            print(f"    {p}")
            # Quote the entire argument to prevent splitting on spaces
            cmd.extend(["-Xlinker", f'"/LIBPATH:{p}"'])
    else:
        print("  [WARNING] Windows SDK not found, linking may fail")

    # Add ROCm lib path using /LIBPATH
    if rocm_lib.exists():
        cmd.extend(["-Xlinker", f'"/LIBPATH:{rocm_lib}"'])

    # Link libraries using full path or explicit /DEFAULTLIB
    cmd.extend(["-Xlinker", '"/DEFAULTLIB:amdhip64.lib"'])
    cmd.extend(["-Xlinker", '"/DEFAULTLIB:kernel32.lib"'])
    cmd.extend(["-Xlinker", '"/DEFAULTLIB:uuid.lib"'])

    print(f"\nLinking DLL...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"  Return code: {result.returncode}")
    if result.stdout:
        print(f"  STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"  STDERR:\n{result.stderr}")

    # Verify DLL exists
    if dll_path.exists():
        size = dll_path.stat().st_size
        print(f"  [OK] DLL created: {dll_path.name} ({size:,} bytes)")
        return True
    else:
        print(f"  [ERROR] DLL was not created")
        return False

def check_exports(dll_path):
    """Check DLL exports using dumpbin"""
    print(f"\nChecking DLL exports...")
    try:
        result = subprocess.run(['where', 'dumpbin'], capture_output=True, text=True)
        if result.returncode != 0:
            print("  dumpbin not found, skipping export check")
            return

        dumpbin = result.stdout.strip().split('\n')[0]
        result = subprocess.run(
            [dumpbin, '/EXPORTS', str(dll_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Parse exports
            lines = result.stdout.split('\n')
            exports = []
            in_exports = False
            for line in lines:
                if 'ordinal' in line.lower() and 'name' in line.lower():
                    in_exports = True
                    continue
                if in_exports and line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        exports.append(parts[3])

            print(f"  Found {len(exports)} exported functions:")
            for exp in exports:
                print(f"    - {exp}")

            # Check for required functions
            required = [
                'launch_sage_attention_forward',
                'launch_quantize_int4',
                'launch_dequantize_int4',
                'launch_quantize_int4_transpose'
            ]
            missing = [r for r in required if r not in exports]
            if missing:
                print(f"\n  [WARNING] Missing required exports: {missing}")
            else:
                print(f"\n  [OK] All required exports present")

    except Exception as e:
        print(f"  Could not check exports: {e}")

def main():
    print("="*70)
    print("SageAttention3 ROCm7 Windows DLL Build")
    print("="*70)
    print(f"Project: {PROJECT_ROOT}")
    print(f"HIPCC: {HIPCC}")
    print(f"Target: {GPU_ARCH}")
    print()

    if not HIPCC.exists():
        print(f"ERROR: hipcc not found at {HIPCC}")
        return 1

    # Compile sources
    print("COMPILATION PHASE")
    print("-"*70)
    obj_files = []
    for src in HIP_SOURCES:
        src_path = PROJECT_ROOT / src
        if not src_path.exists():
            print(f"ERROR: Source file not found: {src_path}")
            return 1

        obj_path = src_path.with_suffix(".o")
        if not compile_hip_file(src_path, obj_path):
            print(f"\nBuild failed during compilation of {src}")
            return 1

        obj_files.append(obj_path)

    # Link DLL
    print()
    print("LINKING PHASE")
    print("-"*70)
    dll_path = PROJECT_ROOT / "sage_attention_rocm7.dll"
    if not link_dll(obj_files, dll_path):
        print("\nBuild failed during linking")
        return 1

    # Check exports
    check_exports(dll_path)

    # Summary
    print()
    print("="*70)
    print("BUILD SUCCESSFUL")
    print("="*70)
    print(f"DLL: {dll_path}")
    print(f"Size: {dll_path.stat().st_size:,} bytes")
    print()
    print("Next steps:")
    print("  1. Test DLL loading with Python ctypes")
    print("  2. Create PyTorch binding")
    print("  3. Test with ComfyUI")
    print("="*70)

    # Clean up object files
    print("\nCleaning up object files...")
    for obj in obj_files:
        if obj.exists():
            obj.unlink()
            print(f"  Removed {obj.name}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
