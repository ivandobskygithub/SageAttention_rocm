"""
Build production version of SageAttention ROCm DLL with all fixes
"""
import os
import subprocess
import sys
from pathlib import Path
import shutil


def setup_environment():
    """Setup environment to use venv ROCm only."""
    # Get venv path
    venv_path = Path("../.venv").resolve()
    if not venv_path.exists():
        print(f"ERROR: venv not found at {venv_path}")
        sys.exit(1)

    rocm_path = venv_path / "Lib/site-packages/_rocm_sdk_core"
    if not rocm_path.exists():
        print(f"ERROR: ROCm SDK not found at {rocm_path}")
        sys.exit(1)

    # Clear system ROCm
    for var in ["HIP_PATH", "ROCM_HOME", "ROCM_PATH", "HSA_PATH"]:
        if var in os.environ:
            print(f"Clearing system {var}: {os.environ[var]}")
            del os.environ[var]

    # Set venv ROCm
    os.environ["HIP_PATH"] = str(rocm_path)
    os.environ["ROCM_HOME"] = str(rocm_path)
    os.environ["HIP_PLATFORM"] = "amd"

    # Add to PATH
    dll_path = rocm_path / "lib/llvm/bin"
    os.environ["PATH"] = f"{dll_path};{os.environ.get('PATH', '')}"

    print(f"Using ROCm from: {rocm_path}")
    print(f"HIP_PATH set to: {os.environ['HIP_PATH']}")

    return venv_path, rocm_path


def build_production_dll():
    """Build the production DLL with optimizations."""
    venv_path, rocm_path = setup_environment()

    # Paths
    hipcc = venv_path / "Scripts/hipcc.exe"
    if not hipcc.exists():
        print(f"ERROR: hipcc not found at {hipcc}")
        sys.exit(1)

    # Source files - use the fixed version
    sources = [
        "hip/attention_forward_simple.hip.cpp",
        "hip/quantization/int4_ops.hip.cpp",
    ]

    # Include directories
    includes = [
        f"-I{rocm_path}/include",
        f"-I{rocm_path}/include/hip",
        "-Ihip",
    ]

    # Windows SDK and MSVC paths - use 8.3 short paths (no spaces!)
    windows_sdk_lib = r"C:\PROGRA~2\WI3CF2~1\10\Lib\100261~1.0\um\x64"
    msvc_lib = r"C:\PROGRA~2\MICROS~3\2022\BUILDT~1\VC\Tools\MSVC\1444~1.352\lib\x64"
    ucrt_lib = r"C:\PROGRA~2\WI3CF2~1\10\Lib\100261~1.0\ucrt\x64"

    # Library directories
    lib_dirs = [
        f"-L{rocm_path}/lib/llvm/lib",
        f"-L{rocm_path}/lib/rocm/lib",
        f"-L{windows_sdk_lib}",
        f"-L{msvc_lib}",
        f"-L{ucrt_lib}",
        f"-L{rocm_path}/lib",
    ]

    # Compile command - production flags
    compile_cmd = [
        str(hipcc),
        # Optimization flags
        "-O2",          # Optimization level 2
        "-DNDEBUG",     # Disable debug assertions
        # Target architecture
        "--offload-arch=gfx1151",  # RDNA3.5
        # C++ standard
        "-std=c++17",
        # Windows DLL flags
        "-shared",
        # Workarounds for ROCm issues
        "-Dmax=fmax",   # Workaround for complex builtins issue
        "-Dmin=fmin",
        # Include paths
        *includes,
        # Source files
        *sources,
        # Library paths
        *lib_dirs,
        # Windows-specific libraries
        "-lamdhip64",
        # Output
        "-o", "sage_attention_rocm7_prod.dll",
    ]

    print("\nCompile command:")
    print(" ".join(compile_cmd))
    print("\nCompiling production DLL...")

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            env=os.environ,
            check=False
        )

        # Print any warnings
        if result.stderr and "warning" in result.stderr.lower():
            print("\n--- Compiler Warnings ---")
            for line in result.stderr.split('\n'):
                if "warning" in line.lower():
                    print(line)

        if result.returncode != 0:
            print(f"\nERROR: Compilation failed with code {result.returncode}")
            if result.stderr:
                print("\n--- Full Compiler Output ---")
                print(result.stderr)
            if result.stdout:
                print("\n--- Stdout ---")
                print(result.stdout)
            sys.exit(1)

        print("\nSUCCESS: Production DLL built successfully!")

        # Check the output
        output_dll = Path("sage_attention_rocm7_prod.dll")
        if output_dll.exists():
            size_kb = output_dll.stat().st_size / 1024
            print(f"Output: {output_dll} ({size_kb:.1f} KB)")

            # Copy to package directory
            package_dll = Path("sageattention3_rocm7/sage_attention_rocm7.dll")
            shutil.copy2(output_dll, package_dll)
            print(f"Copied to: {package_dll}")

            # Also copy to alternative location
            alt_dll = Path("sage_attention_rocm7.dll")
            shutil.copy2(output_dll, alt_dll)
            print(f"Copied to: {alt_dll}")

        else:
            print("ERROR: DLL not found after compilation!")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Compilation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("Building SageAttention ROCm Production DLL")
    print("=" * 60)

    build_production_dll()

    print("\n" + "=" * 60)
    print("Build Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test with: python test_sageattention_package.py")
    print("2. Install package: pip install -e sageattention3_rocm7")
    print("3. Use in code: from sageattention3_rocm7 import sageattn3_rocm7")