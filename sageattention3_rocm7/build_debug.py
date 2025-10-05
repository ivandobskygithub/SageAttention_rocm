"""
Build debug version of SageAttention ROCm DLL with extensive logging
"""
import os
import subprocess
import sys
from pathlib import Path


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


def build_debug_dll():
    """Build the debug DLL with extensive logging."""
    venv_path, rocm_path = setup_environment()

    # Paths
    hipcc = venv_path / "Scripts/hipcc.exe"
    if not hipcc.exists():
        print(f"ERROR: hipcc not found at {hipcc}")
        sys.exit(1)

    # Source files
    sources = [
        "hip/attention_forward_debug.hip.cpp",
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
        # Use short paths to avoid space issues with hipcc
        f"-L{windows_sdk_lib}",
        f"-L{msvc_lib}",
        f"-L{ucrt_lib}",
    ]

    # Compile command - Windows-specific flags
    compile_cmd = [
        str(hipcc),
        # Debug flags
        "-g",           # Debug symbols
        "-O0",          # No optimization
        "-DDEBUG",      # Enable debug macros
        "-v",           # Verbose output
        # Target architecture
        "--offload-arch=gfx1151",  # RDNA3.5
        # Windows DLL flags (no -fPIC on Windows)
        "-shared",
        # Warning flags
        "-Wall",
        "-Wextra",
        # Include paths
        *includes,
        # Source files
        *sources,
        # Library paths
        *lib_dirs,
        f"-L{rocm_path}/lib",  # Add main lib directory
        # Windows-specific libraries
        "-lamdhip64",  # Use amdhip64.lib not amdhip64_7.lib
        # Output
        "-o", "sage_attention_rocm7_debug.dll",
    ]

    print("\nCompile command:")
    print(" ".join(compile_cmd))
    print("\nCompiling...")

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            env=os.environ,
            check=False
        )

        # Print output regardless of success/failure
        if result.stdout:
            print("\n--- Compiler Output ---")
            print(result.stdout)

        if result.stderr:
            print("\n--- Compiler Warnings/Errors ---")
            print(result.stderr)

        if result.returncode != 0:
            print(f"\nERROR: Compilation failed with code {result.returncode}")
            sys.exit(1)

        print("\nSUCCESS: Debug DLL built successfully!")

        # Check the output
        output_dll = Path("sage_attention_rocm7_debug.dll")
        if output_dll.exists():
            size_kb = output_dll.stat().st_size / 1024
            print(f"Output: {output_dll} ({size_kb:.1f} KB)")
        else:
            print("ERROR: DLL not found after compilation!")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Compilation failed: {e}")
        sys.exit(1)


def test_dll():
    """Quick test of the debug DLL."""
    print("\n" + "=" * 60)
    print("Testing Debug DLL")
    print("=" * 60)

    import ctypes

    try:
        # Try to load the DLL
        dll = ctypes.CDLL("./sage_attention_rocm7_debug.dll")
        print("SUCCESS: DLL loaded")

        # Check for the debug function
        if hasattr(dll, "launch_sage_attention_forward_debug"):
            print("SUCCESS: Debug function found")
        else:
            print("WARNING: Debug function not found in exports")

    except OSError as e:
        print(f"ERROR: Could not load DLL: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Building SageAttention ROCm Debug DLL")
    print("=" * 60)

    build_debug_dll()
    test_dll()

    print("\n" + "=" * 60)
    print("Build Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update torch_integration.py to use sage_attention_rocm7_debug.dll")
    print("2. Run test_hip7_minimal.py to test with debug output")
    print("3. Check console output for detailed kernel diagnostics")