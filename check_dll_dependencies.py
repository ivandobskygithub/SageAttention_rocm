"""
Check DLL dependencies and loaded libraries for SageAttention3 ROCm7
"""
import os
import sys
import ctypes
import subprocess
from pathlib import Path

def check_dll_dependencies():
    """Check which DLLs are loaded by sage_attention_rocm7.dll"""

    dll_path = Path("sageattention3_rocm7/sage_attention_rocm7.dll")
    if not dll_path.exists():
        print(f"DLL not found at {dll_path}")
        return

    print("=" * 70)
    print("DLL Dependency Analysis")
    print("=" * 70)

    # Method 1: Use dumpbin if available
    try:
        result = subprocess.run(
            ["dumpbin", "/dependents", str(dll_path)],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            print("\nDumpbin output:")
            print(result.stdout)
    except:
        print("Dumpbin not available (need Visual Studio tools)")

    # Method 2: Try to load and check what gets loaded
    print("\n" + "=" * 70)
    print("Loading DLL and checking loaded modules")
    print("=" * 70)

    # Pre-set environment to use venv libraries
    venv_path = Path(".venv").resolve()
    rocm_dll_path = venv_path / "Lib/site-packages/_rocm_sdk_core/lib/llvm/bin"

    print(f"\nVenv ROCm path: {rocm_dll_path}")
    print(f"Path exists: {rocm_dll_path.exists()}")

    if rocm_dll_path.exists():
        # List HIP-related DLLs in venv
        print("\nHIP DLLs in venv:")
        for dll in rocm_dll_path.glob("*hip*.dll"):
            print(f"  - {dll.name} ({dll.stat().st_size / 1024:.0f} KB)")
        for dll in rocm_dll_path.glob("*rocm*.dll"):
            print(f"  - {dll.name} ({dll.stat().st_size / 1024:.0f} KB)")

    # Set environment
    os.environ["PATH"] = str(rocm_dll_path) + ";" + os.environ.get("PATH", "")

    # Try to load DLL
    print("\n" + "=" * 70)
    print("Attempting to load DLL")
    print("=" * 70)

    try:
        # First, try to pre-load HIP libraries
        hip_dll = rocm_dll_path / "amdhip64_7.dll"
        if hip_dll.exists():
            print(f"Pre-loading {hip_dll.name}...")
            hip_handle = ctypes.CDLL(str(hip_dll))
            print(f"  SUCCESS: Handle = {hip_handle}")

        # Now load our DLL
        print(f"\nLoading {dll_path}...")
        sage_dll = ctypes.CDLL(str(dll_path.resolve()))
        print(f"  SUCCESS: Handle = {sage_dll}")

        # Check loaded modules using psutil if available
        try:
            import psutil
            process = psutil.Process()

            print("\n" + "=" * 70)
            print("Currently loaded HIP/ROCm modules:")
            print("=" * 70)

            hip_modules = []
            for dll_info in process.memory_maps():
                path = dll_info.path
                if any(x in path.lower() for x in ["hip", "rocm", "amd", "sage"]):
                    hip_modules.append(path)

            for module in sorted(set(hip_modules)):
                if "venv" in module:
                    print(f"  [VENV] {module}")
                elif "Windows" in module:
                    print(f"  [SYSTEM] {module}")
                else:
                    print(f"  [OTHER] {module}")

        except ImportError:
            print("\npsutil not available - install with: pip install psutil")

    except OSError as e:
        print(f"  FAILED: {e}")
        print("\nPossible issues:")
        print("  - Missing amdhip64_7.dll")
        print("  - Wrong architecture (32 vs 64 bit)")
        print("  - Missing dependencies")

    print("\n" + "=" * 70)
    print("Environment Variables")
    print("=" * 70)

    env_vars = ["HIP_PATH", "ROCM_HOME", "HIP_PLATFORM", "AMD_COMGR_ACTION_FILE_PATH"]
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"{var}: {value}")


def check_pytorch_rocm():
    """Check PyTorch ROCm configuration"""
    print("\n" + "=" * 70)
    print("PyTorch ROCm Configuration")
    print("=" * 70)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'Unknown'}")

        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"Device capability: {torch.cuda.get_device_capability(0)}")

            # Get current stream
            stream = torch.cuda.current_stream()
            print(f"Current stream: {stream}")
            print(f"Stream pointer: {stream.cuda_stream:x}")

    except Exception as e:
        print(f"Error checking PyTorch: {e}")


if __name__ == "__main__":
    check_dll_dependencies()
    check_pytorch_rocm()