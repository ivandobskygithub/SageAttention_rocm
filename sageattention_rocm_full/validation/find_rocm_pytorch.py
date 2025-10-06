"""
Find and test PyTorch installations with ROCm/AOTriton support
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_pytorch_installation(python_path=None):
    """Check a Python installation for ROCm PyTorch."""
    if python_path:
        cmd = [python_path, "-c"]
    else:
        cmd = [sys.executable, "-c"]

    check_script = """
import json
import sys
try:
    import torch
    info = {
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'hip_available': hasattr(torch.version, 'hip'),
        'has_aotriton': hasattr(torch, '_triton_scaled_dot_attention'),
        'is_rocm': 'rocm' in torch.__version__.lower(),
        'is_cpu_only': '+cpu' in torch.__version__.lower(),
        'python': sys.executable
    }
    if torch.cuda.is_available():
        info['device'] = torch.cuda.get_device_name(0)
    print(json.dumps(info))
except ImportError:
    print(json.dumps({'error': 'PyTorch not installed'}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
"""

    try:
        result = subprocess.run(
            cmd + [check_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout.strip())
    except Exception as e:
        return {'error': str(e)}

    return {'error': 'Failed to check'}


def find_python_environments():
    """Find Python environments that might have ROCm PyTorch."""
    environments = []

    # 1. Current Python
    current = check_pytorch_installation()
    if 'error' not in current:
        environments.append({
            'name': 'Current Python',
            'path': sys.executable,
            'info': current
        })

    # 2. Check conda environments
    try:
        result = subprocess.run(
            ['conda', 'env', 'list'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line and not line.startswith('#') and line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        env_name = parts[0]
                        env_path = parts[-1]
                        if os.path.exists(env_path):
                            python_path = os.path.join(env_path, 'python.exe')
                            if os.path.exists(python_path):
                                info = check_pytorch_installation(python_path)
                                if 'error' not in info:
                                    environments.append({
                                        'name': f'Conda: {env_name}',
                                        'path': python_path,
                                        'info': info
                                    })
    except:
        pass

    # 3. Check common virtual environment locations
    common_paths = [
        Path.home() / '.virtualenvs',
        Path.home() / 'envs',
        Path.cwd().parent / 'venv',
        Path.cwd().parent / '.venv',
        Path.cwd().parent.parent,
    ]

    for base_path in common_paths:
        if base_path.exists():
            for env_dir in base_path.iterdir():
                if env_dir.is_dir():
                    # Check for Scripts/python.exe (Windows)
                    python_path = env_dir / 'Scripts' / 'python.exe'
                    if not python_path.exists():
                        # Check for bin/python (Linux/Mac)
                        python_path = env_dir / 'bin' / 'python'

                    if python_path.exists():
                        info = check_pytorch_installation(str(python_path))
                        if 'error' not in info and info.get('version'):
                            environments.append({
                                'name': f'Venv: {env_dir.name}',
                                'path': str(python_path),
                                'info': info
                            })

    return environments


def main():
    print("=" * 70)
    print("Finding PyTorch Installations with ROCm Support")
    print("=" * 70)
    print()

    print("Searching for Python environments...")
    environments = find_python_environments()

    if not environments:
        print("No Python environments with PyTorch found.")
        return

    # Categorize environments
    rocm_envs = []
    aotriton_envs = []
    cpu_envs = []

    for env in environments:
        info = env['info']
        if info.get('cuda_available'):
            rocm_envs.append(env)
            if info.get('has_aotriton'):
                aotriton_envs.append(env)
        elif info.get('is_rocm') or info.get('has_aotriton'):
            if info.get('has_aotriton'):
                aotriton_envs.append(env)
            rocm_envs.append(env)
        else:
            cpu_envs.append(env)

    # Report findings
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    if aotriton_envs:
        print("\n[BEST] Environments with AOTriton support:")
        for env in aotriton_envs:
            info = env['info']
            print(f"\n  {env['name']}")
            print(f"    Path: {env['path']}")
            print(f"    PyTorch: {info['version']}")
            print(f"    CUDA: {info.get('cuda_available', False)}")
            print(f"    AOTriton: {info.get('has_aotriton', False)}")
            if info.get('device'):
                print(f"    Device: {info['device']}")

    if rocm_envs and not aotriton_envs:
        print("\n[GOOD] Environments with ROCm PyTorch:")
        for env in rocm_envs:
            if env not in aotriton_envs:
                info = env['info']
                print(f"\n  {env['name']}")
                print(f"    Path: {env['path']}")
                print(f"    PyTorch: {info['version']}")
                print(f"    CUDA: {info.get('cuda_available', False)}")

    if cpu_envs and not rocm_envs:
        print("\n[INFO] CPU-only environments:")
        for env in cpu_envs:
            info = env['info']
            print(f"\n  {env['name']}")
            print(f"    Path: {env['path']}")
            print(f"    PyTorch: {info['version']}")

    # Recommendations
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)

    if aotriton_envs:
        best = aotriton_envs[0]
        print(f"\n[SUCCESS] Use this environment for best performance:")
        print(f"   {best['path']}")
        print("\nTo run benchmarks with AOTriton:")
        print(f"  1. {best['path']} -m pip install -e .")
        print(f"  2. set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1")
        print(f"  3. {best['path']} benchmark_aotriton.py")

    elif rocm_envs:
        best = rocm_envs[0]
        print(f"\n[SUCCESS] Use this ROCm environment:")
        print(f"   {best['path']}")
        print("\nTo run benchmarks:")
        print(f"  1. {best['path']} -m pip install -e .")
        print(f"  2. {best['path']} benchmark_aotriton.py")

    else:
        print("\n[ERROR] No ROCm PyTorch found!")
        print("\nTo install PyTorch with ROCm 6.2:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2")

    # Create run script for best environment
    if aotriton_envs or rocm_envs:
        best_env = aotriton_envs[0] if aotriton_envs else rocm_envs[0]

        with open("run_with_rocm.bat", "w") as f:
            f.write("@echo off\n")
            f.write("REM Auto-generated script to run with ROCm PyTorch\n")
            f.write("set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1\n")
            f.write(f'"{best_env["path"]}" benchmark_aotriton.py\n')
            f.write("pause\n")

        print("\n[INFO] Created run_with_rocm.bat to run benchmarks with the best environment")


if __name__ == "__main__":
    main()