# HIP Kernel Debug and Resolution Plan

## Problem Statement
The SageAttention3 ROCm7 attention kernels are failing with "HIP error: invalid argument" while quantization functions work correctly. This indicates the DLL loads but kernel execution fails.

## Root Cause Analysis Plan

### Phase 1: Library Dependency Verification
**Goal**: Ensure DLL uses ONLY venv HIP libraries, not system installations

#### 1.1 Check Current DLL Dependencies
```bash
# Use dumpbin or Dependencies.exe to check what DLLs are loaded
dumpbin /dependents sage_attention_rocm7.dll

# Or use Python to check loaded modules
python -c "import ctypes; dll = ctypes.CDLL('sage_attention_rocm7.dll'); import psutil; proc = psutil.Process(); print([m.path for m in proc.memory_maps() if 'hip' in m.path.lower()])"
```

#### 1.2 Force Venv Library Loading
```python
# Update torch_integration.py to explicitly load venv libraries first
import os
import ctypes

def _load_venv_hip_libraries(venv_path):
    """Pre-load HIP libraries from venv to prevent system library conflicts."""
    hip_paths = [
        venv_path / "Lib/site-packages/_rocm_sdk_core/lib/llvm/bin/amdhip64_7.dll",
        venv_path / "Lib/site-packages/_rocm_sdk_core/lib/rocm/bin/hiprtc.dll",
        venv_path / "Lib/site-packages/_rocm_sdk_core/lib/rocm/bin/hipblas.dll",
    ]

    for hip_dll in hip_paths:
        if hip_dll.exists():
            ctypes.CDLL(str(hip_dll), mode=ctypes.RTLD_GLOBAL)
            print(f"Pre-loaded: {hip_dll}")
```

#### 1.3 Environment Variable Setup
```python
# Set environment before ANY imports
os.environ["HIP_PATH"] = str(venv_path / "Lib/site-packages/_rocm_sdk_core")
os.environ["ROCM_HOME"] = str(venv_path / "Lib/site-packages/_rocm_sdk_core")
os.environ["AMD_COMGR_ACTION_FILE_PATH"] = str(venv_path / "Lib/site-packages/_rocm_sdk_core/lib/llvm/lib")
os.environ["PATH"] = str(venv_path / "Lib/site-packages/_rocm_sdk_core/lib/llvm/bin") + ";" + os.environ["PATH"]
```

### Phase 2: Kernel Argument Debugging

#### 2.1 Add Validation in C++ DLL
```cpp
// attention_forward_simple.hip.cpp - Add validation
extern "C" void launch_sage_attention_forward(
    const void* Q, const void* K, const void* V, void* O,
    const void* delta_s,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int head_dim, float scale, bool is_causal, hipStream_t stream)
{
    // Add validation
    printf("=== Kernel Launch Debug ===\n");
    printf("Q ptr: %p\n", Q);
    printf("K ptr: %p\n", K);
    printf("V ptr: %p\n", V);
    printf("O ptr: %p\n", O);
    printf("Batch: %d, Heads: %d, SeqQ: %d, SeqK: %d, HeadDim: %d\n",
           batch_size, num_heads, seq_len_q, seq_len_k, head_dim);
    printf("Scale: %f, Causal: %d\n", scale, is_causal);
    printf("Stream: %p\n", stream);

    // Check pointer alignment
    if ((uintptr_t)Q % 16 != 0) {
        printf("ERROR: Q not 16-byte aligned!\n");
    }

    // Validate dimensions
    if (batch_size <= 0 || num_heads <= 0 || seq_len_q <= 0 ||
        seq_len_k <= 0 || head_dim <= 0) {
        printf("ERROR: Invalid dimensions!\n");
        return;
    }

    // Check stream validity
    if (stream == nullptr) {
        printf("WARNING: NULL stream, using default\n");
        hipStreamCreate(&stream);
    }

    // Test simple kernel first
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Pre-existing HIP error: %s\n", hipGetErrorString(err));
        hipGetLastError(); // Clear it
    }
}
```

#### 2.2 Create Minimal Test Kernel
```cpp
// test_kernel.hip.cpp
__global__ void test_kernel(float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = idx * 1.0f;
    }
}

extern "C" void test_hip_kernel(void* output, int size, hipStream_t stream) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    printf("Launching test kernel: grid=%d, block=%d, size=%d\n",
           grid.x, block.x, size);

    hipLaunchKernelGGL(test_kernel, grid, block, 0, stream,
                       (float*)output, size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("Test kernel failed: %s\n", hipGetErrorString(err));
    } else {
        printf("Test kernel launched successfully!\n");
    }
}
```

### Phase 3: Python-side Debugging

#### 3.1 Enhanced Error Checking
```python
# torch_integration.py updates
def attention_forward(self, Q, K, V, scale=None, is_causal=False, delta_s=None):
    # Add pre-checks
    print(f"[DEBUG] Q shape: {Q.shape}, dtype: {Q.dtype}, device: {Q.device}")
    print(f"[DEBUG] Q contiguous: {Q.is_contiguous()}, ptr: {Q.data_ptr():x}")
    print(f"[DEBUG] Q stride: {Q.stride()}")

    # Ensure contiguous memory
    if not Q.is_contiguous():
        print("[DEBUG] Making Q contiguous")
        Q = Q.contiguous()

    # Check alignment
    if Q.data_ptr() % 16 != 0:
        print(f"[WARNING] Q not 16-byte aligned: {Q.data_ptr():x}")

    # Get stream explicitly
    stream = torch.cuda.current_stream(self.device)
    stream_ptr = stream.cuda_stream
    print(f"[DEBUG] Stream ptr: {stream_ptr:x}")

    # Set environment for debugging
    os.environ["AMD_SERIALIZE_KERNEL"] = "3"
    os.environ["AMD_LOG_LEVEL"] = "7"

    try:
        self.dll.launch_sage_attention_forward(...)

        # Check for errors immediately
        torch.cuda.synchronize()

    except Exception as e:
        print(f"[ERROR] Kernel launch failed: {e}")

        # Try to get more info
        import subprocess
        result = subprocess.run(["hipcc", "--version"], capture_output=True, text=True)
        print(f"[DEBUG] HIP version: {result.stdout}")
```

#### 3.2 Test with Simple Cases
```python
# test_minimal.py
def test_minimal_attention():
    """Test with smallest possible inputs."""
    import torch
    from sageattention3_rocm7 import SageAttentionROCm

    sage = SageAttentionROCm()

    # Start with tiny tensors
    test_cases = [
        (1, 1, 16, 16),   # Minimal case
        (1, 1, 128, 64),  # Single head, padded
        (1, 8, 128, 64),  # Multi-head
        (2, 8, 256, 64),  # Batch
    ]

    for b, h, s, d in test_cases:
        print(f"\nTesting {b}x{h}x{s}x{d}...")

        Q = torch.ones((b, h, s, d), dtype=torch.float16, device="cuda")
        K = torch.ones((b, h, s, d), dtype=torch.float16, device="cuda")
        V = torch.ones((b, h, s, d), dtype=torch.float16, device="cuda")

        try:
            output = sage.attention_forward(Q, K, V)
            print(f"  SUCCESS: Output shape {output.shape}")
        except Exception as e:
            print(f"  FAILED: {e}")

            # Try with default stream
            torch.cuda.set_stream(torch.cuda.default_stream())
            try:
                output = sage.attention_forward(Q, K, V)
                print(f"  SUCCESS with default stream")
            except:
                pass
```

### Phase 4: Build System Fixes

#### 4.1 Update build_dll.py
```python
# build_dll.py modifications
def build_hip_dll():
    # Use ONLY venv paths
    venv_path = Path(".venv").resolve()
    rocm_path = venv_path / "Lib/site-packages/_rocm_sdk_core"

    # Set environment
    env = os.environ.copy()
    env["HIP_PATH"] = str(rocm_path)
    env["ROCM_HOME"] = str(rocm_path)
    env["PATH"] = f"{rocm_path}/lib/llvm/bin;{env['PATH']}"

    # Compile with debug info
    compile_cmd = [
        str(venv_path / "Scripts/hipcc.exe"),
        "-g",  # Debug symbols
        "-O0",  # No optimization for debugging
        "-DDEBUG",  # Enable debug prints
        "-fPIC",
        "-shared",
        "-o", "sage_attention_rocm7_debug.dll",
        "attention_forward_simple.hip.cpp",
        "-L", str(rocm_path / "lib/rocm/lib"),
        "-lhipblas",
        "-lrocblas",
    ]

    print(f"Compile command: {' '.join(compile_cmd)}")
    subprocess.run(compile_cmd, env=env, check=True)
```

### Phase 5: Testing Strategy

#### 5.1 Progressive Testing
1. Test DLL loading with explicit venv paths
2. Test minimal kernel (just copy data)
3. Test with 1x1x16x16 tensors
4. Gradually increase complexity
5. Compare with working quantization kernels

#### 5.2 Verification Checklist
- [ ] DLL uses only venv HIP libraries
- [ ] No system ROCm libraries loaded
- [ ] Stream pointers are valid
- [ ] Tensor pointers are aligned
- [ ] Tensor layouts match expectations
- [ ] Kernel grid/block dimensions valid
- [ ] HIP error checking after each call

### Phase 6: Implementation Steps

1. **Immediate Actions**:
   - Add library pre-loading to ensure venv libraries used
   - Add debug logging to C++ DLL
   - Create minimal test kernel

2. **Debug Build**:
   - Rebuild DLL with debug symbols
   - Add validation checks
   - Test with simple kernel first

3. **Progressive Fix**:
   - Fix library loading issues
   - Fix tensor layout/alignment
   - Fix kernel arguments
   - Optimize once working

### Expected Outcomes

After implementing this plan:
1. Identify exact cause of "invalid argument" error
2. Ensure ONLY venv HIP libraries are used
3. Get attention kernels working
4. Achieve full API compatibility
5. Performance optimization (secondary goal)

### Success Criteria

- Attention kernels execute without errors
- Results match PyTorch reference within tolerance
- No system ROCm libraries interfere
- All test cases pass