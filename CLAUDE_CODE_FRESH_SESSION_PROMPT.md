# Prompt for Fresh Claude Code Session - SageAttention3 RDNA Port

## Copy and paste this entire document into a new Claude Code session:

---

# Project: Port SageAttention3 to AMD RDNA GPUs using ROCm

## Critical Environment Information
- **Working Directory**: `D:\development\SageAttention`
- **Python Environment**: `.venv` contains ROCm 7.9.0rc20251005 (pre-release)
- **Target GPU**: gfx1151 (AMD RDNA3.5 iGPU - Radeon 890M)
- **Secondary Targets**: gfx1200/gfx1201 (RDNA4 for future optimization)
- **OS**: Windows 10/11
- **Python**: 3.13.5

## Pre-installed ROCm Tools in `.venv`
```
.venv\Scripts\hipcc.exe         # HIP compiler - VERIFIED WORKING for gfx1151
.venv\Scripts\amdclang++.exe    # AMD Clang++ - VERIFIED WORKING for gfx1151
.venv\Scripts\hipInfo.exe       # GPU detection tool
.venv\Scripts\rocm-sdk.exe      # ROCm SDK management

Packages:
- rocm 7.9.0rc20251005
- rocm-sdk-libraries-gfx1151 7.9.0rc20251005
- torch 2.10.0a0+rocm7.9.0rc20251005
- pytorch-triton-rocm 0.0.1
```

## Project Goal
Port SageAttention3's FP4 microscaling attention mechanism from NVIDIA CUDA/Blackwell to AMD ROCm/RDNA. The original uses FP4 quantization for extreme memory efficiency in LLM inference. We need to adapt this for RDNA3.5 (using INT4 emulation) and prepare for RDNA4 (native FP4 support).

**Key Constraint**: SageAttention3 is INFERENCE-ONLY (forward pass). No backward pass needed, which simplifies implementation significantly.

## Technical Architecture

### Original SageAttention3 (CUDA)
- FP4 quantization with microscaling
- Block-scaled quantization (16 elements per block)
- Per-block mean subtraction for stability
- Optimized for NVIDIA Blackwell (SM 12.0a)

### Our RDNA Implementation Strategy
- **RDNA3.5 (gfx1151)**: INT4 emulation since no native FP4
- **RDNA4 (gfx1200)**: Native FP4 when available
- Use WMMA (Wave Matrix Multiply Accumulate) instructions
- Leverage AMD Composable Kernel (CK) library for attention

## Sub-Agent System

You have 5 specialized sub-agents in `.claude/agents/`:

1. **rdna-kernel-agent.md** - HIP kernel development
2. **quantization-agent.md** - FP4/INT4 quantization
3. **ck-integration-agent.md** - Composable Kernel integration
4. **validation-agent.md** - Testing and benchmarking
5. **build-system-agent.md** - CMake and packaging

## Development Workflow

### Phase 1: Environment Validation (FIRST STEP - DO THIS!)
```bash
# 1. Verify ROCm installation
.venv\Scripts\hipcc.exe --version
.venv\Scripts\hipInfo.exe

# 2. Test compilation for gfx1151
echo "__global__ void test() {}" > test.hip.cpp
.venv\Scripts\hipcc.exe test.hip.cpp -o test.exe --offload-arch=gfx1151

# If successful, proceed to Phase 2
```

### Phase 2: Project Setup
```bash
# Create project structure
mkdir -p sageattention3_rocm7/{hip,ck,python,tests,benchmarks}
mkdir -p sageattention3_rocm7/hip/{include,quantization}

# Copy any existing CUDA files for reference
# Note: sageattention3_blackwell/ contains original CUDA implementation
```

### Phase 3: Implementation (USE SUB-AGENTS!)

#### Track 1: Kernel Development
```bash
# Invoke rdna-kernel-agent for HIP kernels
# Read .claude/agents/rdna-kernel-agent.md for context
# Create: sageattention3_rocm7/hip/attention_forward.hip.cpp

# Example implementation task:
"As rdna-kernel-agent, implement the basic attention forward kernel for gfx1151 with WMMA support"
```

#### Track 2: Quantization
```bash
# Invoke quantization-agent for INT4 operations
# Read .claude/agents/quantization-agent.md for context
# Create: sageattention3_rocm7/hip/quantization/int4_ops.hip.cpp

# Example implementation task:
"As quantization-agent, implement INT4 quantization with block scaling for RDNA3.5"
```

#### Track 3: CK Integration (OPTIONAL but recommended)
```bash
# Clone and build Composable Kernel
git clone https://github.com/ROCm/composable_kernel
cd composable_kernel
cmake -B build -D GPU_TARGETS=gfx1151 -D CMAKE_CXX_COMPILER=..\.venv\Scripts\hipcc.exe
cmake --build build

# Invoke ck-integration-agent
"As ck-integration-agent, create CK attention template for RDNA3.5"
```

### Phase 4: Build System
```bash
# Invoke build-system-agent
# Read .claude/agents/build-system-agent.md for context
# Create: CMakeLists.txt, setup.py

# Build command:
.venv\Scripts\python.exe setup.py build_ext --inplace
```

### Phase 5: Testing
```bash
# Invoke validation-agent
# Read .claude/agents/validation-agent.md for context
# Create and run tests

.venv\Scripts\python.exe -m pytest tests/
.venv\Scripts\python.exe benchmarks/benchmark_attention.py
```

## Compilation Commands Reference

### Always use these exact commands:
```bash
# For HIP files
.venv\Scripts\hipcc.exe input.hip.cpp -o output.exe --offload-arch=gfx1151

# For HIP with optimization
.venv\Scripts\hipcc.exe input.hip.cpp -o output.exe --offload-arch=gfx1151 -O3 -ffast-math

# Alternative with amdclang++
.venv\Scripts\amdclang++.exe -x hip input.cpp -o output.exe --offload-arch=gfx1151

# For multiple architectures (future-proofing)
.venv\Scripts\hipcc.exe input.hip.cpp -o output.exe --offload-arch=gfx1151,gfx1200,gfx1201
```

## Critical Implementation Details

### Memory Limits
- RDNA3.5 has 32KB LDS (shared memory) per CU
- Use tile size of 128x128 for attention
- Block scaling with 16 elements per group

### Quantization Strategy
```python
# For RDNA3.5 (gfx1151) - INT4 emulation
def quantize_int4_rdna35(tensor):
    # Scale per 16-element block
    blocks = tensor.reshape(-1, 16)
    scales = blocks.abs().max(dim=1).values / 7  # INT4 range [-8, 7]
    quantized = (blocks / scales.unsqueeze(1)).round().clamp(-8, 7)
    return quantized.to(torch.int8), scales.to(torch.float8_e4m3fn)
```

### Attention Computation
```python
# Forward pass only (no gradients needed!)
def attention_forward(Q, K, V, is_causal=False):
    # Q, K, V shape: [batch, heads, seq_len, head_dim]
    # 1. Quantize inputs to INT4
    # 2. Compute Q @ K^T with WMMA
    # 3. Apply softmax
    # 4. Compute result @ V
    # 5. Dequantize output
    return output
```

## Project Structure to Create

```
sageattention3_rocm7/
├── CMakeLists.txt
├── setup.py
├── README.md
├── hip/
│   ├── attention_forward.hip.cpp    # Main kernel (rdna-kernel-agent)
│   ├── quantization/
│   │   └── int4_ops.hip.cpp         # INT4 operations (quantization-agent)
│   └── include/
│       └── common.h                  # Shared definitions
├── ck/
│   └── ck_attention.cpp             # CK integration (ck-integration-agent)
├── python/
│   ├── __init__.py
│   └── sage_attention_rocm7.py      # Python API
├── tests/
│   └── test_attention.py            # Tests (validation-agent)
└── benchmarks/
    └── benchmark.py                  # Performance tests
```

## Expected Outcomes

### Minimum Success Criteria
- [ ] Compiles for gfx1151 using .venv tools
- [ ] Basic attention forward pass works
- [ ] INT4 quantization reduces memory by >70%
- [ ] Achieves >30 TFLOPS on gfx1151 (50% of peak)

### Stretch Goals
- [ ] CK integration working
- [ ] Python package installable via pip
- [ ] Performance within 70% of CUDA version
- [ ] Support for gfx1200/1201 ready

## Common Issues and Solutions

### Issue: "No ROCm GPU detected"
```bash
# Set override for iGPU
set HSA_OVERRIDE_GFX_VERSION=11.5.1
```

### Issue: "hipcc not found"
```bash
# Use full path
D:\development\SageAttention\.venv\Scripts\hipcc.exe
```

### Issue: "Compilation fails for gfx1151"
```bash
# Verify target architecture
.venv\Scripts\amdgpu-arch.exe  # Should show gfx1151 or compatible
```

## Key Documentation Locations
- Original CUDA implementation: `sageattention3_blackwell/`
- Sub-agent instructions: `.claude/agents/*.md`
- ROCm validation report: `ROCM_VALIDATION_REPORT.md`
- Feasibility study: `RDNA_UPDATED_FEASIBILITY.md`
- Implementation plan: `IMPLEMENTATION_PLAN_FINAL.md`

## START HERE:
1. Validate environment with Phase 1 commands
2. Create project structure
3. Implement basic attention kernel using rdna-kernel-agent context
4. Add INT4 quantization using quantization-agent context
5. Build and test using build-system-agent and validation-agent

Remember: This is FORWARD-PASS ONLY for inference. No training, no gradients, no backward pass needed!

---

## End of prompt. Begin implementation following the workflow above.