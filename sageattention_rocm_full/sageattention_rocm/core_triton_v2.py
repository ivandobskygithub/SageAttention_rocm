"""
Enhanced Triton core with optional HIP fused operations for ROCm 7
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import os

# Force ROCm backend for Triton
os.environ['TRITON_HIP_USE_ROCM'] = '1'

# Import fused operations (with fallbacks)
from .fused_ops import (
    quant_per_block_int8,
    sub_mean,
    prepare_qkv_tensors,
    get_backend_info
)

# Import Triton kernels
from .triton import (
    attn_qk_int8_per_block,
    attn_qk_int8_per_block_causal,
    attn_qk_int8_per_block_causal_varlen,
    attn_qk_int8_block_varlen,
)


def sageattn_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "NHD",
    is_causal: bool = False,
    smooth_k: bool = True,
    pv_accum_dtype: str = "fp32",
    use_fused_ops: bool = True,
    pad_to_multiple: Optional[int] = None
) -> torch.Tensor:
    """
    Optimized SageAttention using Triton kernels with optional HIP fused operations.

    This version combines:
    - Triton attention kernels (portable, work on all ROCm GPUs)
    - Optional HIP fused preprocessing (faster when available)
    - Automatic padding for better performance

    Args:
        q, k, v: Query, Key, Value tensors
        tensor_layout: "NHD" or "HND"
        is_causal: Whether to apply causal masking
        smooth_k: Whether to smooth keys (improves quantization)
        pv_accum_dtype: "fp32" or "fp16" accumulation
        use_fused_ops: Use HIP fused ops if available
        pad_to_multiple: Pad sequence to multiple (e.g., 64 for better perf)

    Returns:
        Attention output tensor
    """

    # Input validation
    assert tensor_layout in ["HND", "NHD"], f"Invalid tensor_layout: {tensor_layout}"
    assert pv_accum_dtype in ["fp32", "fp16"], f"Invalid pv_accum_dtype: {pv_accum_dtype}"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA/ROCm device"

    # Get original shape for reshaping at the end
    original_shape = q.shape
    original_layout = tensor_layout

    # Convert to consistent layout [B*H, S, D] for processing
    if tensor_layout == "NHD":
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        seq_len_q = q.shape[2]
        seq_len_kv = k.shape[2]
        head_dim = q.shape[3]

        # Reshape to [B*H, S, D]
        q = q.view(batch_size * num_heads, seq_len_q, head_dim)
        k = k.view(batch_size * num_heads, seq_len_kv, head_dim)
        v = v.view(batch_size * num_heads, seq_len_kv, head_dim)
    else:  # HND
        # Already in [H, S, D] format, treat as [B*H, S, D] where B=1
        num_heads = q.shape[0]
        seq_len_q = q.shape[1]
        seq_len_kv = k.shape[1]
        head_dim = q.shape[2]
        batch_size = 1

    # Validate dimensions
    assert head_dim % 16 == 0, f"Head dimension must be divisible by 16, got {head_dim}"

    # Preprocessing with fused operations
    if use_fused_ops and smooth_k:
        # Use fused mean subtraction if available
        k = sub_mean(k.clone())
    elif smooth_k:
        # Fallback to PyTorch
        k_mean = k.mean(dim=-1, keepdim=True)
        k = k - k_mean

    # Optional padding for better performance
    padded_seq_len = seq_len_q
    if pad_to_multiple and seq_len_q % pad_to_multiple != 0:
        padded_seq_len = ((seq_len_q + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

        # Pad tensors
        q_padded = torch.zeros(q.shape[0], padded_seq_len, head_dim,
                               dtype=q.dtype, device=q.device)
        q_padded[:, :seq_len_q] = q
        q = q_padded

        if seq_len_kv == seq_len_q:  # Self-attention
            k_padded = torch.zeros(k.shape[0], padded_seq_len, head_dim,
                                  dtype=k.dtype, device=k.device)
            k_padded[:, :seq_len_kv] = k
            k = k_padded

            v_padded = torch.zeros(v.shape[0], padded_seq_len, head_dim,
                                  dtype=v.dtype, device=v.device)
            v_padded[:, :seq_len_kv] = v
            v = v_padded

    # Quantize Q and K to INT8
    sm_scale = head_dim ** -0.5

    # Use fused quantization if available
    q_int8, q_scale = quant_per_block_int8(q, block_size=128)
    k_int8, k_scale = quant_per_block_int8(k, block_size=128)

    # Call Triton attention kernel
    if is_causal:
        output = attn_qk_int8_per_block_causal.forward(
            q_int8, k_int8, v,
            q_scale, k_scale,
            sm_scale,
            pv_accum_dtype == "fp32"
        )
    else:
        output = attn_qk_int8_per_block.forward(
            q_int8, k_int8, v,
            q_scale, k_scale,
            sm_scale,
            pv_accum_dtype == "fp32"
        )

    # Remove padding if applied
    if padded_seq_len != seq_len_q:
        output = output[:, :seq_len_q, :]

    # Reshape back to original layout
    if original_layout == "NHD":
        output = output.view(batch_size, num_heads, seq_len_q, head_dim)

    return output


def benchmark_backends(
    batch_size: int = 2,
    num_heads: int = 32,
    seq_len: int = 1024,
    head_dim: int = 128,
    num_iters: int = 100
):
    """
    Benchmark different backend combinations.
    """
    import time

    print("\nBenchmarking SageAttention ROCm Backends")
    print("=" * 60)

    # Get backend info
    info = get_backend_info()
    print(f"Device: {info.get('device', 'Unknown')}")
    print(f"Triton available: {info['triton']}")
    print(f"HIP fused ops available: {info['hip_fused']}")
    print(f"HIP validated: {info.get('hip_validated', False)}")
    print("=" * 60)

    # Create test tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                   dtype=torch.float16, device='cuda')

    configs = [
        ("Triton only", False, None),
        ("Triton + PyTorch fused", True, None),
        ("Triton + padding to 64", True, 64),
        ("Triton + padding to 128", True, 128),
    ]

    if info['hip_fused']:
        configs.append(("Triton + HIP fused ops", True, None))

    print(f"\nConfig: B={batch_size}, H={num_heads}, L={seq_len}, D={head_dim}")
    print(f"Iterations: {num_iters}")
    print("-" * 60)

    for name, use_fused, pad_multiple in configs:
        try:
            # Warmup
            for _ in range(3):
                _ = sageattn_optimized(
                    q, k, v,
                    use_fused_ops=use_fused,
                    pad_to_multiple=pad_multiple
                )
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(num_iters):
                _ = sageattn_optimized(
                    q, k, v,
                    use_fused_ops=use_fused,
                    pad_to_multiple=pad_multiple
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            avg_time = elapsed / num_iters * 1000  # ms
            tflops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim / (avg_time / 1000) / 1e12

            print(f"{name:30s}: {avg_time:7.2f} ms, {tflops:6.2f} TFLOPS")

        except Exception as e:
            print(f"{name:30s}: Failed - {e}")

    print("=" * 60)


if __name__ == "__main__":
    # Run benchmark when executed directly
    benchmark_backends()