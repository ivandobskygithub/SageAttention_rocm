"""
Fused operations for SageAttention ROCm
Falls back to Triton if HIP kernels not available
"""

import torch
from typing import Optional, Tuple

# Try to import HIP kernels
HAS_HIP_FUSED = False
try:
    import sageattention_rocm._fused_hip as fused_hip
    HAS_HIP_FUSED = True
    print("Using HIP fused kernels")
except ImportError:
    print("HIP fused kernels not available, using Triton fallbacks")

# Import Triton fallbacks
from .triton import quant_per_block, quant_per_thread


def quant_per_block_int8(
    input: torch.Tensor,
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8 with per-block scaling.

    Args:
        input: Input tensor [B, H, S, D] or [B*H, S, D]
        block_size: Size of quantization blocks

    Returns:
        Quantized INT8 tensor and scale factors
    """

    if HAS_HIP_FUSED:
        # Use HIP kernel
        return fused_hip.quant_per_block_int8(input, block_size)
    else:
        # Use Triton fallback
        return quant_per_block.quantize(input, block_size)


def sub_mean(input: torch.Tensor) -> torch.Tensor:
    """
    Subtract per-sequence mean from tensor.

    Args:
        input: Input tensor [B, S, D] or [B, H, S, D]

    Returns:
        Mean-subtracted tensor
    """

    if HAS_HIP_FUSED:
        # Use HIP kernel
        return fused_hip.sub_mean(input)
    else:
        # Triton fallback - simple PyTorch implementation
        if input.dim() == 3:
            mean = input.mean(dim=-1, keepdim=True)
        else:  # 4D
            mean = input.mean(dim=-1, keepdim=True)
        return input - mean


def transpose_pad_permute(
    input: torch.Tensor,
    padded_seq_len: int
) -> torch.Tensor:
    """
    Transpose, pad, and permute tensor for optimal memory layout.

    Args:
        input: Input tensor [B, H, S, D]
        padded_seq_len: Target sequence length with padding

    Returns:
        Permuted tensor [B, S_padded, H, D]
    """

    if HAS_HIP_FUSED:
        # Use HIP kernel
        return fused_hip.transpose_pad_permute(input, padded_seq_len)
    else:
        # PyTorch fallback
        B, H, S, D = input.shape

        # Create padded output
        output = torch.zeros(B, padded_seq_len, H, D,
                           dtype=input.dtype, device=input.device)

        # Transpose and copy
        # [B, H, S, D] -> [B, S, H, D]
        input_transposed = input.transpose(1, 2)
        output[:, :S, :, :] = input_transposed

        return output


def quant_and_smooth_int8(
    input: torch.Tensor,
    smooth: bool = True,
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Combined quantization with optional smoothing.

    Args:
        input: Input tensor
        smooth: Whether to apply smoothing (mean subtraction)
        block_size: Quantization block size

    Returns:
        Quantized tensor, scales, and optional mean values
    """

    mean_vals = None

    if smooth:
        # Subtract mean first for better quantization
        original_input = input
        input = sub_mean(input.clone())
        mean_vals = original_input.mean(dim=-1, keepdim=True)

    # Quantize
    quantized, scales = quant_per_block_int8(input, block_size)

    return quantized, scales, mean_vals


def prepare_qkv_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    smooth_k: bool = True,
    pad_to_multiple: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare Q, K, V tensors for attention computation.

    Args:
        q, k, v: Query, Key, Value tensors
        smooth_k: Whether to smooth keys
        pad_to_multiple: Pad sequence length to multiple of this

    Returns:
        Processed Q, K, V tensors
    """

    # Apply smoothing to keys if requested
    if smooth_k:
        k = sub_mean(k.clone())

    # Pad if requested
    if pad_to_multiple:
        B, H, S, D = q.shape
        padded_S = ((S + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

        if padded_S > S:
            q = transpose_pad_permute(q, padded_S)
            k = transpose_pad_permute(k, padded_S)
            v = transpose_pad_permute(v, padded_S)

            # Transpose back to [B, H, S_padded, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

    return q, k, v


# Utility functions for testing
def validate_hip_kernels() -> bool:
    """Check if HIP kernels are available and working."""
    if not HAS_HIP_FUSED:
        return False

    try:
        # Test with small tensor
        test_tensor = torch.randn(2, 8, 64, 64, device='cuda', dtype=torch.float16)
        _ = quant_per_block_int8(test_tensor)
        return True
    except Exception as e:
        print(f"HIP kernel validation failed: {e}")
        return False


def get_backend_info() -> dict:
    """Get information about available backends."""
    info = {
        'triton': True,  # Always available
        'hip_fused': HAS_HIP_FUSED,
        'hip_validated': validate_hip_kernels() if HAS_HIP_FUSED else False,
    }

    # Add device info
    if torch.cuda.is_available():
        info['device'] = torch.cuda.get_device_properties(0).name
        info['device_count'] = torch.cuda.device_count()
    else:
        info['device'] = 'CPU'
        info['device_count'] = 0

    return info


# Print backend status on import
if __name__ != "__main__":
    backend_info = get_backend_info()
    print(f"SageAttention ROCm Fused Ops: Triton={'✓' if backend_info['triton'] else '✗'}, "
          f"HIP={'✓' if backend_info['hip_fused'] else '✗'}")