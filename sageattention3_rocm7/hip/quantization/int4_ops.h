/*
 * INT4 Quantization Operations Header
 * API declarations for INT4 quantization/dequantization kernels
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Quantize FP16 tensor to INT4 with block-scaled symmetric quantization
 *
 * @param input      Input FP16 tensor
 * @param output     Output packed INT4 tensor (2 values per byte)
 * @param scales     Scale factors (one per INT4_BLOCK_SIZE elements)
 * @param batch_size Batch dimension
 * @param num_tokens Number of tokens (sequence length)
 * @param num_heads  Number of attention heads
 * @param head_dim   Head dimension (must be 64 or 128)
 * @param stride_*   Tensor strides for each dimension
 * @param stream     HIP stream for async execution
 */
void launch_quantize_int4(
    const half* input,
    uint8_t* output,
    half* scales,
    int batch_size,
    int num_tokens,
    int num_heads,
    int head_dim,
    int stride_batch_in,
    int stride_token_in,
    int stride_head_in,
    int stride_batch_out,
    int stride_token_out,
    int stride_head_out,
    int stride_batch_scale,
    int stride_token_scale,
    int stride_head_scale,
    hipStream_t stream
);

/**
 * Dequantize INT4 tensor back to FP16
 *
 * @param input      Input packed INT4 tensor
 * @param scales     Scale factors
 * @param output     Output FP16 tensor
 * @param batch_size Batch dimension
 * @param num_tokens Number of tokens
 * @param num_heads  Number of attention heads
 * @param head_dim   Head dimension
 * @param stride_*   Tensor strides
 * @param stream     HIP stream
 */
void launch_dequantize_int4(
    const uint8_t* input,
    const half* scales,
    half* output,
    int batch_size,
    int num_tokens,
    int num_heads,
    int head_dim,
    int stride_batch_in,
    int stride_token_in,
    int stride_head_in,
    int stride_batch_scale,
    int stride_token_scale,
    int stride_head_scale,
    int stride_batch_out,
    int stride_token_out,
    int stride_head_out,
    hipStream_t stream
);

/**
 * Quantize FP16 tensor to INT4 with transpose
 * Useful for K matrix preparation in attention (K^T)
 *
 * @param input      Input FP16 tensor [batch, tokens, heads, dim]
 * @param output     Output packed INT4 [batch, heads, dim, tokens/2]
 * @param scales     Scale factors
 * @param batch_size Batch dimension
 * @param num_tokens Number of tokens
 * @param num_heads  Number of attention heads
 * @param head_dim   Head dimension
 * @param stride_*   Tensor strides
 * @param stream     HIP stream
 */
void launch_quantize_int4_transpose(
    const half* input,
    uint8_t* output,
    half* scales,
    int batch_size,
    int num_tokens,
    int num_heads,
    int head_dim,
    int stride_batch_in,
    int stride_token_in,
    int stride_head_in,
    int stride_batch_out,
    int stride_head_out,
    int stride_dim_out,
    int stride_batch_scale,
    int stride_head_scale,
    int stride_dim_scale,
    hipStream_t stream
);

#ifdef __cplusplus
}
#endif
