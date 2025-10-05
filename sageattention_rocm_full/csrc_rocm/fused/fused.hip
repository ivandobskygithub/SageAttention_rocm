/*
 * HIP implementation of fused operations for SageAttention ROCm
 * Copyright (c) 2024 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include "../utils/hip_utils.h"
#include "../utils/reduction_utils.hip.h"

// Kernel for per-block INT8 quantization
template <typename T, int BLOCK_SIZE = 128>
__global__ void quant_per_block_int8_kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    const int n,
    const int block_size)
{
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_start = block_id * block_size;
    const int block_end = min(block_start + block_size, n);

    // Shared memory for reduction
    __shared__ float shared_max;

    // Find max absolute value in block
    float local_max = 0.0f;
    for (int i = block_start + tid; i < block_end; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf((float)input[i]));
    }

    // Reduce to find block max
    local_max = block_reduce_max<float, BLOCK_SIZE>(local_max);

    if (tid == 0) {
        shared_max = local_max;
        scale[block_id] = local_max / 127.0f;
    }
    __syncthreads();

    // Quantize using the scale
    float inv_scale = 127.0f / (shared_max + 1e-6f);
    for (int i = block_start + tid; i < block_end; i += blockDim.x) {
        float val = (float)input[i] * inv_scale;
        // Round to nearest integer
        val = roundf(val);
        // Clamp to INT8 range
        val = fmaxf(-128.0f, fminf(127.0f, val));
        output[i] = (int8_t)val;
    }
}

// Kernel for mean subtraction
template <typename T>
__global__ void sub_mean_kernel(
    T* __restrict__ data,
    const int batch_size,
    const int seq_len,
    const int head_dim)
{
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    const int offset = batch_idx * seq_len * head_dim + seq_idx * head_dim;
    data += offset;

    // Compute mean across head_dim
    float sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        sum += (float)data[i];
    }

    // Reduce sum across threads
    sum = block_reduce_sum<float, 256>(sum);
    __shared__ float mean;

    if (tid == 0) {
        mean = sum / head_dim;
    }
    __syncthreads();

    // Subtract mean
    for (int i = tid; i < head_dim; i += blockDim.x) {
        data[i] = (T)((float)data[i] - mean);
    }
}

// Kernel for transpose and permute operations
template <typename T>
__global__ void transpose_pad_permute_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int padded_seq_len)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    // Decompose index (B, H, S, D) layout
    int remaining = idx;
    const int d = remaining % head_dim;
    remaining /= head_dim;
    const int s = remaining % seq_len;
    remaining /= seq_len;
    const int h = remaining % num_heads;
    const int b = remaining / num_heads;

    // Compute output index with padding (B, S_padded, H, D) layout
    const int out_idx = b * padded_seq_len * num_heads * head_dim +
                       s * num_heads * head_dim +
                       h * head_dim +
                       d;

    output[out_idx] = input[idx];
}

// Python bindings
torch::Tensor quant_per_block_int8_hip(
    torch::Tensor input,
    int block_size)
{
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::kFloat16 || input.dtype() == torch::kFloat32,
                "Input must be FP16 or FP32");

    const auto n = input.numel();
    const auto num_blocks = (n + block_size - 1) / block_size;

    // Allocate output tensors
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(input.device());
    auto output = torch::empty_like(input, options);

    auto scale_options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto scale = torch::empty({num_blocks}, scale_options);

    // Launch kernel
    const int threads = 256;
    const int blocks = num_blocks;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "quant_per_block_int8", ([&] {
        quant_per_block_int8_kernel<scalar_t, 256><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            n,
            block_size
        );
    }));

    HIP_CHECK(hipGetLastError());

    return output;
}

torch::Tensor sub_mean_hip(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 3 || input.dim() == 4,
                "Input must be 3D (B,S,D) or 4D (B,H,S,D)");

    int batch_size, seq_len, head_dim;

    if (input.dim() == 3) {
        batch_size = input.size(0);
        seq_len = input.size(1);
        head_dim = input.size(2);
    } else {
        batch_size = input.size(0) * input.size(1);
        seq_len = input.size(2);
        head_dim = input.size(3);
        input = input.view({batch_size, seq_len, head_dim});
    }

    // Launch kernel
    dim3 blocks(batch_size, seq_len);
    dim3 threads(256);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sub_mean", ([&] {
        sub_mean_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            head_dim
        );
    }));

    HIP_CHECK(hipGetLastError());

    return input;
}

torch::Tensor transpose_pad_permute_hip(
    torch::Tensor input,
    int padded_seq_len)
{
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (B,H,S,D)");

    const int batch_size = input.size(0);
    const int num_heads = input.size(1);
    const int seq_len = input.size(2);
    const int head_dim = input.size(3);

    TORCH_CHECK(padded_seq_len >= seq_len, "Padded length must be >= sequence length");

    // Create output tensor with new layout
    auto output = torch::zeros({batch_size, padded_seq_len, num_heads, head_dim},
                              input.options());

    // Launch kernel
    const int total = batch_size * num_heads * seq_len * head_dim;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(input.scalar_type(), "transpose_pad_permute", ([&] {
        transpose_pad_permute_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            padded_seq_len
        );
    }));

    HIP_CHECK(hipGetLastError());

    return output;
}

// Export functions for Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_per_block_int8", &quant_per_block_int8_hip,
          "Per-block INT8 quantization (HIP)");
    m.def("sub_mean", &sub_mean_hip,
          "Subtract mean from tensor (HIP)");
    m.def("transpose_pad_permute", &transpose_pad_permute_hip,
          "Transpose, pad and permute tensor (HIP)");
}