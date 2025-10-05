#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <cmath>
#include <algorithm>
#include "include/common.h"

using namespace rocwmma;

// Forward declaration of quantization functions
__global__ void quantize_tensor_int4(const half* input, uint8_t* output, half* scales, int n);
__global__ void dequantize_tensor_int4(const uint8_t* input, const half* scales, half* output, int n);

// WMMA-accelerated attention kernel for RDNA3.5
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void sage_attention_forward_rdna(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const float* __restrict__ delta_s,
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_k,
    const int head_dim,
    const float scale,
    const bool is_causal
) {
    // Thread and block indices
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Block indices for this CTA
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int block_row = blockIdx.x;

    // Calculate global offsets
    const int q_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    const int k_offset = (batch_idx * num_heads + head_idx) * seq_len_k * head_dim;
    const int v_offset = (batch_idx * num_heads + head_idx) * seq_len_k * head_dim;
    const int o_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;

    // Shared memory allocation for tiles
    extern __shared__ half smem[];
    half* smem_q = smem;
    half* smem_k = smem + BLOCK_M * BLOCK_K;
    half* smem_v = smem_k + BLOCK_N * BLOCK_K;
    half* smem_s = smem_v + BLOCK_N * head_dim; // For storing attention scores

    // WMMA fragments for RDNA3.5
    // Using 16x16x16 tiles which are supported on RDNA
    fragment<matrix_a, TILE_M, TILE_N, TILE_K, half, row_major> frag_q;
    fragment<matrix_b, TILE_M, TILE_N, TILE_K, half, col_major> frag_k;
    fragment<accumulator, TILE_M, TILE_N, TILE_K, float> frag_s;
    fragment<accumulator, TILE_M, TILE_N, TILE_K, float> frag_o;

    // Initialize output accumulator
    fill_fragment(frag_o, 0.0f);

    // Row bounds for this block
    const int row_start = block_row * BLOCK_M;
    const int row_end = min(row_start + BLOCK_M, seq_len_q);

    // Process K/V blocks
    for (int block_col = 0; block_col < (seq_len_k + BLOCK_N - 1) / BLOCK_N; ++block_col) {
        const int col_start = block_col * BLOCK_N;
        const int col_end = min(col_start + BLOCK_N, seq_len_k);

        // Apply causal mask if needed
        if (is_causal && col_start > row_end) {
            break; // Skip blocks that are fully masked
        }

        // Cooperatively load Q tile into shared memory
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int global_row = row_start + row;
            int global_col = col;

            if (global_row < seq_len_q && global_col < head_dim) {
                smem_q[row * BLOCK_K + col] = Q[q_offset + global_row * head_dim + global_col];
            } else {
                smem_q[row * BLOCK_K + col] = __float2half(0.0f);
            }
        }

        // Cooperatively load K tile into shared memory (transposed for col_major)
        for (int i = tid; i < BLOCK_N * BLOCK_K; i += blockDim.x) {
            int row = i / BLOCK_K;
            int col = i % BLOCK_K;
            int global_row = col_start + row;
            int global_col = col;

            if (global_row < seq_len_k && global_col < head_dim) {
                smem_k[row * BLOCK_K + col] = K[k_offset + global_row * head_dim + global_col];
            } else {
                smem_k[row * BLOCK_K + col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute Q @ K^T using WMMA
        fill_fragment(frag_s, 0.0f);

        // Loop over K dimension
        for (int k = 0; k < BLOCK_K; k += TILE_K) {
            // Load fragments
            load_matrix_sync(frag_q, &smem_q[(warp_id / 2) * TILE_M * BLOCK_K + k], BLOCK_K);
            load_matrix_sync(frag_k, &smem_k[(warp_id % 2) * TILE_N * BLOCK_K + k], BLOCK_K);

            // Perform matrix multiply-accumulate
            mma_sync(frag_s, frag_q, frag_k, frag_s);
        }

        // Store attention scores to shared memory
        store_matrix_sync(&smem_s[(warp_id / 2) * TILE_M * BLOCK_N + (warp_id % 2) * TILE_N],
                         frag_s, BLOCK_N, mem_row_major);

        __syncthreads();

        // Apply scale and per-block mean subtraction
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x) {
            int row = i / BLOCK_N;
            int col = i % BLOCK_N;
            int global_row = row_start + row;
            int global_col = col_start + col;

            if (global_row < seq_len_q && global_col < seq_len_k) {
                float score = __half2float(smem_s[row * BLOCK_N + col]) * scale;

                // Apply per-block mean subtraction if provided
                if (delta_s != nullptr) {
                    int block_idx = (global_row / BLOCK_SCALE_SIZE) * ((seq_len_k + BLOCK_SCALE_SIZE - 1) / BLOCK_SCALE_SIZE)
                                  + (global_col / BLOCK_SCALE_SIZE);
                    score -= delta_s[batch_idx * num_heads + head_idx * ((seq_len_q + BLOCK_SCALE_SIZE - 1) / BLOCK_SCALE_SIZE)
                                   * ((seq_len_k + BLOCK_SCALE_SIZE - 1) / BLOCK_SCALE_SIZE) + block_idx];
                }

                // Apply causal mask
                if (is_causal && global_col > global_row) {
                    score = -INFINITY;
                }

                smem_s[row * BLOCK_N + col] = __float2half(score);
            }
        }

        __syncthreads();

        // Compute softmax over the scores (simplified version)
        // In production, use a more numerically stable implementation
        float row_max[BLOCK_M];
        float row_sum[BLOCK_M];

        // Find max per row
        for (int row = tid; row < BLOCK_M; row += blockDim.x) {
            float max_val = -INFINITY;
            for (int col = 0; col < BLOCK_N; ++col) {
                if (col_start + col < seq_len_k) {
                    max_val = fmaxf(max_val, __half2float(smem_s[row * BLOCK_N + col]));
                }
            }
            row_max[row] = max_val;
        }

        __syncthreads();

        // Compute exp and sum
        for (int row = tid; row < BLOCK_M; row += blockDim.x) {
            float sum = 0.0f;
            for (int col = 0; col < BLOCK_N; ++col) {
                if (col_start + col < seq_len_k) {
                    float val = expf(__half2float(smem_s[row * BLOCK_N + col]) - row_max[row]);
                    smem_s[row * BLOCK_N + col] = __float2half(val);
                    sum += val;
                }
            }
            row_sum[row] = sum;
        }

        __syncthreads();

        // Normalize
        for (int row = tid; row < BLOCK_M; row += blockDim.x) {
            if (row_sum[row] > 0) {
                for (int col = 0; col < BLOCK_N; ++col) {
                    smem_s[row * BLOCK_N + col] = __float2half(__half2float(smem_s[row * BLOCK_N + col]) / row_sum[row]);
                }
            }
        }

        __syncthreads();

        // Load V tile and compute attention @ V
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int global_row = col_start + row;

            if (global_row < seq_len_k && col < head_dim) {
                smem_v[row * head_dim + col] = V[v_offset + global_row * head_dim + col];
            } else {
                smem_v[row * head_dim + col] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute attention @ V using regular matrix multiply (simplified)
        // In production, use WMMA for this as well
        for (int row = tid; row < BLOCK_M; row += blockDim.x) {
            if (row_start + row < seq_len_q) {
                for (int col = 0; col < head_dim; ++col) {
                    float sum = 0.0f;
                    for (int k = 0; k < BLOCK_N; ++k) {
                        if (col_start + k < seq_len_k) {
                            sum += __half2float(smem_s[row * BLOCK_N + k]) * __half2float(smem_v[k * head_dim + col]);
                        }
                    }
                    // Accumulate to output
                    atomicAdd(&O[o_offset + (row_start + row) * head_dim + col], __float2half(sum));
                }
            }
        }
    }
}

// Kernel launcher function
extern "C" __declspec(dllexport) void launch_sage_attention_forward(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    const float* delta_s,
    int batch_size,
    int num_heads,
    int seq_len_q,
    int seq_len_k,
    int head_dim,
    float scale,
    bool is_causal,
    hipStream_t stream
) {
    // Calculate grid and block dimensions
    dim3 grid_dim(
        (seq_len_q + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
        num_heads,
        batch_size
    );
    dim3 block_dim(256); // 8 warps per block

    // Calculate shared memory size
    size_t smem_size = sizeof(half) * (
        BLOCK_SIZE_M * BLOCK_SIZE_K +  // Q tile
        BLOCK_SIZE_N * BLOCK_SIZE_K +  // K tile
        BLOCK_SIZE_N * head_dim +      // V tile
        BLOCK_SIZE_M * BLOCK_SIZE_N    // Attention scores
    );

    // Launch kernel
    hipLaunchKernelGGL(
        sage_attention_forward_rdna<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K>,
        grid_dim,
        block_dim,
        smem_size,
        stream,
        Q, K, V, O, delta_s,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        scale, is_causal
    );
}