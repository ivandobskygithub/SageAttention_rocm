/*
 * ROCm/HIP port of SageAttention kernel for gfx90a (MI200)
 *
 * Copyright (c) 2024 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <hipblas/hipblas.h>

// ROCm MFMA intrinsics for matrix operations
#include <rocwmma/rocwmma.hpp>

#include "../hip_utils.h"
#include "../hip_math.h"
#include "attn_hip_utils.h"

// Define architecture-specific parameters
#define WARP_SIZE 64  // AMD wavefront size
#define MFMA_M 16     // MFMA tile dimensions
#define MFMA_N 16
#define MFMA_K 16

// For INT8 operations on CDNA2
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

using namespace rocwmma;

template<
    uint32_t CTA_Q,      // Thread block tile size for Q
    uint32_t CTA_K,      // Thread block tile size for K
    uint32_t WARP_Q,     // Warp tile size for Q
    uint32_t WARP_K,     // Warp tile size for K
    uint32_t HEAD_DIM,   // Attention head dimension
    typename DTypeQK,    // Data type for Q/K (int8)
    typename DTypeSV,    // Data type for S/V (fp16)
    typename DTypeOut    // Output data type
>
__global__ void qk_int8_sv_f16_attn_kernel_hip(
    int8_t* __restrict__ Q,
    int8_t* __restrict__ K,
    __half* __restrict__ V,
    DTypeOut* __restrict__ O,
    float* __restrict__ Q_scale,
    float* __restrict__ K_scale,
    const uint32_t qo_len,
    const uint32_t kv_len,
    const uint32_t num_heads,
    const uint32_t batch_size,
    const float sm_scale,
    const bool is_causal)
{
    // Thread block and warp IDs
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t lane_id = tid % WARP_SIZE;

    // Block indices
    const uint32_t batch_idx = blockIdx.z;
    const uint32_t head_idx = blockIdx.y;
    const uint32_t q_block_idx = blockIdx.x;

    // Shared memory for Q, K, V tiles
    extern __shared__ char smem[];
    int8_t* smem_q = (int8_t*)smem;
    int8_t* smem_k = (int8_t*)(smem_q + CTA_Q * HEAD_DIM);
    __half* smem_v = (__half*)(smem_k + CTA_K * HEAD_DIM);

    // Calculate global offsets
    const uint32_t q_offset = batch_idx * num_heads * qo_len * HEAD_DIM +
                             head_idx * qo_len * HEAD_DIM +
                             q_block_idx * CTA_Q * HEAD_DIM;

    const uint32_t k_offset = batch_idx * num_heads * kv_len * HEAD_DIM +
                             head_idx * kv_len * HEAD_DIM;

    // Initialize accumulator for output
    float acc[WARP_Q][HEAD_DIM / WARP_SIZE];
    #pragma unroll
    for (int i = 0; i < WARP_Q; i++) {
        #pragma unroll
        for (int j = 0; j < HEAD_DIM / WARP_SIZE; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Online softmax state
    float m_prev[WARP_Q];  // Previous max
    float l_prev[WARP_Q];  // Previous sum
    #pragma unroll
    for (int i = 0; i < WARP_Q; i++) {
        m_prev[i] = -INFINITY;
        l_prev[i] = 0.0f;
    }

    // Load Q tile to shared memory (cooperative load)
    cooperative_load_tile<int8_t, CTA_Q, HEAD_DIM>(
        Q + q_offset, smem_q, tid, CTA_Q, HEAD_DIM
    );
    __syncthreads();

    // Main loop over K/V blocks
    for (uint32_t kv_block = 0; kv_block < (kv_len + CTA_K - 1) / CTA_K; kv_block++) {

        // Check causal masking
        if (is_causal) {
            uint32_t q_pos = q_block_idx * CTA_Q;
            uint32_t k_pos = kv_block * CTA_K;
            if (q_pos < k_pos) {
                continue;  // Skip this K/V block due to causal mask
            }
        }

        // Load K tile to shared memory
        uint32_t k_tile_offset = k_offset + kv_block * CTA_K * HEAD_DIM;
        cooperative_load_tile<int8_t, CTA_K, HEAD_DIM>(
            K + k_tile_offset, smem_k, tid, CTA_K, HEAD_DIM
        );
        __syncthreads();

        // Compute QK^T using MFMA instructions
        // For INT8, we use MFMA with appropriate conversions
        float qk_acc[WARP_Q][WARP_K];

        #pragma unroll
        for (int wq = 0; wq < WARP_Q / MFMA_M; wq++) {
            #pragma unroll
            for (int wk = 0; wk < WARP_K / MFMA_N; wk++) {
                // Load fragments from shared memory
                fragment<matrix_a, MFMA_M, MFMA_N, MFMA_K, __half, row_major> q_frag;
                fragment<matrix_b, MFMA_M, MFMA_N, MFMA_K, __half, col_major> k_frag;
                fragment<accumulator, MFMA_M, MFMA_N, MFMA_K, float> acc_frag;

                // Convert INT8 to FP16 and load
                // Note: This is simplified - actual implementation would need proper conversion
                load_matrix_sync(q_frag, (__half*)(smem_q + wq * MFMA_M * HEAD_DIM), HEAD_DIM);
                load_matrix_sync(k_frag, (__half*)(smem_k + wk * MFMA_N * HEAD_DIM), HEAD_DIM);

                // Perform matrix multiplication
                fill_fragment(acc_frag, 0.0f);
                mma_sync(acc_frag, q_frag, k_frag, acc_frag);

                // Store to local accumulator
                store_matrix_sync(&qk_acc[wq * MFMA_M][wk * MFMA_N], acc_frag, WARP_K, mem_row_major);
            }
        }

        // Apply scaling and softmax
        #pragma unroll
        for (int i = 0; i < WARP_Q; i++) {
            float m_new = m_prev[i];
            float l_new = 0.0f;

            #pragma unroll
            for (int j = 0; j < WARP_K; j++) {
                // Apply Q/K scales and softmax scale
                float q_scale_val = Q_scale[head_idx];
                float k_scale_val = K_scale[head_idx];
                qk_acc[i][j] *= q_scale_val * k_scale_val * sm_scale;

                // Online softmax - find new max
                m_new = fmaxf(m_new, qk_acc[i][j]);
            }

            // Compute exponentials and sum
            #pragma unroll
            for (int j = 0; j < WARP_K; j++) {
                qk_acc[i][j] = __expf(qk_acc[i][j] - m_new);
                l_new += qk_acc[i][j];
            }

            // Update running statistics
            float m_diff = m_prev[i] - m_new;
            float l_correction = __expf(m_diff);
            l_new = l_prev[i] * l_correction + l_new;

            // Rescale previous accumulator
            #pragma unroll
            for (int d = 0; d < HEAD_DIM / WARP_SIZE; d++) {
                acc[i][d] *= l_correction;
            }

            m_prev[i] = m_new;
            l_prev[i] = l_new;
        }

        // Load V tile
        uint32_t v_tile_offset = k_offset + kv_block * CTA_K * HEAD_DIM;
        cooperative_load_tile<__half, CTA_K, HEAD_DIM>(
            V + v_tile_offset, smem_v, tid, CTA_K, HEAD_DIM
        );
        __syncthreads();

        // Compute S * V using MFMA
        #pragma unroll
        for (int wq = 0; wq < WARP_Q / MFMA_M; wq++) {
            #pragma unroll
            for (int wv = 0; wv < HEAD_DIM / MFMA_N / WARP_SIZE; wv++) {
                fragment<matrix_a, MFMA_M, MFMA_N, MFMA_K, float, row_major> s_frag;
                fragment<matrix_b, MFMA_M, MFMA_N, MFMA_K, __half, col_major> v_frag;
                fragment<accumulator, MFMA_M, MFMA_N, MFMA_K, float> out_frag;

                // Load attention scores
                load_matrix_sync(s_frag, &qk_acc[wq * MFMA_M][0], WARP_K);

                // Load values
                load_matrix_sync(v_frag, smem_v + wv * MFMA_N, HEAD_DIM);

                // Load current accumulator
                load_matrix_sync(out_frag, &acc[wq * MFMA_M][wv * MFMA_N], HEAD_DIM / WARP_SIZE);

                // Accumulate S * V
                mma_sync(out_frag, s_frag, v_frag, out_frag);

                // Store back to accumulator
                store_matrix_sync(&acc[wq * MFMA_M][wv * MFMA_N], out_frag, HEAD_DIM / WARP_SIZE, mem_row_major);
            }
        }

        __syncthreads();
    }

    // Final normalization and output
    #pragma unroll
    for (int i = 0; i < WARP_Q; i++) {
        float inv_l = 1.0f / l_prev[i];
        #pragma unroll
        for (int j = 0; j < HEAD_DIM / WARP_SIZE; j++) {
            acc[i][j] *= inv_l;
        }
    }

    // Write output to global memory
    const uint32_t o_offset = batch_idx * num_heads * qo_len * HEAD_DIM +
                             head_idx * qo_len * HEAD_DIM +
                             q_block_idx * CTA_Q * HEAD_DIM;

    cooperative_store_tile<DTypeOut, WARP_Q, HEAD_DIM>(
        acc, O + o_offset, tid, WARP_Q, HEAD_DIM
    );
}

// Helper function for cooperative tile loading
template<typename T, int ROWS, int COLS>
__device__ void cooperative_load_tile(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int tid,
    int rows,
    int cols)
{
    // Simple cooperative load - each thread loads multiple elements
    const int total_elements = rows * cols;
    const int threads = blockDim.x;
    const int elems_per_thread = (total_elements + threads - 1) / threads;

    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid + i * threads;
        if (idx < total_elements) {
            dst[idx] = src[idx];
        }
    }
}

// Helper function for cooperative tile storing
template<typename T, int ROWS, int COLS>
__device__ void cooperative_store_tile(
    const float acc[][COLS],
    T* __restrict__ dst,
    int tid,
    int rows,
    int cols)
{
    // Convert float accumulator to output type and store
    const int total_elements = rows * cols;
    const int threads = blockDim.x;
    const int elems_per_thread = (total_elements + threads - 1) / threads;

    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid + i * threads;
        if (idx < total_elements) {
            int row = idx / cols;
            int col = idx % cols;
            dst[idx] = (T)acc[row][col];
        }
    }
}

// Host-side kernel launcher
extern "C" void launch_qk_int8_sv_f16_attn_hip(
    void* q, void* k, void* v, void* o,
    void* q_scale, void* k_scale,
    int batch_size, int num_heads,
    int qo_len, int kv_len, int head_dim,
    float sm_scale, bool is_causal,
    hipStream_t stream)
{
    // Configure kernel launch parameters
    const int CTA_Q = 64;
    const int CTA_K = 64;
    const int WARP_Q = 16;
    const int WARP_K = 16;

    dim3 grid(
        (qo_len + CTA_Q - 1) / CTA_Q,
        num_heads,
        batch_size
    );

    dim3 block(256);  // 4 warps on AMD (256 / 64)

    // Calculate shared memory size
    int smem_size = (CTA_Q + CTA_K) * head_dim * sizeof(int8_t) +
                    CTA_K * head_dim * sizeof(__half);

    // Launch kernel
    hipLaunchKernelGGL(
        (qk_int8_sv_f16_attn_kernel_hip<CTA_Q, CTA_K, WARP_Q, WARP_K, 128, int8_t, __half, __half>),
        grid, block, smem_size, stream,
        (int8_t*)q, (int8_t*)k, (__half*)v, (__half*)o,
        (float*)q_scale, (float*)k_scale,
        qo_len, kv_len, num_heads, batch_size,
        sm_scale, is_causal
    );
}