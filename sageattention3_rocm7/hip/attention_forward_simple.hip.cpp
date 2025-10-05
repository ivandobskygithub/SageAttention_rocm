#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// AMD device math library forward declaration
extern "C" __device__ float __ocml_exp_f32(float);

// Device-side math functions
__device__ __forceinline__ float device_expf(float x) {
    // Use AMD's OCML (Open Compute Math Library)
    return __ocml_exp_f32(x);
}

// Constants for RDNA3.5 (gfx1151)
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 64;
constexpr int BLOCK_SIZE_K = 32;
constexpr int BLOCK_SCALE_SIZE = 16;

// Simplified attention kernel without rocWMMA
// This uses standard thread-level matrix multiplication
__global__ void sage_attention_forward_simple(
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
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_row_idx = blockIdx.x * blockDim.x + tid;

    // Bounds check
    if (q_row_idx >= seq_len_q) {
        return;
    }

    // Calculate global offsets
    const int q_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    const int k_offset = (batch_idx * num_heads + head_idx) * seq_len_k * head_dim;
    const int v_offset = (batch_idx * num_heads + head_idx) * seq_len_k * head_dim;
    const int o_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;

    // Shared memory for attention scores
    extern __shared__ half smem[];
    half* attention_scores = smem;

    // Load Q vector for this thread
    half q_vec[64];  // Assuming head_dim <= 64
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        q_vec[d] = Q[q_offset + q_row_idx * head_dim + d];
    }

    // Compute attention scores: Q @ K^T
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // First pass: compute scores and find max
    for (int k_idx = 0; k_idx < seq_len_k; ++k_idx) {
        // Apply causal mask
        if (is_causal && k_idx > q_row_idx) {
            attention_scores[tid * seq_len_k + k_idx] = __float2half(-INFINITY);
            continue;
        }

        // Compute dot product Q[q_row_idx, :] @ K[k_idx, :]
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            half k_val = K[k_offset + k_idx * head_dim + d];
            score += __half2float(q_vec[d]) * __half2float(k_val);
        }

        // Apply scale
        score *= scale;

        // Apply per-block mean subtraction if provided
        if (delta_s != nullptr) {
            int block_idx = (q_row_idx / BLOCK_SCALE_SIZE) * ((seq_len_k + BLOCK_SCALE_SIZE - 1) / BLOCK_SCALE_SIZE)
                          + (k_idx / BLOCK_SCALE_SIZE);
            int delta_offset = batch_idx * num_heads + head_idx * ((seq_len_q + BLOCK_SCALE_SIZE - 1) / BLOCK_SCALE_SIZE)
                             * ((seq_len_k + BLOCK_SCALE_SIZE - 1) / BLOCK_SCALE_SIZE) + block_idx;
            score -= delta_s[delta_offset];
        }

        attention_scores[tid * seq_len_k + k_idx] = __float2half(score);
        max_score = (score > max_score) ? score : max_score;
    }

    // Second pass: compute softmax
    for (int k_idx = 0; k_idx < seq_len_k; ++k_idx) {
        if (is_causal && k_idx > q_row_idx) {
            attention_scores[tid * seq_len_k + k_idx] = __float2half(0.0f);
            continue;
        }

        float score = __half2float(attention_scores[tid * seq_len_k + k_idx]);
        float exp_score = device_expf(score - max_score);
        attention_scores[tid * seq_len_k + k_idx] = __float2half(exp_score);
        sum_exp += exp_score;
    }

    // Normalize
    float inv_sum = (sum_exp > 0) ? (1.0f / sum_exp) : 0.0f;
    for (int k_idx = 0; k_idx < seq_len_k; ++k_idx) {
        float normalized = __half2float(attention_scores[tid * seq_len_k + k_idx]) * inv_sum;
        attention_scores[tid * seq_len_k + k_idx] = __float2half(normalized);
    }

    // Compute output: attention_scores @ V
    for (int d = 0; d < head_dim; ++d) {
        float output_val = 0.0f;
        for (int k_idx = 0; k_idx < seq_len_k; ++k_idx) {
            float attn = __half2float(attention_scores[tid * seq_len_k + k_idx]);
            half v_val = V[v_offset + k_idx * head_dim + d];
            output_val += attn * __half2float(v_val);
        }
        O[o_offset + q_row_idx * head_dim + d] = __float2half(output_val);
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
    // Each thread processes one Q row
    int threads_per_block = 256;
    int num_blocks = (seq_len_q + threads_per_block - 1) / threads_per_block;

    dim3 grid_dim(num_blocks, num_heads, batch_size);
    dim3 block_dim(threads_per_block);

    // Shared memory for attention scores
    // Each thread needs seq_len_k scores
    size_t smem_size = sizeof(half) * threads_per_block * seq_len_k;

    // Launch kernel
    hipLaunchKernelGGL(
        sage_attention_forward_simple,
        grid_dim,
        block_dim,
        smem_size,
        stream,
        Q, K, V, O, delta_s,
        batch_size, num_heads, seq_len_q, seq_len_k, head_dim,
        scale, is_causal
    );
}
