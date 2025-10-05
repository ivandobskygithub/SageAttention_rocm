#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <algorithm>
#include <cstdio>

// Constants for RDNA3.5 (gfx1151)
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 64;
constexpr int BLOCK_SIZE_K = 32;
constexpr int BLOCK_SCALE_SIZE = 16;

// Debug kernel - just copy Q to O to verify basic functionality
__global__ void test_copy_kernel(
    const half* __restrict__ Q,
    half* __restrict__ O,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        O[idx] = Q[idx];
    }
}

// Simplified attention kernel with debug checks
__global__ void sage_attention_forward_debug(
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
    // Debug: Print from first thread
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("[KERNEL] batch=%d, heads=%d, seq_q=%d, seq_k=%d, head_dim=%d\n",
               batch_size, num_heads, seq_len_q, seq_len_k, head_dim);
        printf("[KERNEL] scale=%f, is_causal=%d\n", scale, is_causal);
        printf("[KERNEL] Q[0]=%f, K[0]=%f, V[0]=%f\n",
               __half2float(Q[0]), __half2float(K[0]), __half2float(V[0]));
    }

    // Thread and block indices
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_row_idx = blockIdx.x * blockDim.x + tid;

    // Bounds check
    if (q_row_idx >= seq_len_q) {
        return;
    }

    // For now, just copy Q to O as a test
    const int q_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;
    const int o_offset = (batch_idx * num_heads + head_idx) * seq_len_q * head_dim;

    for (int d = 0; d < head_dim; ++d) {
        O[o_offset + q_row_idx * head_dim + d] = Q[q_offset + q_row_idx * head_dim + d];
    }
}

extern "C" {

// Debug version of launch function with extensive validation
void launch_sage_attention_forward_debug(
    const void* Q, const void* K, const void* V, void* O,
    const void* delta_s,
    int batch_size, int num_heads, int seq_len_q, int seq_len_k,
    int head_dim, float scale, bool is_causal, hipStream_t stream)
{
    printf("\n=== launch_sage_attention_forward_debug ===\n");
    printf("Parameters:\n");
    printf("  Batch: %d, Heads: %d\n", batch_size, num_heads);
    printf("  SeqQ: %d, SeqK: %d, HeadDim: %d\n", seq_len_q, seq_len_k, head_dim);
    printf("  Scale: %f, Causal: %d\n", scale, is_causal);
    printf("  Stream: %p\n", stream);
    printf("Pointers:\n");
    printf("  Q: %p\n", Q);
    printf("  K: %p\n", K);
    printf("  V: %p\n", V);
    printf("  O: %p\n", O);
    printf("  delta_s: %p\n", delta_s);

    // Validate dimensions (HIP 7 requirements)
    if (batch_size <= 0 || num_heads <= 0 || seq_len_q <= 0 ||
        seq_len_k <= 0 || head_dim <= 0) {
        printf("ERROR: Invalid dimensions! All must be > 0\n");
        return;
    }

    // Check pointer alignment
    if ((uintptr_t)Q % 16 != 0) printf("WARNING: Q not 16-byte aligned\n");
    if ((uintptr_t)K % 16 != 0) printf("WARNING: K not 16-byte aligned\n");
    if ((uintptr_t)V % 16 != 0) printf("WARNING: V not 16-byte aligned\n");
    if ((uintptr_t)O % 16 != 0) printf("WARNING: O not 16-byte aligned\n");

    // Clear any previous errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("WARNING: Pre-existing HIP error: %s\n", hipGetErrorString(err));
        hipGetLastError(); // Clear it
    }

    // Use default stream if NULL (HIP 7 safety)
    if (stream == nullptr) {
        printf("INFO: Using default stream\n");
        stream = 0; // Default stream
    }

    // Test 1: Simple copy kernel first
    printf("\nTest 1: Running simple copy kernel...\n");
    {
        int total_elements = batch_size * num_heads * seq_len_q * head_dim;
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        printf("  Grid: %d, Block: %d, Total elements: %d\n", blocks, threads, total_elements);

        hipLaunchKernelGGL(test_copy_kernel,
                          dim3(blocks), dim3(threads), 0, stream,
                          (const half*)Q, (half*)O, total_elements);

        err = hipGetLastError();
        if (err != hipSuccess) {
            printf("ERROR: Copy kernel launch failed: %s (code %d)\n",
                   hipGetErrorString(err), err);
            return;
        }

        // Synchronize to check for execution errors
        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            printf("ERROR: Copy kernel execution failed: %s (code %d)\n",
                   hipGetErrorString(err), err);
            return;
        }

        printf("  SUCCESS: Copy kernel completed\n");
    }

    // Test 2: Attention kernel with minimal configuration
    printf("\nTest 2: Running attention kernel...\n");
    {
        // Conservative grid/block configuration
        int threads_per_block = 64; // Conservative for RDNA3
        int blocks_per_seq = (seq_len_q + threads_per_block - 1) / threads_per_block;

        dim3 grid(blocks_per_seq, num_heads, batch_size);
        dim3 block(threads_per_block);

        // Calculate shared memory size
        size_t smem_size = threads_per_block * seq_len_k * sizeof(half);

        printf("  Grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
        printf("  Block: %d\n", block.x);
        printf("  Shared memory: %zu bytes\n", smem_size);

        // Check shared memory limit
        int max_smem;
        hipDeviceGetAttribute(&max_smem, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);
        printf("  Max shared memory per block: %d bytes\n", max_smem);

        if (smem_size > max_smem) {
            printf("ERROR: Requested shared memory (%zu) exceeds limit (%d)\n",
                   smem_size, max_smem);
            return;
        }

        // Launch debug kernel
        hipLaunchKernelGGL(sage_attention_forward_debug,
                          grid, block, smem_size, stream,
                          (const half*)Q, (const half*)K, (const half*)V, (half*)O,
                          (const float*)delta_s,
                          batch_size, num_heads, seq_len_q, seq_len_k,
                          head_dim, scale, is_causal);

        err = hipGetLastError();
        if (err != hipSuccess) {
            printf("ERROR: Attention kernel launch failed: %s (code %d)\n",
                   hipGetErrorString(err), err);

            // Provide detailed error analysis
            switch(err) {
                case hipErrorInvalidValue:
                    printf("  Likely cause: Invalid kernel arguments\n");
                    break;
                case hipErrorInvalidConfiguration:
                    printf("  Likely cause: Invalid grid/block configuration\n");
                    printf("  Check: Total threads per block <= %d\n", 1024);
                    break;
                case hipErrorInvalidDevicePointer:
                    printf("  Likely cause: Invalid device pointer\n");
                    break;
                case hipErrorOutOfMemory:
                    printf("  Likely cause: Out of GPU memory\n");
                    break;
                default:
                    printf("  Unknown error code\n");
            }
            return;
        }

        // Synchronize to check for execution errors
        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            printf("ERROR: Attention kernel execution failed: %s (code %d)\n",
                   hipGetErrorString(err), err);
            return;
        }

        printf("  SUCCESS: Attention kernel completed\n");
    }

    printf("=== launch_sage_attention_forward_debug complete ===\n\n");
}

} // extern "C"