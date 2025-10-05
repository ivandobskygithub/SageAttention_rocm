/*
 * INT4 Quantization Implementation for SageAttention3 ROCm Port
 * Optimized for AMD RDNA3.5 (gfx1151 - Radeon 890M iGPU)
 *
 * This implementation provides block-scaled symmetric INT4 quantization
 * as a replacement for NVIDIA's native FP4 operations used in Blackwell.
 *
 * Key features:
 * - Block-scaled quantization (16 elements per block)
 * - Symmetric range: -8 to 7
 * - Efficient packing: 2 INT4 values per byte
 * - Optimized for RDNA memory bandwidth
 * - 32KB LDS (shared memory) constraint-aware
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <algorithm>

// Constants for INT4 quantization
constexpr int INT4_ELEMENTS_PER_THREAD = 16;  // Process 16 elements per thread
constexpr int INT4_BLOCK_SIZE = 16;            // Block size for scale factor
constexpr int INT4_VALUES_PER_BYTE = 2;        // Two INT4 values packed per byte

// RDNA constants
constexpr int WARP_SIZE = 32;  // RDNA wave size
constexpr int INT4_MAX = 7;
constexpr int INT4_MIN = -8;

// Type aliases for clarity
using int4_t = int8_t;  // INT4 stored as signed 8-bit

// ============================================================================
// Helper Functions
// ============================================================================

__device__ __forceinline__ half hmax(half a, half b) {
    return __half2float(a) > __half2float(b) ? a : b;
}

__device__ __forceinline__ half habs(half a) {
    return __float2half(fabsf(__half2float(a)));
}

// ============================================================================
// Utility Functions for INT4 Operations
// ============================================================================

/**
 * Pack two INT4 values into a single byte
 * Layout: [b3 b2 b1 b0 | a3 a2 a1 a0]
 *         MSB nibble   | LSB nibble
 */
__device__ __forceinline__ uint8_t pack_int4_pair(int4_t a, int4_t b) {
    uint8_t a_nibble = static_cast<uint8_t>(a) & 0x0F;
    uint8_t b_nibble = static_cast<uint8_t>(b) & 0x0F;
    return (b_nibble << 4) | a_nibble;
}

/**
 * Unpack two INT4 values from a single byte
 * Performs sign extension for proper INT4 representation
 */
__device__ __forceinline__ void unpack_int4_pair(uint8_t packed, int4_t& a, int4_t& b) {
    // Extract nibbles
    int8_t a_nibble = static_cast<int8_t>(packed & 0x0F);
    int8_t b_nibble = static_cast<int8_t>((packed >> 4) & 0x0F);

    // Sign extension: if MSB of nibble is 1, extend to negative
    a = (a_nibble > 7) ? (a_nibble - 16) : a_nibble;
    b = (b_nibble > 7) ? (b_nibble - 16) : b_nibble;
}

/**
 * Quantize a single FP16 value to INT4 with scaling
 * Uses symmetric quantization: -8 to 7 range
 */
__device__ __forceinline__ int4_t quantize_fp16_to_int4(half value, half scale) {
    float val_f32 = __half2float(value);
    float scale_f32 = __half2float(scale);

    // Avoid division by zero
    if (scale_f32 == 0.0f) {
        return 0;
    }

    // Normalize and round
    float normalized = val_f32 / scale_f32;
    int quantized = __float2int_rn(normalized);

    // Clamp to INT4 range [-8, 7]
    return static_cast<int4_t>(min(max(quantized, INT4_MIN), INT4_MAX));
}

/**
 * Dequantize INT4 value back to FP16
 */
__device__ __forceinline__ half dequantize_int4_to_fp16(int4_t value, half scale) {
    float val_f32 = static_cast<float>(value);
    float scale_f32 = __half2float(scale);
    return __float2half(val_f32 * scale_f32);
}

/**
 * Compute maximum absolute value for a block using warp reduction
 */
__device__ __forceinline__ half compute_block_absmax(const half* values, int count) {
    half local_max = __float2half(0.0f);

    #pragma unroll
    for (int i = 0; i < count; ++i) {
        local_max = hmax(local_max, habs(values[i]));
    }

    // Warp reduction to find maximum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        half other = __shfl_down(local_max, offset, WARP_SIZE);
        local_max = hmax(local_max, other);
    }

    return local_max;
}

// ============================================================================
// Packed Vector Types for Efficient Memory Access
// ============================================================================

/**
 * Packed vector of FP16 values for vectorized loads
 */
struct PackedHalf16 {
    half2 data[8];  // 16 half values as 8 half2 pairs
};

/**
 * Packed INT4 data (8 bytes = 16 INT4 values)
 */
struct PackedInt4_16 {
    uint8_t data[8];  // 16 INT4 values packed into 8 bytes
};

// ============================================================================
// INT4 Quantization Kernel (Standard Layout)
// ============================================================================

/**
 * Quantize FP16 tensor to INT4 with block scaling
 *
 * Input layout:  [batch, num_tokens, num_heads, head_dim]
 * Output layout: [batch, num_tokens, num_heads, head_dim/2] (packed)
 * Scales layout: [batch, num_tokens, num_heads, head_dim/16] (one per block)
 *
 * Each thread processes INT4_ELEMENTS_PER_THREAD elements
 * Block size is INT4_BLOCK_SIZE elements for scale computation
 */
template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void quantize_int4_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    half* __restrict__ scales,
    int batch_size,
    int num_tokens,
    int num_heads,
    int stride_batch_in,
    int stride_token_in,
    int stride_head_in,
    int stride_batch_out,
    int stride_token_out,
    int stride_head_out,
    int stride_batch_scale,
    int stride_token_scale,
    int stride_head_scale
) {
    static_assert(HEAD_DIM % INT4_ELEMENTS_PER_THREAD == 0,
                  "HEAD_DIM must be divisible by INT4_ELEMENTS_PER_THREAD");

    // Thread and block indices
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int token_block_idx = blockIdx.x;

    constexpr int THREADS_PER_TOKEN = HEAD_DIM / INT4_ELEMENTS_PER_THREAD;

    // Calculate token ID for this thread
    const int local_token_id = threadIdx.x / THREADS_PER_TOKEN;
    const int token_id = token_block_idx * BLOCK_SIZE + local_token_id;
    const int feature_thread_id = threadIdx.x % THREADS_PER_TOKEN;

    // Bounds check
    if (token_id >= num_tokens) {
        return;
    }

    // Load input values (16 elements per thread)
    half input_vals[INT4_ELEMENTS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
        int feature_idx = feature_thread_id * INT4_ELEMENTS_PER_THREAD + i;
        int global_idx = batch_idx * stride_batch_in +
                        token_id * stride_token_in +
                        head_idx * stride_head_in +
                        feature_idx;
        input_vals[i] = input[global_idx];
    }

    // Compute local absolute maximum
    half local_absmax = __float2half(0.0f);
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
        local_absmax = hmax(local_absmax, habs(input_vals[i]));
    }

    // Reduce across threads processing same quantization block
    // Each INT4_BLOCK_SIZE elements share one scale factor
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Warp shuffle reduction (within 16-element blocks)
    if constexpr (INT4_ELEMENTS_PER_THREAD == 16) {
        // Each thread has a full block, no inter-thread reduction needed
    } else {
        // Need to reduce across multiple threads
        #pragma unroll
        for (int offset = INT4_BLOCK_SIZE / (INT4_ELEMENTS_PER_THREAD * 2);
             offset > 0; offset /= 2) {
            half other = __shfl_xor_sync(0xFFFFFFFF, local_absmax, offset, WARP_SIZE);
            local_absmax = hmax(local_absmax, other);
        }
    }

    // Compute scale factor: max_abs / 7.0 (INT4_MAX = 7)
    float absmax_f32 = __half2float(local_absmax);
    float scale_f32 = absmax_f32 / 7.0f;
    half scale = __float2half(scale_f32);
    half scale_inv = (scale_f32 == 0.0f) ? __float2half(0.0f) :
                                           __float2half(1.0f / scale_f32);

    // Quantize values to INT4
    int4_t quantized_vals[INT4_ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
        float val_f32 = __half2float(input_vals[i]) * __half2float(scale_inv);
        int val_int = __float2int_rn(val_f32);
        quantized_vals[i] = static_cast<int4_t>(min(max(val_int, INT4_MIN), INT4_MAX));
    }

    // Pack INT4 values into bytes (2 values per byte)
    uint8_t packed_vals[INT4_ELEMENTS_PER_THREAD / 2];
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD / 2; ++i) {
        packed_vals[i] = pack_int4_pair(quantized_vals[i * 2],
                                        quantized_vals[i * 2 + 1]);
    }

    // Write packed output
    int out_base = batch_idx * stride_batch_out +
                   token_id * stride_token_out +
                   head_idx * stride_head_out +
                   feature_thread_id * (INT4_ELEMENTS_PER_THREAD / 2);

    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD / 2; ++i) {
        output[out_base + i] = packed_vals[i];
    }

    // Write scale factor (one per INT4_BLOCK_SIZE elements)
    // Only first thread in block writes the scale
    if (feature_thread_id % (INT4_BLOCK_SIZE / INT4_ELEMENTS_PER_THREAD) == 0) {
        int scale_idx = batch_idx * stride_batch_scale +
                       token_id * stride_token_scale +
                       head_idx * stride_head_scale +
                       (feature_thread_id * INT4_ELEMENTS_PER_THREAD) / INT4_BLOCK_SIZE;
        scales[scale_idx] = scale;
    }
}

// ============================================================================
// INT4 Dequantization Kernel
// ============================================================================

/**
 * Dequantize INT4 tensor back to FP16
 */
template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void dequantize_int4_kernel(
    const uint8_t* __restrict__ input,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int batch_size,
    int num_tokens,
    int num_heads,
    int stride_batch_in,
    int stride_token_in,
    int stride_head_in,
    int stride_batch_scale,
    int stride_token_scale,
    int stride_head_scale,
    int stride_batch_out,
    int stride_token_out,
    int stride_head_out
) {
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int token_block_idx = blockIdx.x;

    constexpr int THREADS_PER_TOKEN = HEAD_DIM / INT4_ELEMENTS_PER_THREAD;

    const int local_token_id = threadIdx.x / THREADS_PER_TOKEN;
    const int token_id = token_block_idx * BLOCK_SIZE + local_token_id;
    const int feature_thread_id = threadIdx.x % THREADS_PER_TOKEN;

    if (token_id >= num_tokens) {
        return;
    }

    // Load scale factor for this block
    int scale_idx = batch_idx * stride_batch_scale +
                   token_id * stride_token_scale +
                   head_idx * stride_head_scale +
                   (feature_thread_id * INT4_ELEMENTS_PER_THREAD) / INT4_BLOCK_SIZE;
    half scale = scales[scale_idx];

    // Load packed INT4 values
    int in_base = batch_idx * stride_batch_in +
                  token_id * stride_token_in +
                  head_idx * stride_head_in +
                  feature_thread_id * (INT4_ELEMENTS_PER_THREAD / 2);

    uint8_t packed_vals[INT4_ELEMENTS_PER_THREAD / 2];
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD / 2; ++i) {
        packed_vals[i] = input[in_base + i];
    }

    // Unpack and dequantize
    half output_vals[INT4_ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD / 2; ++i) {
        int4_t val_a, val_b;
        unpack_int4_pair(packed_vals[i], val_a, val_b);

        output_vals[i * 2] = dequantize_int4_to_fp16(val_a, scale);
        output_vals[i * 2 + 1] = dequantize_int4_to_fp16(val_b, scale);
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
        int feature_idx = feature_thread_id * INT4_ELEMENTS_PER_THREAD + i;
        int out_idx = batch_idx * stride_batch_out +
                     token_id * stride_token_out +
                     head_idx * stride_head_out +
                     feature_idx;
        output[out_idx] = output_vals[i];
    }
}

// ============================================================================
// INT4 Quantization with Transpose (for K matrix)
// ============================================================================

/**
 * Quantize and transpose tensor for efficient K^T operations
 *
 * Input layout:  [batch, num_tokens, num_heads, head_dim]
 * Output layout: [batch, num_heads, head_dim, num_tokens/2] (transposed + packed)
 * Scales layout: [batch, num_heads, head_dim, num_tokens/16]
 */
template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void quantize_int4_transpose_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    half* __restrict__ scales,
    int batch_size,
    int num_tokens,
    int num_heads,
    int stride_batch_in,
    int stride_token_in,
    int stride_head_in,
    int stride_batch_out,
    int stride_head_out,
    int stride_dim_out,
    int stride_batch_scale,
    int stride_head_scale,
    int stride_dim_scale
) {
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int token_block_idx = blockIdx.x;

    constexpr int THREADS_PER_TOKEN = HEAD_DIM / INT4_ELEMENTS_PER_THREAD;

    const int local_token_id = threadIdx.x / THREADS_PER_TOKEN;
    const int token_id = token_block_idx * BLOCK_SIZE + local_token_id;
    const int feature_thread_id = threadIdx.x % THREADS_PER_TOKEN;

    // Use shared memory for transpose
    __shared__ half smem[BLOCK_SIZE * HEAD_DIM];

    // Load input to shared memory
    if (token_id < num_tokens) {
        #pragma unroll
        for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
            int feature_idx = feature_thread_id * INT4_ELEMENTS_PER_THREAD + i;
            int global_idx = batch_idx * stride_batch_in +
                           token_id * stride_token_in +
                           head_idx * stride_head_in +
                           feature_idx;
            int smem_idx = local_token_id * HEAD_DIM + feature_idx;
            smem[smem_idx] = input[global_idx];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
            int feature_idx = feature_thread_id * INT4_ELEMENTS_PER_THREAD + i;
            int smem_idx = local_token_id * HEAD_DIM + feature_idx;
            smem[smem_idx] = __float2half(0.0f);
        }
    }

    __syncthreads();

    // Read transposed data from shared memory
    half input_vals[INT4_ELEMENTS_PER_THREAD];
    const int dim_id = feature_thread_id * INT4_ELEMENTS_PER_THREAD / HEAD_DIM;
    const int local_dim_offset = (feature_thread_id * INT4_ELEMENTS_PER_THREAD) % HEAD_DIM;

    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
        int smem_token_idx = local_token_id + i;  // Transposed access
        int smem_dim_idx = local_dim_offset;
        if (smem_token_idx < BLOCK_SIZE) {
            input_vals[i] = smem[smem_dim_idx * BLOCK_SIZE + smem_token_idx];
        } else {
            input_vals[i] = __float2half(0.0f);
        }
    }

    // Quantization logic (similar to standard kernel)
    half local_absmax = __float2half(0.0f);
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
        local_absmax = hmax(local_absmax, habs(input_vals[i]));
    }

    float absmax_f32 = __half2float(local_absmax);
    float scale_f32 = absmax_f32 / 7.0f;
    half scale = __float2half(scale_f32);
    half scale_inv = (scale_f32 == 0.0f) ? __float2half(0.0f) :
                                           __float2half(1.0f / scale_f32);

    int4_t quantized_vals[INT4_ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD; ++i) {
        float val_f32 = __half2float(input_vals[i]) * __half2float(scale_inv);
        int val_int = __float2int_rn(val_f32);
        quantized_vals[i] = static_cast<int4_t>(min(max(val_int, INT4_MIN), INT4_MAX));
    }

    uint8_t packed_vals[INT4_ELEMENTS_PER_THREAD / 2];
    #pragma unroll
    for (int i = 0; i < INT4_ELEMENTS_PER_THREAD / 2; ++i) {
        packed_vals[i] = pack_int4_pair(quantized_vals[i * 2],
                                        quantized_vals[i * 2 + 1]);
    }

    // Write transposed packed output
    if (token_id < num_tokens) {
        int dim_idx = feature_thread_id * INT4_ELEMENTS_PER_THREAD;
        int out_base = batch_idx * stride_batch_out +
                      head_idx * stride_head_out +
                      dim_idx * stride_dim_out +
                      token_id / 2;  // Packed dimension

        #pragma unroll
        for (int i = 0; i < INT4_ELEMENTS_PER_THREAD / 2; ++i) {
            output[out_base + i * stride_dim_out] = packed_vals[i];
        }

        // Write scale
        if (feature_thread_id % (INT4_BLOCK_SIZE / INT4_ELEMENTS_PER_THREAD) == 0) {
            int scale_idx = batch_idx * stride_batch_scale +
                          head_idx * stride_head_scale +
                          dim_idx * stride_dim_scale +
                          token_id / INT4_BLOCK_SIZE;
            scales[scale_idx] = scale;
        }
    }
}

// ============================================================================
// Host-side Launcher Functions (C API for Python binding)
// ============================================================================

extern "C" {

/**
 * Launch INT4 quantization kernel
 */
__declspec(dllexport) void launch_quantize_int4(
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
) {
    constexpr int BLOCK_SIZE = 128;

    // Dispatch based on head dimension
    auto launch_kernel = [&]<int HEAD_DIM>() {
        constexpr int THREADS_PER_TOKEN = HEAD_DIM / INT4_ELEMENTS_PER_THREAD;
        dim3 grid_dim(
            (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE,
            batch_size,
            num_heads
        );
        dim3 block_dim(BLOCK_SIZE * THREADS_PER_TOKEN);

        hipLaunchKernelGGL(
            (quantize_int4_kernel<HEAD_DIM, BLOCK_SIZE>),
            grid_dim, block_dim, 0, stream,
            input, output, scales,
            batch_size, num_tokens, num_heads,
            stride_batch_in, stride_token_in, stride_head_in,
            stride_batch_out, stride_token_out, stride_head_out,
            stride_batch_scale, stride_token_scale, stride_head_scale
        );
    };

    if (head_dim == 64) {
        launch_kernel.template operator()<64>();
    } else if (head_dim == 128) {
        launch_kernel.template operator()<128>();
    } else {
        // Error: unsupported head dimension
        return;
    }
}

/**
 * Launch INT4 dequantization kernel
 */
__declspec(dllexport) void launch_dequantize_int4(
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
) {
    constexpr int BLOCK_SIZE = 128;

    auto launch_kernel = [&]<int HEAD_DIM>() {
        constexpr int THREADS_PER_TOKEN = HEAD_DIM / INT4_ELEMENTS_PER_THREAD;
        dim3 grid_dim(
            (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE,
            batch_size,
            num_heads
        );
        dim3 block_dim(BLOCK_SIZE * THREADS_PER_TOKEN);

        hipLaunchKernelGGL(
            (dequantize_int4_kernel<HEAD_DIM, BLOCK_SIZE>),
            grid_dim, block_dim, 0, stream,
            input, scales, output,
            batch_size, num_tokens, num_heads,
            stride_batch_in, stride_token_in, stride_head_in,
            stride_batch_scale, stride_token_scale, stride_head_scale,
            stride_batch_out, stride_token_out, stride_head_out
        );
    };

    if (head_dim == 64) {
        launch_kernel.template operator()<64>();
    } else if (head_dim == 128) {
        launch_kernel.template operator()<128>();
    }
}

/**
 * Launch INT4 quantization with transpose kernel
 */
__declspec(dllexport) void launch_quantize_int4_transpose(
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
) {
    constexpr int BLOCK_SIZE = 128;

    auto launch_kernel = [&]<int HEAD_DIM>() {
        constexpr int THREADS_PER_TOKEN = HEAD_DIM / INT4_ELEMENTS_PER_THREAD;
        dim3 grid_dim(
            (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE,
            batch_size,
            num_heads
        );
        dim3 block_dim(BLOCK_SIZE * THREADS_PER_TOKEN);

        // Shared memory size calculation
        size_t smem_size = sizeof(half) * BLOCK_SIZE * HEAD_DIM;

        hipLaunchKernelGGL(
            (quantize_int4_transpose_kernel<HEAD_DIM, BLOCK_SIZE>),
            grid_dim, block_dim, smem_size, stream,
            input, output, scales,
            batch_size, num_tokens, num_heads,
            stride_batch_in, stride_token_in, stride_head_in,
            stride_batch_out, stride_head_out, stride_dim_out,
            stride_batch_scale, stride_head_scale, stride_dim_scale
        );
    };

    if (head_dim == 64) {
        launch_kernel.template operator()<64>();
    } else if (head_dim == 128) {
        launch_kernel.template operator()<128>();
    }
}

} // extern "C"
