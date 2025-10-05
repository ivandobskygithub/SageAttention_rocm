#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

// Constants for RDNA3.5 (gfx1151)
constexpr int WARP_SIZE = 32;  // RDNA wave size
constexpr int LDS_SIZE = 32768; // 32KB shared memory per CU
constexpr int BLOCK_SIZE_M = 128;
constexpr int BLOCK_SIZE_N = 128;
constexpr int BLOCK_SIZE_K = 32;

// Tile sizes for attention computation
constexpr int TILE_M = 16;  // WMMA tile dimensions for RDNA
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;

// Quantization constants
constexpr int BLOCK_SCALE_SIZE = 16; // Elements per quantization block
constexpr int INT4_MAX = 7;
constexpr int INT4_MIN = -8;

// Helper functions
__device__ __forceinline__ half hmax(half a, half b) {
    return __half2float(a) > __half2float(b) ? a : b;
}

__device__ __forceinline__ half habs(half a) {
    return __float2half(fabsf(__half2float(a)));
}

__device__ __forceinline__ int4_t quantize_to_int4(half val, half scale) {
    float normalized = __half2float(val) / __half2float(scale);
    int quantized = __float2int_rn(normalized);
    return min(max(quantized, INT4_MIN), INT4_MAX);
}

__device__ __forceinline__ half dequantize_from_int4(int4_t val, half scale) {
    return __float2half(float(val) * __half2float(scale));
}

// Pack two INT4 values into one byte
__device__ __forceinline__ uint8_t pack_int4(int4_t a, int4_t b) {
    return ((b & 0xF) << 4) | (a & 0xF);
}

// Unpack two INT4 values from one byte
__device__ __forceinline__ void unpack_int4(uint8_t packed, int4_t& a, int4_t& b) {
    a = (packed & 0xF);
    if (a > 7) a -= 16; // Sign extension
    b = ((packed >> 4) & 0xF);
    if (b > 7) b -= 16; // Sign extension
}

// Structure for attention parameters
struct AttentionParams {
    // Pointers to tensors
    const half* Q;
    const half* K;
    const half* V;
    half* O;

    // Quantized versions
    const uint8_t* Q_int4;
    const uint8_t* K_int4;
    const uint8_t* V_int4;
    const half* Q_scales;
    const half* K_scales;
    const half* V_scales;

    // Per-block mean subtraction values
    const float* delta_s;

    // Dimensions
    int batch_size;
    int num_heads;
    int seq_len_q;
    int seq_len_k;
    int head_dim;

    // Scaling factor for softmax
    float scale;

    // Causal mask flag
    bool is_causal;

    // Use quantization
    bool use_int4_quantization;
};