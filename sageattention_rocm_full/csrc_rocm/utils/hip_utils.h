/*
 * HIP utility functions for ROCm port of SageAttention
 * Copyright (c) 2024 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// Check HIP errors
#define HIP_CHECK(call)                                                \
  do {                                                                 \
    hipError_t error = call;                                          \
    if (error != hipSuccess) {                                        \
      AT_ERROR("HIP error: ", hipGetErrorString(error),              \
               " at ", __FILE__, ":", __LINE__);                     \
    }                                                                  \
  } while (0)

// Check tensor properties (reused from CUDA)
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Get thread/block IDs (same as CUDA)
__device__ __forceinline__ uint32_t get_lane_id() {
    return threadIdx.x % warpSize;  // warpSize is 64 on AMD, 32 on NVIDIA
}

__device__ __forceinline__ uint32_t get_warp_id() {
    return threadIdx.x / warpSize;
}

// AMD wavefront is 64 threads (vs NVIDIA warp of 32)
constexpr int AMD_WAVEFRONT_SIZE = 64;
constexpr int NVIDIA_WARP_SIZE = 32;

// Helper to get actual warp/wavefront size
__device__ __forceinline__ uint32_t get_wave_size() {
#ifdef __HIP_PLATFORM_AMD__
    return AMD_WAVEFRONT_SIZE;
#else
    return NVIDIA_WARP_SIZE;
#endif
}

// Memory fence (same syntax as CUDA)
__device__ __forceinline__ void thread_fence() {
    __threadfence();
}

__device__ __forceinline__ void block_fence() {
    __threadfence_block();
}

// Shared memory helpers
template <typename T>
__device__ __forceinline__ T* shared_ptr() {
    extern __shared__ T smem[];
    return smem;
}

// FP16 vector types (compatible with CUDA)
using half2 = __half2;

// Helper for grid-stride loops
#define GRID_STRIDE_LOOP(i, n)                                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);       \
       i += blockDim.x * gridDim.x)

// Launch configuration helpers
inline dim3 get_blocks(int64_t n, int threads = 256) {
    return dim3((n + threads - 1) / threads);
}

inline dim3 get_threads(int threads = 256) {
    return dim3(threads);
}

// Type traits for HIP
template <typename T>
struct CudaTypeTraits;

template <>
struct CudaTypeTraits<float> {
    typedef float Type;
    static constexpr int size = 1;
};

template <>
struct CudaTypeTraits<__half> {
    typedef __half Type;
    static constexpr int size = 1;
};

template <>
struct CudaTypeTraits<__half2> {
    typedef __half2 Type;
    static constexpr int size = 2;
};

// Atomic operations (same as CUDA)
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = __int_as_float(atomicMax((int*)addr, __float_as_int(value)));
    return old;
}

// Fast math operations (HIP equivalents)
__device__ __forceinline__ float fast_exp(float x) {
#ifdef __HIP_PLATFORM_AMD__
    return __expf(x);  // AMD fast math
#else
    return __expf(x);  // CUDA fast math
#endif
}

__device__ __forceinline__ float fast_log2(float x) {
#ifdef __HIP_PLATFORM_AMD__
    return __log2f(x);  // AMD fast math
#else
    return __log2f(x);  // CUDA fast math
#endif
}

__device__ __forceinline__ float fast_tanh(float x) {
#ifdef __HIP_PLATFORM_AMD__
    return tanhf(x);  // No fast tanh on AMD, use standard
#else
    return tanhf(x);  // CUDA version
#endif
}