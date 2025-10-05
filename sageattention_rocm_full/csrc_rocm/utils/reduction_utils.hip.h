/*
 * HIP reduction utilities for ROCm port
 * Copyright (c) 2024 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <hip/hip_runtime.h>
#include "hip_utils.h"

// Warp reduction primitives using HIP shuffle intrinsics

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#ifdef __HIP_PLATFORM_AMD__
    // AMD wavefront is 64 threads
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 64);
    }
#else
    // NVIDIA warp is 32 threads
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset, 32);
    }
#endif
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
#ifdef __HIP_PLATFORM_AMD__
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor(val, offset, 64));
    }
#else
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset, 32));
    }
#endif
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_min(T val) {
#ifdef __HIP_PLATFORM_AMD__
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = min(val, __shfl_xor(val, offset, 64));
    }
#else
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = min(val, __shfl_xor_sync(0xffffffff, val, offset, 32));
    }
#endif
    return val;
}

// Block reduction using shared memory
template <typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val) {
    __shared__ T shared[BLOCK_SIZE / warpSize];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Reduce within warp
    val = warp_reduce_sum(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if within valid range
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    // Final reduce within first warp
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

template <typename T, int BLOCK_SIZE>
__device__ T block_reduce_max(T val) {
    __shared__ T shared[BLOCK_SIZE / warpSize];

    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -1e20f;

    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// All-reduce operations (broadcast result to all threads)
template <typename T>
__device__ __forceinline__ T warp_all_reduce_sum(T val) {
    val = warp_reduce_sum(val);
    // Broadcast to all threads in warp
#ifdef __HIP_PLATFORM_AMD__
    val = __shfl(val, 0, 64);  // AMD broadcast from lane 0
#else
    val = __shfl_sync(0xffffffff, val, 0, 32);  // NVIDIA broadcast
#endif
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_all_reduce_max(T val) {
    val = warp_reduce_max(val);
#ifdef __HIP_PLATFORM_AMD__
    val = __shfl(val, 0, 64);
#else
    val = __shfl_sync(0xffffffff, val, 0, 32);
#endif
    return val;
}

// Segmented reduction (for variable-length sequences)
template <typename T>
__device__ T segmented_reduce_sum(T val, int segment_size) {
    int lane = threadIdx.x % warpSize;

#ifdef __HIP_PLATFORM_AMD__
    // AMD: handle up to 64-thread segments
    for (int offset = 1; offset < segment_size && offset < 64; offset <<= 1) {
        T recv = __shfl_up(val, offset, 64);
        if (lane >= offset && lane < segment_size) {
            val += recv;
        }
    }
#else
    // NVIDIA: handle up to 32-thread segments
    for (int offset = 1; offset < segment_size && offset < 32; offset <<= 1) {
        T recv = __shfl_up_sync(0xffffffff, val, offset, 32);
        if (lane >= offset && lane < segment_size) {
            val += recv;
        }
    }
#endif

    return val;
}