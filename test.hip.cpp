#include <hip/hip_runtime.h>

__global__ void test_kernel() {
    // Simple test kernel
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
}

int main() {
    // Simple test to verify compilation
    dim3 blocks(1);
    dim3 threads(32);
    hipLaunchKernelGGL(test_kernel, blocks, threads, 0, 0);
    hipDeviceSynchronize();

    printf("Test kernel compiled and executed successfully!\n");
    return 0;
}