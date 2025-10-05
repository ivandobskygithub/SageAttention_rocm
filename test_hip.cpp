
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void test_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello from HIP kernel on gfx1151!\n");
    }
}

int main() {
    std::cout << "Testing HIP compilation for RDNA3.5 (gfx1151)" << std::endl;

    int device_count = 0;
    hipGetDeviceCount(&device_count);
    std::cout << "HIP Device count: " << device_count << std::endl;

    if (device_count > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "GCN arch: " << prop.gcnArch << std::endl;

        // Launch kernel
        test_kernel<<<1, 64>>>();
        hipDeviceSynchronize();
    }

    return 0;
}
