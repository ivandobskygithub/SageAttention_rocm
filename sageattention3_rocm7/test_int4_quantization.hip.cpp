/*
 * Test program for INT4 quantization operations
 * Verifies compilation and basic functionality
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "hip/quantization/int4_ops.h"

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Simple test configuration
constexpr int BATCH_SIZE = 2;
constexpr int NUM_TOKENS = 128;
constexpr int NUM_HEADS = 8;
constexpr int HEAD_DIM = 64;

void test_quantization_dequantization() {
    std::cout << "Testing INT4 Quantization/Dequantization..." << std::endl;
    std::cout << "Configuration: " << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Num tokens: " << NUM_TOKENS << std::endl;
    std::cout << "  Num heads: " << NUM_HEADS << std::endl;
    std::cout << "  Head dim: " << HEAD_DIM << std::endl;

    // Calculate sizes
    const int input_size = BATCH_SIZE * NUM_TOKENS * NUM_HEADS * HEAD_DIM;
    const int output_size = BATCH_SIZE * NUM_TOKENS * NUM_HEADS * (HEAD_DIM / 2);
    const int scale_size = BATCH_SIZE * NUM_TOKENS * NUM_HEADS * (HEAD_DIM / 16);

    // Allocate host memory
    std::vector<half> h_input(input_size);
    std::vector<uint8_t> h_output(output_size);
    std::vector<half> h_scales(scale_size);
    std::vector<half> h_reconstructed(input_size);

    // Initialize input with random FP16 values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < input_size; ++i) {
        h_input[i] = __float2half(dis(gen));
    }

    std::cout << "Generated " << input_size << " random FP16 values" << std::endl;

    // Allocate device memory
    half* d_input;
    uint8_t* d_output;
    half* d_scales;
    half* d_reconstructed;

    HIP_CHECK(hipMalloc(&d_input, input_size * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_output, output_size * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc(&d_scales, scale_size * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_reconstructed, input_size * sizeof(half)));

    std::cout << "Allocated device memory" << std::endl;

    // Copy input to device
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_size * sizeof(half),
                        hipMemcpyHostToDevice));

    // Calculate strides (assuming contiguous layout)
    const int stride_batch_in = NUM_TOKENS * NUM_HEADS * HEAD_DIM;
    const int stride_token_in = NUM_HEADS * HEAD_DIM;
    const int stride_head_in = HEAD_DIM;

    const int stride_batch_out = NUM_TOKENS * NUM_HEADS * (HEAD_DIM / 2);
    const int stride_token_out = NUM_HEADS * (HEAD_DIM / 2);
    const int stride_head_out = HEAD_DIM / 2;

    const int stride_batch_scale = NUM_TOKENS * NUM_HEADS * (HEAD_DIM / 16);
    const int stride_token_scale = NUM_HEADS * (HEAD_DIM / 16);
    const int stride_head_scale = HEAD_DIM / 16;

    // Create HIP stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    std::cout << "Launching quantization kernel..." << std::endl;

    // Launch quantization kernel
    launch_quantize_int4(
        d_input, d_output, d_scales,
        BATCH_SIZE, NUM_TOKENS, NUM_HEADS, HEAD_DIM,
        stride_batch_in, stride_token_in, stride_head_in,
        stride_batch_out, stride_token_out, stride_head_out,
        stride_batch_scale, stride_token_scale, stride_head_scale,
        stream
    );

    HIP_CHECK(hipStreamSynchronize(stream));
    std::cout << "Quantization completed" << std::endl;

    // Launch dequantization kernel
    std::cout << "Launching dequantization kernel..." << std::endl;

    launch_dequantize_int4(
        d_output, d_scales, d_reconstructed,
        BATCH_SIZE, NUM_TOKENS, NUM_HEADS, HEAD_DIM,
        stride_batch_out, stride_token_out, stride_head_out,
        stride_batch_scale, stride_token_scale, stride_head_scale,
        stride_batch_in, stride_token_in, stride_head_in,
        stream
    );

    HIP_CHECK(hipStreamSynchronize(stream));
    std::cout << "Dequantization completed" << std::endl;

    // Copy results back to host
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, output_size * sizeof(uint8_t),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_scales.data(), d_scales, scale_size * sizeof(half),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_reconstructed.data(), d_reconstructed,
                        input_size * sizeof(half), hipMemcpyDeviceToHost));

    // Verify results: compute reconstruction error
    double total_error = 0.0;
    double max_error = 0.0;
    int num_valid = 0;

    for (int i = 0; i < input_size; ++i) {
        float original = __half2float(h_input[i]);
        float reconstructed = __half2float(h_reconstructed[i]);
        float error = std::abs(original - reconstructed);

        total_error += error;
        max_error = std::max(max_error, static_cast<double>(error));
        num_valid++;
    }

    double mean_error = total_error / num_valid;

    std::cout << "\n=== Quantization Quality Metrics ===" << std::endl;
    std::cout << "Mean absolute error: " << mean_error << std::endl;
    std::cout << "Max absolute error: " << max_error << std::endl;

    // Check if errors are within acceptable range
    // INT4 quantization has limited precision, so expect some error
    const double acceptable_mean_error = 0.15;  // ~1.5 INT4 steps
    const double acceptable_max_error = 0.5;

    if (mean_error < acceptable_mean_error && max_error < acceptable_max_error) {
        std::cout << "\n✓ TEST PASSED: Quantization errors within acceptable range"
                  << std::endl;
    } else {
        std::cout << "\n✗ TEST FAILED: Quantization errors too large" << std::endl;
        std::cout << "  Expected mean error < " << acceptable_mean_error
                  << ", got " << mean_error << std::endl;
        std::cout << "  Expected max error < " << acceptable_max_error
                  << ", got " << max_error << std::endl;
    }

    // Print some sample values for debugging
    std::cout << "\nSample values (first 8 elements):" << std::endl;
    std::cout << "Original -> Reconstructed (Error)" << std::endl;
    for (int i = 0; i < std::min(8, input_size); ++i) {
        float orig = __half2float(h_input[i]);
        float recon = __half2float(h_reconstructed[i]);
        float err = std::abs(orig - recon);
        std::cout << "  " << orig << " -> " << recon << " (" << err << ")" << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_scales));
    HIP_CHECK(hipFree(d_reconstructed));
    HIP_CHECK(hipStreamDestroy(stream));

    std::cout << "\nTest cleanup completed" << std::endl;
}

int main() {
    std::cout << "INT4 Quantization Test Suite for RDNA3.5" << std::endl;
    std::cout << "=========================================\n" << std::endl;

    // Check HIP device
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return 1;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::cout << "Using HIP device: " << prop.name << std::endl;
    std::cout << "Architecture: " << prop.gcnArchName << std::endl;
    std::cout << "Compute units: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock / 1024
              << " KB" << std::endl;
    std::cout << std::endl;

    try {
        test_quantization_dequantization();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
