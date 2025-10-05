/**
 * PyTorch C++ Extension Binding for SageAttention3 ROCm Kernels
 *
 * This file creates Python bindings for the HIP kernels using PyTorch's
 * C++ extension API (pybind11).
 */

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>

// Forward declarations of kernel launcher functions
extern "C" {
    void launch_sage_attention_forward(
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
    );

    void launch_quantize_int4(
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
    );

    void launch_dequantize_int4(
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
    );
}

// Macro for checking HIP errors
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
        } \
    } while(0)

/**
 * Python binding for attention_forward
 */
torch::Tensor attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    float scale,
    bool is_causal,
    torch::optional<torch::Tensor> delta_s,
    bool use_int4_quantization
) {
    // Validate inputs
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(O.is_cuda(), "O must be a CUDA tensor");

    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be float16");
    TORCH_CHECK(O.dtype() == torch::kFloat16, "O must be float16");

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(O.is_contiguous(), "O must be contiguous");

    // Get dimensions
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len_q = Q.size(2);
    const int head_dim = Q.size(3);
    const int seq_len_k = K.size(2);

    // Validate shapes
    TORCH_CHECK(K.size(0) == batch_size, "K batch size mismatch");
    TORCH_CHECK(K.size(1) == num_heads, "K num_heads mismatch");
    TORCH_CHECK(K.size(3) == head_dim, "K head_dim mismatch");
    TORCH_CHECK(V.size(0) == batch_size, "V batch size mismatch");
    TORCH_CHECK(V.size(1) == num_heads, "V num_heads mismatch");
    TORCH_CHECK(V.size(2) == seq_len_k, "V seq_len mismatch");
    TORCH_CHECK(V.size(3) == head_dim, "V head_dim mismatch");

    // Get data pointers
    const half* Q_ptr = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    const half* K_ptr = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    const half* V_ptr = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    half* O_ptr = reinterpret_cast<half*>(O.data_ptr<at::Half>());

    const float* delta_s_ptr = nullptr;
    if (delta_s.has_value()) {
        TORCH_CHECK(delta_s.value().is_cuda(), "delta_s must be a CUDA tensor");
        TORCH_CHECK(delta_s.value().dtype() == torch::kFloat32, "delta_s must be float32");
        delta_s_ptr = delta_s.value().data_ptr<float>();
    }

    // Get current HIP stream
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    launch_sage_attention_forward(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        delta_s_ptr,
        batch_size, num_heads,
        seq_len_q, seq_len_k, head_dim,
        scale, is_causal,
        stream
    );

    // Check for errors
    HIP_CHECK(hipGetLastError());

    return O;
}

/**
 * Python binding for quantize_int4
 */
std::tuple<torch::Tensor, torch::Tensor> quantize_int4(
    torch::Tensor input,
    int block_size
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be float16");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");

    // Get dimensions
    const int batch_size = input.size(0);
    const int num_tokens = input.size(1);
    const int num_heads = input.dim() >= 3 ? input.size(2) : 1;
    const int head_dim = input.size(-1);

    // Calculate output dimensions
    const int output_dim = head_dim / 2; // Packed: 2 INT4 values per byte
    const int scale_dim = head_dim / block_size;

    // Allocate output tensors
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(input.device());
    torch::Tensor output = torch::zeros({batch_size, num_tokens, num_heads, output_dim}, options);

    auto scale_options = torch::TensorOptions().dtype(torch::kFloat16).device(input.device());
    torch::Tensor scales = torch::zeros({batch_size, num_tokens, num_heads, scale_dim}, scale_options);

    // Get strides
    const int stride_batch_in = input.stride(0);
    const int stride_token_in = input.stride(1);
    const int stride_head_in = input.dim() >= 3 ? input.stride(2) : 0;

    const int stride_batch_out = output.stride(0);
    const int stride_token_out = output.stride(1);
    const int stride_head_out = output.stride(2);

    const int stride_batch_scale = scales.stride(0);
    const int stride_token_scale = scales.stride(1);
    const int stride_head_scale = scales.stride(2);

    // Get data pointers
    const half* input_ptr = reinterpret_cast<const half*>(input.data_ptr<at::Half>());
    uint8_t* output_ptr = output.data_ptr<uint8_t>();
    half* scales_ptr = reinterpret_cast<half*>(scales.data_ptr<at::Half>());

    // Get current HIP stream
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    launch_quantize_int4(
        input_ptr, output_ptr, scales_ptr,
        batch_size, num_tokens, num_heads, head_dim,
        stride_batch_in, stride_token_in, stride_head_in,
        stride_batch_out, stride_token_out, stride_head_out,
        stride_batch_scale, stride_token_scale, stride_head_scale,
        stream
    );

    // Check for errors
    HIP_CHECK(hipGetLastError());

    return std::make_tuple(output, scales);
}

/**
 * Python binding for dequantize_int4
 */
torch::Tensor dequantize_int4(
    torch::Tensor input,
    torch::Tensor scales,
    std::vector<int64_t> output_shape
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");

    // Get dimensions
    const int batch_size = input.size(0);
    const int num_tokens = input.size(1);
    const int num_heads = input.size(2);
    const int head_dim = output_shape.back();

    // Allocate output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(input.device());
    torch::Tensor output = torch::zeros(output_shape, options);

    // Get strides
    const int stride_batch_in = input.stride(0);
    const int stride_token_in = input.stride(1);
    const int stride_head_in = input.stride(2);

    const int stride_batch_scale = scales.stride(0);
    const int stride_token_scale = scales.stride(1);
    const int stride_head_scale = scales.stride(2);

    const int stride_batch_out = output.stride(0);
    const int stride_token_out = output.stride(1);
    const int stride_head_out = output.stride(2);

    // Get data pointers
    const uint8_t* input_ptr = input.data_ptr<uint8_t>();
    const half* scales_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
    half* output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());

    // Get current HIP stream
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    launch_dequantize_int4(
        input_ptr, scales_ptr, output_ptr,
        batch_size, num_tokens, num_heads, head_dim,
        stride_batch_in, stride_token_in, stride_head_in,
        stride_batch_scale, stride_token_scale, stride_head_scale,
        stride_batch_out, stride_token_out, stride_head_out,
        stream
    );

    // Check for errors
    HIP_CHECK(hipGetLastError());

    return output;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SageAttention3 ROCm7 kernels for AMD RDNA3.5";

    m.def("attention_forward", &attention_forward,
          "Sage attention forward pass with optional INT4 quantization",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("O"),
          py::arg("scale"),
          py::arg("is_causal"),
          py::arg("delta_s") = py::none(),
          py::arg("use_int4_quantization") = false);

    m.def("quantize_int4", &quantize_int4,
          "Quantize FP16 tensor to INT4 with block scaling",
          py::arg("input"),
          py::arg("block_size") = 16);

    m.def("dequantize_int4", &dequantize_int4,
          "Dequantize INT4 tensor back to FP16",
          py::arg("input"),
          py::arg("scales"),
          py::arg("output_shape"));
}
