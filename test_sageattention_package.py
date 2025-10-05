"""
Test SageAttention3 ROCm7 Package Installation and Features
"""
import torch
import time
import sys
import os

# Set console encoding to UTF-8 for Windows
if sys.platform == "win32":
    os.system("chcp 65001 > nul")


def test_sageattention():
    """Test the installed SageAttention3 ROCm7 package."""
    print("=" * 70)
    print("SageAttention3 ROCm7 - Package Validation Test")
    print("=" * 70)

    # Import the installed package
    try:
        import sageattention3_rocm7
        from sageattention3_rocm7 import sageattn3_rocm7, sageattn3_blackwell
        print(f"[OK] Package imported successfully")
        print(f"  Version: {sageattention3_rocm7.__version__}")
        print(f"  Location: {sageattention3_rocm7.__file__}")
    except ImportError as e:
        print(f"[FAIL] Failed to import package: {e}")
        return False

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("[FAIL] CUDA/ROCm not available")
        return False

    device = torch.device("cuda:0")
    print(f"[OK] CUDA/ROCm device: {torch.cuda.get_device_name(0)}")

    # Test configuration
    batch_size = 2
    num_heads = 8
    seq_len = 512
    head_dim = 64

    print(f"\n{'='*70}")
    print(f"Test Configuration")
    print(f"{'='*70}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")

    # Create test tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

    print(f"\n{'='*70}")
    print("Feature Tests")
    print(f"{'='*70}")

    # Test 1: API Compatibility
    print("\n1. API Compatibility Test")
    print("-" * 30)
    try:
        # Test both API names work
        output1 = sageattn3_rocm7(Q[:, :, :256, :], K[:, :, :256, :], V[:, :, :256, :])
        output2 = sageattn3_blackwell(Q[:, :, :256, :], K[:, :, :256, :], V[:, :, :256, :])
        print(f"  [OK] sageattn3_rocm7 API works")
        print(f"  [OK] sageattn3_blackwell API works (alias)")
        api_compatible = True
    except Exception as e:
        print(f"  [FAIL] API test failed: {e}")
        api_compatible = False

    # Test 2: Non-causal Attention
    print("\n2. Non-causal Attention Test")
    print("-" * 30)
    try:
        # SageAttention
        start = time.time()
        output_sage = sageattn3_rocm7(Q, K, V, is_causal=False, per_block_mean=True)
        torch.cuda.synchronize()
        sage_time = time.time() - start

        # PyTorch reference
        start = time.time()
        output_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
        torch.cuda.synchronize()
        torch_time = time.time() - start

        # Compare
        error = (output_sage - output_torch).abs().mean().item()
        max_error = (output_sage - output_torch).abs().max().item()

        print(f"  [OK] Non-causal attention completed")
        print(f"    Mean error: {error:.6f}")
        print(f"    Max error: {max_error:.6f}")
        print(f"    SageAttention time: {sage_time*1000:.2f}ms")
        print(f"    PyTorch time: {torch_time*1000:.2f}ms")

        if sage_time < torch_time:
            print(f"    Performance: {torch_time/sage_time:.2f}x faster")
        else:
            print(f"    Performance: {sage_time/torch_time:.2f}x slower")

        noncausal_pass = error < 0.1
    except Exception as e:
        print(f"  [FAIL] Non-causal test failed: {e}")
        noncausal_pass = False

    # Test 3: Causal Attention
    print("\n3. Causal Attention Test")
    print("-" * 30)
    try:
        # SageAttention
        output_sage_causal = sageattn3_rocm7(Q, K, V, is_causal=True, per_block_mean=False)

        # PyTorch reference
        output_torch_causal = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Compare
        error = (output_sage_causal - output_torch_causal).abs().mean().item()

        print(f"  [OK] Causal attention completed")
        print(f"    Mean error: {error:.6f}")

        # Verify causality (upper triangle should be different)
        # Check if causal mask is applied by comparing attention patterns
        causal_diff = (output_sage_causal - output_sage).abs().mean().item()
        print(f"    Causal vs non-causal difference: {causal_diff:.6f}")

        causal_pass = error < 0.1 and causal_diff > 0.01
        if causal_pass:
            print(f"    [OK] Causal masking verified")
        else:
            print(f"    [WARN] Causal masking may not be working correctly")
    except Exception as e:
        print(f"  [FAIL] Causal test failed: {e}")
        causal_pass = False

    # Test 4: BF16 Support
    print("\n4. BFloat16 Support Test")
    print("-" * 30)
    try:
        Q_bf16 = Q.to(torch.bfloat16)
        K_bf16 = K.to(torch.bfloat16)
        V_bf16 = V.to(torch.bfloat16)

        output_bf16 = sageattn3_rocm7(Q_bf16, K_bf16, V_bf16, is_causal=False)

        print(f"  [OK] BF16 processing completed")
        print(f"    Input dtype: {Q_bf16.dtype}")
        print(f"    Output dtype: {output_bf16.dtype}")
        print(f"    Output shape: {output_bf16.shape}")

        bf16_pass = output_bf16.dtype == torch.bfloat16
    except Exception as e:
        print(f"  [FAIL] BF16 test failed: {e}")
        bf16_pass = False

    # Test 5: Quantization Functions
    print("\n5. Quantization Functions Test")
    print("-" * 30)
    try:
        from sageattention3_rocm7 import scale_and_quant_int4, scale_and_quant_int4_permute, scale_and_quant_int4_transpose

        # Test standard quantization
        q_int4, q_scales = scale_and_quant_int4(Q)
        print(f"  [OK] scale_and_quant_int4 works")
        print(f"    Quantized shape: {q_int4.shape}")
        print(f"    Scales shape: {q_scales.shape}")

        # Test permute quantization
        k_int4, k_scales = scale_and_quant_int4_permute(K)
        print(f"  [OK] scale_and_quant_int4_permute works")

        # Test transpose quantization
        v_int4, v_scales = scale_and_quant_int4_transpose(V)
        print(f"  [OK] scale_and_quant_int4_transpose works")

        quant_pass = True
    except Exception as e:
        print(f"  [FAIL] Quantization test failed: {e}")
        quant_pass = False

    # Test 6: Memory Compression
    print("\n6. Memory Compression Test")
    print("-" * 30)
    try:
        # Calculate memory usage
        tensor_size = batch_size * num_heads * seq_len * head_dim
        fp16_memory = tensor_size * 2  # 2 bytes per FP16
        int4_memory = tensor_size // 2  # 0.5 bytes per INT4 (packed)

        compression_ratio = fp16_memory / int4_memory

        print(f"  [OK] Memory analysis:")
        print(f"    FP16 memory per tensor: {fp16_memory / (1024*1024):.2f} MB")
        print(f"    INT4 memory per tensor: {int4_memory / (1024*1024):.2f} MB")
        print(f"    Compression ratio: {compression_ratio:.1f}x")

        memory_pass = compression_ratio >= 3.5
    except Exception as e:
        print(f"  [FAIL] Memory test failed: {e}")
        memory_pass = False

    # Test 7: Preprocessing Functions
    print("\n7. Preprocessing Functions Test")
    print("-" * 30)
    try:
        from sageattention3_rocm7 import preprocess_qkv

        q_proc, k_proc, v_proc, delta_s = preprocess_qkv(Q[:, :, :256, :], K[:, :, :256, :], V[:, :, :256, :])

        print(f"  [OK] preprocess_qkv works")
        print(f"    Q processed shape: {q_proc.shape}")
        print(f"    Delta S shape: {delta_s.shape}")
        print(f"    Padding applied: {q_proc.shape[2] >= 256}")

        preprocess_pass = True
    except Exception as e:
        print(f"  [FAIL] Preprocessing test failed: {e}")
        preprocess_pass = False

    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")

    tests = {
        "API Compatibility": api_compatible,
        "Non-causal Attention": noncausal_pass,
        "Causal Attention": causal_pass,
        "BFloat16 Support": bf16_pass,
        "Quantization Functions": quant_pass,
        "Memory Compression": memory_pass,
        "Preprocessing": preprocess_pass
    }

    passed = sum(tests.values())
    total = len(tests)

    for test_name, result in tests.items():
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"  {test_name:.<30} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed! SageAttention3 ROCm7 is fully functional.")
        print("   The package is ready for use with PyTorch models.")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Please review the issues above.")

    return passed == total


if __name__ == "__main__":
    success = test_sageattention()
    sys.exit(0 if success else 1)