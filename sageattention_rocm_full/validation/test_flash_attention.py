"""
Test Flash Attention and optimized kernels in PyTorch ROCm
"""

import torch
import torch.nn.functional as F
import time

def test_attention_backends():
    """Test which attention backends are available."""
    print("=" * 70)
    print("Testing Attention Backends in PyTorch ROCm")
    print("=" * 70)

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Compute Capability: {props.major}.{props.minor}")

    if not torch.cuda.is_available():
        print("\nCUDA not available, cannot test backends")
        return

    # Create test tensors
    batch = 1
    heads = 8
    seq_len = 512
    head_dim = 64

    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

    print("\n" + "-" * 70)
    print("Testing Available Backends:")
    print("-" * 70)

    backends_status = {}

    # Test Math backend (always available)
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            backends_status['Math'] = 'Available'
            print("Math backend: Available")
    except Exception as e:
        backends_status['Math'] = f'Failed: {e}'
        print(f"Math backend: Failed - {e}")

    # Test Flash Attention
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            backends_status['Flash'] = 'Available'
            print("Flash Attention: Available")
    except Exception as e:
        backends_status['Flash'] = 'Not available'
        print(f"Flash Attention: Not available - {e}")

    # Test Memory Efficient Attention
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=True
        ):
            output = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            backends_status['MemEfficient'] = 'Available'
            print("Memory Efficient: Available")
    except Exception as e:
        backends_status['MemEfficient'] = 'Not available'
        print(f"Memory Efficient: Not available - {e}")

    # Test default (auto-select)
    try:
        output = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        print("\nDefault backend: Works (auto-selects best)")
    except Exception as e:
        print(f"\nDefault backend: Failed - {e}")

    return backends_status


def benchmark_backends():
    """Benchmark different attention backends."""

    if not torch.cuda.is_available():
        print("\nBenchmarking requires CUDA")
        return

    print("\n" + "=" * 70)
    print("Benchmarking Attention Backends")
    print("=" * 70)

    configs = [
        (1, 8, 256, 64),
        (1, 16, 512, 128),
        (2, 16, 1024, 128),
    ]

    print(f"\n{'Config':<25} {'Math(ms)':<12} {'Flash(ms)':<12} {'MemEff(ms)':<12} {'Auto(ms)':<12}")
    print("-" * 73)

    for batch, heads, seq_len, head_dim in configs:
        config_str = f"B={batch},H={heads},L={seq_len}"

        q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

        times = {}

        # Benchmark Math
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                # Warmup
                for _ in range(3):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

                # Benchmark
                start = time.perf_counter()
                for _ in range(20):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                times['math'] = (time.perf_counter() - start) / 20 * 1000
        except:
            times['math'] = -1

        # Benchmark Flash
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                # Warmup
                for _ in range(3):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

                # Benchmark
                start = time.perf_counter()
                for _ in range(20):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                times['flash'] = (time.perf_counter() - start) / 20 * 1000
        except:
            times['flash'] = -1

        # Benchmark Memory Efficient
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                # Warmup
                for _ in range(3):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()

                # Benchmark
                start = time.perf_counter()
                for _ in range(20):
                    _ = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                times['memeff'] = (time.perf_counter() - start) / 20 * 1000
        except:
            times['memeff'] = -1

        # Benchmark Auto
        # Warmup
        for _ in range(3):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(20):
            _ = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        times['auto'] = (time.perf_counter() - start) / 20 * 1000

        # Format output
        math_str = f"{times['math']:.2f}" if times['math'] > 0 else "N/A"
        flash_str = f"{times['flash']:.2f}" if times['flash'] > 0 else "N/A"
        memeff_str = f"{times['memeff']:.2f}" if times['memeff'] > 0 else "N/A"
        auto_str = f"{times['auto']:.2f}"

        print(f"{config_str:<25} {math_str:<12} {flash_str:<12} {memeff_str:<12} {auto_str:<12}")


def main():
    # Test available backends
    backends = test_attention_backends()

    # Run benchmarks
    benchmark_backends()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if backends:
        available = [k for k, v in backends.items() if v == 'Available']
        if available:
            print(f"\nAvailable optimized backends: {', '.join(available)}")
            if 'Flash' in available or 'MemEfficient' in available:
                print("\n[SUCCESS] Optimized attention kernels are available!")
                print("PyTorch will automatically use the best backend.")
            else:
                print("\n[INFO] Only Math backend available (slower)")
        else:
            print("\n[WARNING] No backends working properly")

    print("\nFor SageAttention, we can leverage these backends for better performance.")


if __name__ == "__main__":
    main()