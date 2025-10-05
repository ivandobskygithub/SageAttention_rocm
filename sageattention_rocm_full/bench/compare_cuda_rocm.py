"""
Compare CUDA SageAttention with ROCm implementation
Provides side-by-side performance and correctness comparison
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd

# Add paths for both implementations
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # ROCm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # CUDA

# Flags to control what's available
HAS_CUDA_SAGE = False
HAS_ROCM_SAGE = False
HAS_FLASH_ATTN = False

# Try importing CUDA SageAttention
try:
    from sageattention import sageattn as sageattn_cuda
    from sageattention import sageattn_qk_int8_pv_fp16_cuda
    HAS_CUDA_SAGE = True
    print("✓ CUDA SageAttention available")
except ImportError as e:
    print(f"✗ CUDA SageAttention not available: {e}")

# Try importing ROCm SageAttention
try:
    from sageattention_rocm import sageattn as sageattn_rocm
    from sageattention_rocm import get_rocm_device_info
    from sageattention_rocm.core_triton_v2 import sageattn_optimized
    HAS_ROCM_SAGE = True
    print("✓ ROCm SageAttention available")
except ImportError as e:
    print(f"✗ ROCm SageAttention not available: {e}")

# Try importing FlashAttention
try:
    from flash_attn import flash_attn_func
    from flash_attn.utils.benchmark import benchmark_forward
    HAS_FLASH_ATTN = True
    print("✓ FlashAttention available")
except ImportError:
    print("✗ FlashAttention not available")


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    implementation: str
    seq_len: int
    time_ms: float
    tflops: float
    memory_mb: float
    is_causal: bool
    error_vs_baseline: Optional[float] = None


class ComprehensiveComparison:
    """Compare different attention implementations."""

    def __init__(self, device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.results = []

        # Check available implementations
        self.implementations = {}

        if HAS_CUDA_SAGE:
            self.implementations['cuda_sage'] = self._run_cuda_sage
            self.implementations['cuda_sage_triton'] = self._run_cuda_sage_triton

        if HAS_ROCM_SAGE:
            self.implementations['rocm_sage'] = self._run_rocm_sage
            self.implementations['rocm_sage_opt'] = self._run_rocm_sage_optimized

        if HAS_FLASH_ATTN:
            self.implementations['flash_attn'] = self._run_flash_attn

        # Always have PyTorch baseline
        self.implementations['pytorch'] = self._run_pytorch

        print(f"\nAvailable implementations: {list(self.implementations.keys())}")

    def _run_pytorch(self, q, k, v, is_causal=False):
        """PyTorch SDPA baseline."""
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    def _run_cuda_sage(self, q, k, v, is_causal=False):
        """CUDA SageAttention."""
        if not HAS_CUDA_SAGE:
            return None
        return sageattn_cuda(q, k, v, tensor_layout="NHD", is_causal=is_causal)

    def _run_cuda_sage_triton(self, q, k, v, is_causal=False):
        """CUDA SageAttention Triton backend."""
        if not HAS_CUDA_SAGE:
            return None
        from sageattention import sageattn_qk_int8_pv_fp16_triton
        return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal)

    def _run_rocm_sage(self, q, k, v, is_causal=False):
        """ROCm SageAttention."""
        if not HAS_ROCM_SAGE:
            return None
        return sageattn_rocm(q, k, v, tensor_layout="NHD", is_causal=is_causal)

    def _run_rocm_sage_optimized(self, q, k, v, is_causal=False):
        """ROCm SageAttention with optimizations."""
        if not HAS_ROCM_SAGE:
            return None
        return sageattn_optimized(q, k, v, is_causal=is_causal,
                                 use_fused_ops=True, pad_to_multiple=128)

    def _run_flash_attn(self, q, k, v, is_causal=False):
        """FlashAttention implementation."""
        if not HAS_FLASH_ATTN:
            return None
        # FlashAttention expects (batch, seqlen, nheads, headdim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        out = flash_attn_func(q_t, k_t, v_t, causal=is_causal)
        return out.transpose(1, 2)

    def benchmark_single(
        self,
        impl_name: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        num_warmup: int = 5,
        num_iters: int = 50
    ) -> Optional[BenchmarkResult]:
        """Benchmark a single implementation."""
        if impl_name not in self.implementations:
            return None

        impl_fn = self.implementations[impl_name]

        try:
            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = impl_fn(q, k, v, is_causal)
            torch.cuda.synchronize()

            # Measure time
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(num_iters):
                with torch.no_grad():
                    output = impl_fn(q, k, v, is_causal)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            # Calculate metrics
            batch, heads, seq_len, head_dim = q.shape
            time_ms = (elapsed / num_iters) * 1000
            flops = 4 * batch * heads * seq_len * seq_len * head_dim
            if is_causal:
                flops = flops // 2
            tflops = flops / (time_ms / 1000) / 1e12

            # Memory usage
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            # Error vs baseline (if not baseline itself)
            error = None
            if impl_name != 'pytorch':
                with torch.no_grad():
                    baseline = self._run_pytorch(q, k, v, is_causal)
                    diff = (output - baseline).abs()
                    error = diff.max().item()

            return BenchmarkResult(
                implementation=impl_name,
                seq_len=seq_len,
                time_ms=time_ms,
                tflops=tflops,
                memory_mb=memory_mb,
                is_causal=is_causal,
                error_vs_baseline=error
            )

        except Exception as e:
            print(f"  {impl_name} failed: {e}")
            return None

    def run_comparison(
        self,
        batch_size: int = 4,
        num_heads: int = 32,
        head_dim: int = 128,
        seq_lens: List[int] = None
    ):
        """Run comprehensive comparison."""
        if seq_lens is None:
            seq_lens = [512, 1024, 2048, 4096, 8192]

        print(f"\nConfiguration: B={batch_size}, H={num_heads}, D={head_dim}")
        print("=" * 80)

        # Results storage
        results_dict = {
            'non_causal': {},
            'causal': {}
        }

        for is_causal in [False, True]:
            print(f"\n{'Causal' if is_causal else 'Non-causal'} Attention:")
            print("-" * 60)

            for seq_len in seq_lens:
                print(f"\nSequence Length: {seq_len}")
                print(f"{'Implementation':>20} {'Time(ms)':>10} {'TFLOPS':>10} "
                      f"{'Memory(MB)':>12} {'Error':>10}")
                print("-" * 70)

                # Create test tensors
                q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                              dtype=self.dtype, device=self.device)
                k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                              dtype=self.dtype, device=self.device)
                v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                              dtype=self.dtype, device=self.device)

                seq_results = {}

                for impl_name in self.implementations:
                    result = self.benchmark_single(
                        impl_name, q, k, v, is_causal
                    )

                    if result:
                        self.results.append(result)
                        seq_results[impl_name] = result

                        error_str = f"{result.error_vs_baseline:.2e}" if result.error_vs_baseline else "baseline"
                        print(f"{impl_name:>20} {result.time_ms:>10.2f} "
                              f"{result.tflops:>10.2f} {result.memory_mb:>12.1f} "
                              f"{error_str:>10}")

                causal_key = 'causal' if is_causal else 'non_causal'
                results_dict[causal_key][seq_len] = seq_results

        return results_dict

    def plot_results(self, save_path: Optional[str] = None):
        """Plot comparison results."""
        if not self.results:
            print("No results to plot")
            return

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Implementation': r.implementation,
                'SeqLen': r.seq_len,
                'TFLOPS': r.tflops,
                'Time_ms': r.time_ms,
                'Causal': r.is_causal
            }
            for r in self.results
        ])

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: TFLOPS vs Sequence Length (Non-causal)
        ax = axes[0, 0]
        for impl in df['Implementation'].unique():
            data = df[(df['Implementation'] == impl) & (~df['Causal'])]
            ax.plot(data['SeqLen'], data['TFLOPS'], marker='o', label=impl)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('TFLOPS')
        ax.set_title('Performance: Non-causal Attention')
        ax.legend()
        ax.grid(True)

        # Plot 2: TFLOPS vs Sequence Length (Causal)
        ax = axes[0, 1]
        for impl in df['Implementation'].unique():
            data = df[(df['Implementation'] == impl) & (df['Causal'])]
            ax.plot(data['SeqLen'], data['TFLOPS'], marker='o', label=impl)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('TFLOPS')
        ax.set_title('Performance: Causal Attention')
        ax.legend()
        ax.grid(True)

        # Plot 3: Speedup vs PyTorch (Non-causal)
        ax = axes[1, 0]
        pytorch_data = df[(df['Implementation'] == 'pytorch') & (~df['Causal'])]
        for impl in df['Implementation'].unique():
            if impl == 'pytorch':
                continue
            impl_data = df[(df['Implementation'] == impl) & (~df['Causal'])]
            if len(impl_data) > 0 and len(pytorch_data) > 0:
                speedups = []
                seq_lens = []
                for seq_len in impl_data['SeqLen'].unique():
                    impl_time = impl_data[impl_data['SeqLen'] == seq_len]['Time_ms'].values
                    pytorch_time = pytorch_data[pytorch_data['SeqLen'] == seq_len]['Time_ms'].values
                    if len(impl_time) > 0 and len(pytorch_time) > 0:
                        speedup = pytorch_time[0] / impl_time[0]
                        speedups.append(speedup)
                        seq_lens.append(seq_len)
                if speedups:
                    ax.plot(seq_lens, speedups, marker='o', label=impl)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Speedup vs PyTorch')
        ax.set_title('Speedup: Non-causal Attention')
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True)

        # Plot 4: Memory Usage
        ax = axes[1, 1]
        for impl in df['Implementation'].unique():
            data = df[(df['Implementation'] == impl) & (~df['Causal'])]
            if 'Memory_MB' in df.columns:
                ax.plot(data['SeqLen'], data.get('Memory_MB', 0), marker='o', label=impl)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def generate_report(self, output_file: str = "comparison_report.md"):
        """Generate detailed comparison report."""
        with open(output_file, 'w') as f:
            f.write("# SageAttention CUDA vs ROCm Comparison Report\n\n")

            # System info
            f.write("## System Information\n\n")
            if torch.cuda.is_available():
                f.write(f"- Device: {torch.cuda.get_device_name(0)}\n")
                f.write(f"- PyTorch: {torch.__version__}\n")

            if HAS_ROCM_SAGE:
                device_info = get_rocm_device_info()
                if device_info:
                    f.write(f"- ROCm Device: {device_info['name']}\n")
                    f.write(f"- Architecture: {device_info.get('arch', 'unknown')}\n")

            # Available implementations
            f.write("\n## Available Implementations\n\n")
            for impl in self.implementations:
                f.write(f"- {impl}\n")

            # Performance summary
            if self.results:
                f.write("\n## Performance Summary\n\n")

                # Best performers by sequence length
                df = pd.DataFrame([{
                    'impl': r.implementation,
                    'seq': r.seq_len,
                    'tflops': r.tflops,
                    'causal': r.is_causal
                } for r in self.results])

                for causal in [False, True]:
                    f.write(f"\n### {'Causal' if causal else 'Non-causal'} Attention\n\n")
                    f.write("| Seq Length | Best Implementation | TFLOPS | Speedup vs PyTorch |\n")
                    f.write("|------------|-------------------|--------|-------------------|\n")

                    for seq_len in sorted(df['seq'].unique()):
                        seq_df = df[(df['seq'] == seq_len) & (df['causal'] == causal)]
                        if len(seq_df) > 0:
                            best = seq_df.loc[seq_df['tflops'].idxmax()]
                            pytorch = seq_df[seq_df['impl'] == 'pytorch']
                            if len(pytorch) > 0:
                                speedup = best['tflops'] / pytorch.iloc[0]['tflops']
                                f.write(f"| {seq_len:10d} | {best['impl']:17s} | "
                                       f"{best['tflops']:6.2f} | {speedup:17.2f}x |\n")

                # Correctness summary
                f.write("\n## Correctness Summary\n\n")
                f.write("| Implementation | Max Error vs PyTorch | Status |\n")
                f.write("|----------------|---------------------|--------|\n")

                impl_errors = {}
                for r in self.results:
                    if r.error_vs_baseline is not None:
                        if r.implementation not in impl_errors:
                            impl_errors[r.implementation] = []
                        impl_errors[r.implementation].append(r.error_vs_baseline)

                for impl, errors in impl_errors.items():
                    max_error = max(errors)
                    status = "✓ Pass" if max_error < 1e-2 else "⚠ Warning" if max_error < 1e-1 else "✗ Fail"
                    f.write(f"| {impl:14s} | {max_error:19.2e} | {status:7s} |\n")

            f.write("\n---\n*Generated by SageAttention ROCm Comparison Suite*\n")

        print(f"\nReport saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare CUDA and ROCm SageAttention')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=32, help='Number of heads')
    parser.add_argument('--head-dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--seq-lens', type=int, nargs='+',
                      default=[512, 1024, 2048, 4096],
                      help='Sequence lengths to test')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    parser.add_argument('--report', type=str, default='comparison_report.md',
                      help='Output report filename')

    args = parser.parse_args()

    # Check if any implementation is available
    if not any([HAS_CUDA_SAGE, HAS_ROCM_SAGE]):
        print("Error: No SageAttention implementation available")
        print("Please install either CUDA or ROCm SageAttention")
        return 1

    # Run comparison
    comparison = ComprehensiveComparison()
    results = comparison.run_comparison(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        seq_lens=args.seq_lens
    )

    # Generate report
    comparison.generate_report(args.report)

    # Plot if requested
    if args.plot or args.save_plot:
        try:
            comparison.plot_results(args.save_plot)
        except ImportError:
            print("Matplotlib/pandas not available for plotting")

    # Print summary
    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)
    print(f"Results saved to: {args.report}")

    return 0


if __name__ == "__main__":
    exit(main())