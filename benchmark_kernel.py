import argparse
import importlib
import statistics

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA GEMM kernels implemented using CUTLASS CuTe."
    )

    parser.add_argument(
        "--kernel",
        required=True,
        choices=["matmul_01", "matmul_02", "matmul_03", "matmul_04"],
        help="Kernel module to benchmark",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup launches per shape",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=200,
        help="Number of launches per timing sample",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use a minimal workload suitable for Nsight Compute",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------

def benchmark_kernel(fn, warmup=50, repeats=200):
    """
    Benchmark an already-compiled GPU kernel.

    Returns median, mean, minimum, and p95 latency in microseconds.
    Uses CUDA events to measure GPU execution time.
    """

    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    samples_us = []

    # Ten timing samples
    for _ in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(repeats):
            fn()

        end.record()
        end.synchronize()

        elapsed_ms = start.elapsed_time(end)
        latency_us = elapsed_ms * 1000.0 / repeats

        samples_us.append(latency_us)

    samples_us.sort()

    return {
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "min_us": samples_us[0],
        "p95_us": samples_us[int(0.95 * (len(samples_us) - 1))],
    }


def calculate_tflops(M, N, K, latency_us):
    """Calculate GEMM throughput assuming 2*M*N*K floating-point operations."""

    total_flops = 2 * M * N * K

    return total_flops / (latency_us * 1e-6) / 1e12


# ---------------------------------------------------------
# GEMM benchmark
# ---------------------------------------------------------

def benchmark(
    gemm,
    warmup=50,
    repeats=200,
    tests=None,
):
    """
    Compile and benchmark a GEMM implementation.

    gemm:
        The launch_gemm function imported from a kernel module.

    tests:
        List of (M, N, K) shapes.
    """

    if tests is None:
        tests = [ (4096, 4096, 4096) ]

    for M, N, K in tests:
        print(f"\nGEMM shape: M={M}, N={N}, K={K}")

        # Inputs and output
        A = torch.randn( (M, K), device="cuda", dtype=torch.float16 )

        B = torch.randn(
            (K, N),
            device="cuda",
            dtype=torch.float16,
        )

        C = torch.empty(
            (M, N),
            device="cuda",
            dtype=torch.float32,
        )

        # Convert PyTorch tensors to CuTe-compatible tensors
        a = from_dlpack(A, assumed_align=16)
        b = from_dlpack(B, assumed_align=16)
        c = from_dlpack(C, assumed_align=16)

        # Compile once per shape, outside the timed benchmark
        compiled = cute.compile(gemm, a, b, c, M, N, K)

        # Launch the compiled kernel
        def fn():
            compiled(a, b, c)

        stats = benchmark_kernel(
            fn,
            warmup=warmup,
            repeats=repeats,
        )

        print("Latency statistics:")
        for name, value in stats.items():
            print(f"  {name}: {value:.3f} us")

        tflops = calculate_tflops(
            M,
            N,
            K,
            stats["median_us"],
        )

        print(f"  TFLOP/s: {tflops:.4f}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Dynamically import the selected kernel
    module = importlib.import_module(
        f"kernels.{args.kernel}"
    )

    gemm = module.launch_gemm

    print(f"Selected kernel: {args.kernel}")

    if args.profile:
        # Minimal workload for Nsight Compute
        print("Mode: profiling")

        benchmark(
            gemm,
            warmup=1,
            repeats=1,
            tests=[
                (1024, 1024, 1024),
            ],
        )
    else:
        # Normal benchmarking
        print("Mode: benchmarking")

        benchmark(
            gemm,
            warmup=args.warmup,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
