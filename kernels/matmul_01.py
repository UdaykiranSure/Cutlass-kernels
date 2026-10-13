import torch
import cutlass
import cutlass.cute as cute

"""
Naive Matmul implementation
"""

@cute.jit
def gemm_kernel(A, B, C, M:cutlass.Constexpr, N:cutalss.Constexpr, K:cutalss.constexpr):
    tid = cute.arch.thread_idx()

    row = tid // N
    col = tid % N

    if row < M and col < N:
        acc = 0.0
        for k in range(K):
            a = A[row, k]
            b = B[k, col]
            acc += a*b
        C[row, col] = acc
    