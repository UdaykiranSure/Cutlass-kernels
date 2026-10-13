import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


BM = 16
BN = 16
BK = 16

@cute.kernel 
def gemm_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    smem = cutlass.memory.SmemAllocator()

    a_layout = cute.make_layout(
        (BM, BK), stride = (BK, 1)
    )
    b_layout = cute.make_layout(
        (BK, BN), stride = (BN, 1)
    )

    sA = smem.allocate_tensor(cutlass.Float32, a_layout)
    sB = smem.allocate_tensor(cutlass.Float32, b_layout)

    local_row = tid // BN
    local_col = tid % BN

    row = by*BM + local_row
    col = bx*BN + local_col

    acc = cutlass.Float32(0.0)
    for kt in cutlass.range(cute.ceil_div(K, BK)):
        k0 = kt * BK

        ar = tid // BK
        ak = tid % BK

        global_ar = by * BM + ar
        global_ak = k0 + ak

        if global_ar < M and global_ak < K:
            sA[ar, ak] = A[global_ar, global_ak] 
        else:
            sA[ar, ak] = cutlass.Float32(0.0)

        bk = tid // BN
        bc = tid % BN

        global_bk = k0 + bk
        global_bc = bx * BN + bc

        if global_bk < K and global_bc < N:
            sB[bk, bc] = B[global_bk, global_bc]
        else:
            sB[bk, bc] = cutlass.Float32(0.0)

        cute.arch.sync_threads()

        for kk in cutlass.range_constexpr(BK):
            acc += sA[local_row, kk] * sB[kk, local_col]
        cute.arch.sync_threads()

    if row < M and col < N:
        C[row, col] = acc


@cute.jit
def launch_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr):
    gemm_kernel(A, B, C, M, N, K).launch(
        grid = (cute.ceil_div(N, BN), cute.ceil_div(M, BM), 1),
        block = (BM*BK, 1, 1)
    )


def run_gemm(A, B):
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    _, N = B.shape

    C = torch.empty((M, N), device =A.device, dtype = torch.float32)
    
    a = from_dlpack(A, assumed_align=16)
    b = from_dlpack(B, assumed_align=16)
    c = from_dlpack(C, assumed_align=16)

    compiled = cute.compile(launch_gemm, a, b, c, M, N, K)
    compiled(a, b, c)

    return C


if __name__ == "__main__":
    torch.manual_seed(0)

    tests = [(256,256,256), (1024, 1024, 1024), (1024, 4096, 1024), (1024, 1024, 4096)]

    for M, N, K in tests:
        A = torch.randn((M, K), device="cuda", dtype=torch.float16)
        B = torch.randn((K, N), device="cuda", dtype=torch.float16)

        C = run_gemm(A, B)
        reference = A.float() @ B.float()

        torch.cuda.synchronize()
        print(f"Max absolute error: ({M, N, K})",
            (C - reference).abs().max().item())
        print("Correct:", torch.allclose(C, reference, rtol=1e-4, atol=1e-4))

        
