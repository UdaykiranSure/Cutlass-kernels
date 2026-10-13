import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

"""
Improved upton tiled version with multiple outputs/thread
Instead of one thread owning C[i,j]
it computes C[i+0, j], C[i+1, j], C[i+2, j], C[i+3, j] i.e 4 rows in its column

"""


@cute.kernel
def gemm_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    smem = cutlass.memory.SmemAllocator()

    a_layout = cute.make_layout(
        (32, 16), stride = (16, 1)
    )
    b_layout = cute.make_layout(
        (16, 32), stride = (32, 1)
    )

    sA = smem.allocate_tensor(cutlass.Float32, a_layout)
    sB = smem.allocate_tensor(cutlass.Float32, b_layout)

    col = bx * 32 + tid%32
    row0 = by * 32 + (tid//32) * 4

    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)
    acc2 = cutlass.Float32(0.0)
    acc3 = cutlass.Float32(0.0)

    for kt in cutlass.range_constexpr(cute.ceil_div(K, 16)):
        k0 = kt * 16

        for q in cutlass.range_constexpr(2):
            idx = tid + q * 256

            ar = idx // 16
            ak = idx % 16

            global_ar = by * 32 + ar
            global_ak = k0 + ak

            if global_ar < M and global_ak < K:
                sA[ar, ak] = A[global_ar, global_ak]
            else:
                sA[ar, ak] = cutlass.Float32(0.0)

            bk = idx // 32
            bc = idx % 32
            
            global_bk = k0 + bk
            global_bc = bx * 32 + bc

            if global_bk < K and global_bc < N:
                sB[bk, bc]  = B[global_bk, global_bc]  
            else:
                sB[bk, bc] = cutlass.Float32(0.0)
            
        cute.arch.sync_threads()

        for kk in cutlass.range_constexpr(16):
            bVal = sB[kk, (tid%32)]
            acc0 += sA[(tid//32)*4 + 0, kk] * bVal
            acc1 += sA[(tid//32)*4 + 1, kk] * bVal
            acc2 += sA[(tid//32)*4 + 2, kk] * bVal
            acc3 += sA[(tid//32)*4 + 3, kk] * bVal

        cute.arch.sync_threads()

    if row0 < M and col < N:
        C[row0 + 0, col] = acc0
    if row0 + 1 < M and col < N:
        C[row0 + 1, col] = acc1
    if row0 + 2 < M and col < N:
        C[row0 + 2, col] = acc2
    if row0 + 3 < M and col < N:    
        C[row0 + 3, col] = acc3


@cute.jit
def launch_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr):
    gemm_kernel(A, B, C, M, N, K).launch(
        grid = (cute.ceil_div(N, 32), cute.ceil_div(M, 32)),
        block = (32*8,1,1)
    )

def run_gemm(A, B):
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]
    assert A.is_contiguous() and B.is_contiguous()

    M,K = A.shape
    _,N = B.shape

    C = torch.empty((M, N),device = A.device, dtype = torch.float32)

    a = from_dlpack(A, assumed_align = 16)
    b = from_dlpack(B, assumed_align = 16)
    c = from_dlpack(C, assumed_align = 16)

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

        

