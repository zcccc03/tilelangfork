"""NT GEMM: explicit FP16 L0A/L0B operands, FP32 L0C, FP16 GM output.

Run on Ascend 950: python examples/ascend/example_gemm_fp16_nt_l0.py
Generate Ascend source without running: add --compile-only.
"""

import argparse

import tilelang
import tilelang.ascend.language as T


def gemm(M=512, N=1024, K=768, block_M=128, block_N=128, block_K=32):
    """Compute C = A @ B.T through GM -> L1 -> L0A/L0B -> L0C -> GM."""
    if any(value <= 0 for value in (M, N, K, block_M, block_N, block_K)):
        raise ValueError("Matrix and tile dimensions must be positive")
    if M % block_M or N % block_N or K % block_K:
        raise ValueError("M, N and K must be divisible by their tile dimensions")
    n_tiles = N // block_N
    num_tiles = (M // block_M) * n_tiles
    num_blocks = min(32, num_tiles)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(num_blocks) as bx:
            A_l1 = T.alloc_l1((block_M, block_K), T.float16)
            B_l1 = T.alloc_l1((block_N, block_K), T.float16)
            A_l0a = T.alloc_l0a((block_M, block_K), T.float16)
            B_l0b = T.alloc_l0b((block_N, block_K), T.float16)
            C_l0c = T.alloc_l0c((block_M, block_N), T.float32)

            for tile in T.Persistent([num_tiles], num_blocks, bx):
                m = (tile // n_tiles) * block_M
                n = (tile % n_tiles) * block_N
                for k in T.Pipelined(K // block_K, num_stages=0):
                    # GM -> L1. B is stored as [N, K] for NT GEMM.
                    T.copy(A[m : m + block_M, k * block_K : (k + 1) * block_K], A_l1)
                    T.copy(B[n : n + block_N, k * block_K : (k + 1) * block_K], B_l1)
                    # Explicitly stage both operands for the Cube instruction.
                    T.copy(A_l1, A_l0a)
                    T.copy(B_l1, B_l0b)
                    T.gemm(A_l0a, B_l0b, C_l0c, transpose_A=False, transpose_B=True, clear_accum=(k == 0))
                # Convert the FP32 accumulator to FP16 while writing to GM.
                T.copy(C_l0c, C[m : m + block_M, n : n + block_N])

    return main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true", help="print generated Ascend source without device execution")
    args = parser.parse_args()
    M, N, K = 512, 1024, 768
    program = gemm(M, N, K)

    if args.compile_only:
        artifact = tilelang.lower(program, target="ascend", enable_host_codegen=False, enable_device_compile=False)
        print(artifact.kernel_source)
        return

    import torch
    import torch_npu  # noqa: F401 -- register the NPU backend

    generator = torch.Generator().manual_seed(0)
    a = torch.randn((M, K), generator=generator, dtype=torch.float16)
    b = torch.randn((N, K), generator=generator, dtype=torch.float16)
    expected = (a.float() @ b.float().T).to(torch.float16)

    kernel = tilelang.compile(program, target="ascend", out_idx=[2])
    actual = kernel(a.npu(), b.npu()).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    max_error = (actual.float() - expected.float()).abs().max().item()
    print(f"L0 NT GEMM correctness passed: A[{M}, {K}] @ B[{N}, {K}].T -> C[{M}, {N}]")
    print(f"input=float16, accumulation=float32, output=float16, max_abs_error={max_error:.6g}")


if __name__ == "__main__":
    main()
