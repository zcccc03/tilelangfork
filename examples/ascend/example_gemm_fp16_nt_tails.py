"""NT GEMM with M/N/K tails: FP16 inputs/output and FP32 L0C accumulation.

Run all cases on Ascend 950:
    python examples/ascend/example_gemm_fp16_nt_tails.py
Run one case or custom dimensions:
    python examples/ascend/example_gemm_fp16_nt_tails.py --case k
    python examples/ascend/example_gemm_fp16_nt_tails.py --m 531 --n 1041 --k 783
Use --compile-only for source generation, or --device-compile for device
compilation without executing the kernel. Neither mode verifies numerical results.
"""

import argparse

import tilelang
import tilelang.ascend.language as T


CASES = {
    "aligned": (512, 1024, 768),
    "m": (531, 1024, 768),
    "n": (512, 1041, 768),
    "k": (512, 1024, 783),
    "mnk": (531, 1041, 783),
}


def gemm(M=531, N=1041, K=783, block_M=128, block_N=64, block_K=32):
    """Compute A[M,K] @ B[N,K].T, including partial input/output tiles."""
    if any(value <= 0 for value in (M, N, K, block_M, block_N, block_K)):
        raise ValueError("Matrix and tile dimensions must be positive")
    if block_M % 16 or block_N % 16 or block_K % 16:
        raise ValueError("FP16 L1 GEMM tile dimensions must be multiples of 16; M/N/K may have tails")

    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    # Round up: every partial output tile and the final K chunk must execute.
    m_tiles = (M + block_M - 1) // block_M
    n_tiles = (N + block_N - 1) // block_N
    k_tiles = (K + block_K - 1) // block_K
    num_tiles = m_tiles * n_tiles
    num_blocks = min(32, num_tiles)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(num_blocks) as bx:
            A_l1 = T.alloc_l1((block_M, block_K), in_dtype)
            B_l1 = T.alloc_l1((block_N, block_K), in_dtype)
            C_l0c = T.alloc_l0c((block_M, block_N), accum_dtype)

            for tile in T.Persistent([num_tiles], num_blocks, bx):
                m = (tile // n_tiles) * block_M
                n = (tile % n_tiles) * block_N
                for k in T.Pipelined(k_tiles, num_stages=0):
                    # Request the full tile, even at the boundary. The Ascend
                    # OOB-padding pass clamps GM reads and zero-fills the L1
                    # tail. A shorter in-bounds slice alone would not request
                    # padding of the remaining L1 storage.
                    T.copy(A[m : m + block_M, k * block_K : (k + 1) * block_K], A_l1)
                    T.copy(B[n : n + block_N, k * block_K : (k + 1) * block_K], B_l1)
                    T.gemm(A_l1, B_l1, C_l0c, transpose_A=False, transpose_B=True, clear_accum=(k == 0))
                # The backend clamps the GM destination to valid M/N and
                # converts FP32 -> FP16 during L0C -> GM writeback.
                T.copy(C_l0c, C[m : m + block_M, n : n + block_N])

    return main


def run_case(name, shape, *, compile_only=False, device_compile=False):
    M, N, K = shape
    print(f"[{name}] M={M}, N={N}, K={K}; tile=128x64x32; remainder={M % 128}/{N % 64}/{K % 32}", flush=True)
    program = gemm(M, N, K)
    if compile_only or device_compile:
        artifact = tilelang.lower(program, target="ascend", enable_host_codegen=False, enable_device_compile=device_compile)
        print(artifact.kernel_source, flush=True)
        status = "DEVICE COMPILE PASSED" if device_compile else "SOURCE GENERATION PASSED"
        print(f"[{name}] {status}; numerical results have not been verified", flush=True)
        return

    import torch
    import torch_npu  # noqa: F401 -- register the NPU backend

    generator = torch.Generator().manual_seed(0)
    a = torch.randn((M, K), generator=generator, dtype=torch.float16)
    b = torch.randn((N, K), generator=generator, dtype=torch.float16)
    expected = (a.float() @ b.float().T).to(torch.float16)
    kernel = tilelang.compile(program, target="ascend", out_idx=None)
    # NaN output initialization also exposes output tiles missed by the
    # persistent traversal; the whole logical C is compared to the reference.
    actual_npu = torch.full((M, N), float("nan"), dtype=torch.float16, device="npu")
    kernel(a.npu(), b.npu(), actual_npu)
    actual = actual_npu.cpu()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    max_error = (actual.float() - expected.float()).abs().max().item()
    print(f"[{name}] NUMERICAL PASS; input=float16, accumulation=float32, output=float16, max_abs_error={max_error:.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--m", type=int, help="custom M (unspecified dimensions default to 512/1024/768)")
    parser.add_argument("--n", type=int, help="custom N")
    parser.add_argument("--k", type=int, help="custom K")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true", help="generate Ascend source without device compilation or execution")
    mode.add_argument("--device-compile", action="store_true", help="compile device code without numerical execution")
    args = parser.parse_args()
    if any(value is not None for value in (args.m, args.n, args.k)):
        shape = (args.m if args.m is not None else 512, args.n if args.n is not None else 1024, args.k if args.k is not None else 768)
        selected = {"custom": shape}
    else:
        selected = CASES if args.case == "all" else {args.case: CASES[args.case]}
    for name, shape in selected.items():
        run_case(name, shape, compile_only=args.compile_only, device_compile=args.device_compile)


if __name__ == "__main__":
    main()
