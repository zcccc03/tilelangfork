"""Block-scaled NT GEMM: explicit FP8 L0 operands and E8M0 scale handles, FP32 output.

Run on Ascend 950: python examples/ascend/example_gemm_blockscaled_nt_l0.py
Generate Ascend source without running: add --compile-only.
Each row has one E8M0 scale per 32 K elements, packed in pairs as uint16.
Data and bound scale handles are explicitly loaded from L1 before Cube computation.
"""

import argparse

import tilelang
import tilelang.ascend.language as T


def gemm(M=512, N=1024, K=768, block_M=128, block_N=128, block_K=128):
    """Compute C = scaled(A) @ scaled(B).T; only full tiles are supported."""
    if any(value <= 0 for value in (M, N, K, block_M, block_N, block_K)):
        raise ValueError("Matrix and tile dimensions must be positive")
    if M % block_M or N % block_N or K % block_K:
        raise ValueError("M, N and K must be divisible by their tile dimensions")
    if block_K % 64:
        raise ValueError("Block-scaled GEMM requires block_K divisible by 64")
    sf_cols = K // 64
    tile_sf_cols = block_K // 64
    n_tiles = N // block_N
    num_tiles = (M // block_M) * n_tiles
    num_blocks = min(32, num_tiles)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float8_e4m3fn"),
        B: T.Tensor((N, K), "float8_e4m3fn"),
        SA: T.Tensor((sf_cols, M), "uint16"),
        SB: T.Tensor((sf_cols, N), "uint16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(num_blocks) as bx:
            A_l1 = T.alloc_l1((block_M, block_K), "float8_e4m3fn")
            B_l1 = T.alloc_l1((block_N, block_K), "float8_e4m3fn")
            SA_l1 = T.alloc_l1((block_M, tile_sf_cols), "uint16")
            SB_l1 = T.alloc_l1((block_N, tile_sf_cols), "uint16")
            A_l0a = T.alloc_l0a((block_M, block_K), "float8_e4m3fn")
            B_l0b = T.alloc_l0b((block_N, block_K), "float8_e4m3fn")
            # Each scale handle is bound to its data tile, not a separate UB buffer.
            SA_l0a = T.alloc_l0a_sf(A_l0a)
            SB_l0b = T.alloc_l0b_sf(B_l0b)
            C_l0c = T.alloc_l0c((block_M, block_N), "float32")

            for tile in T.Persistent([num_tiles], num_blocks, bx):
                m = (tile // n_tiles) * block_M
                n = (tile % n_tiles) * block_N
                for k in T.Pipelined(K // block_K, num_stages=0):
                    T.copy(A[m : m + block_M, k * block_K : (k + 1) * block_K], A_l1)
                    T.copy(B[n : n + block_N, k * block_K : (k + 1) * block_K], B_l1)
                    # GM scales are [K/64, rows]; L1 scales are [rows, tile_K/64].
                    T.copy(SA[k * tile_sf_cols : (k + 1) * tile_sf_cols, m : m + block_M], SA_l1, transpose=True)
                    T.copy(SB[k * tile_sf_cols : (k + 1) * tile_sf_cols, n : n + block_N], SB_l1, transpose=True)
                    # L1 -> L0 data, then load the associated hardware scale slots.
                    T.copy(A_l1, A_l0a)
                    T.copy(SA_l1, SA_l0a)
                    T.copy(B_l1, B_l0b)
                    T.copy(SB_l1, SB_l0b)
                    T.gemm_blockscaled(
                        A_l0a, B_l0b, C_l0c, SA_l0a, SB_l0b, transpose_B=True, clear_accum=(k == 0)
                    )
                T.copy(C_l0c, C[m : m + block_M, n : n + block_N])

    return main


def make_inputs(M, N, K):
    """Create exact FP8 fractions and non-identity, row/group-varying scales."""
    import torch

    generator = torch.Generator().manual_seed(0)
    a = (torch.randint(-4, 5, (M, K), generator=generator).float() / 8).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (N, K), generator=generator).float() / 8).to(torch.float8_e4m3fn)
    scales = []
    for rows in (M, N):
        # E8M0 byte e represents 2**(e-127). Pack two K32 groups per uint16.
        scale = torch.randint(125, 130, (rows, K // 32), generator=generator, dtype=torch.uint8)
        scales.append(scale.view(torch.uint16).T.contiguous())
    return a, b, *scales


def ref_program(a, b, sa, sb):
    """Decode packed E8M0 scales, apply each to its K32 group, then multiply."""
    import torch

    sa_bytes = sa.T.contiguous().view(torch.uint8)
    sb_bytes = sb.T.contiguous().view(torch.uint8)
    a_scaled = a.float() * torch.exp2(sa_bytes.float() - 127).repeat_interleave(32, dim=1)
    b_scaled = b.float() * torch.exp2(sb_bytes.float() - 127).repeat_interleave(32, dim=1)
    return a_scaled @ b_scaled.T


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

    inputs = make_inputs(M, N, K)
    expected = ref_program(*inputs)
    kernel = tilelang.compile(program, target="ascend", out_idx=[4])
    actual = kernel(*(value.npu() for value in inputs)).cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    max_error = (actual - expected).abs().max().item()
    print(f"L0 block-scaled NT GEMM correctness passed: C[{M}, {N}], K={K}")
    print(f"input=FP8 E4M3, scale=E8M0/K32, accumulation/output=float32, max_abs_error={max_error:.6g}")


if __name__ == "__main__":
    main()
