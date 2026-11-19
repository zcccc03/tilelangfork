"""Persistent block-scaled GEMM, following example_gemm.py, with K=96 by default."""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def gemm(M_DIM=1024, K_DIM=96, N_DIM=1024, enable_unit_flag=True):
    """Compute C = scaled(X) @ scaled(W).T with E4M3 inputs and FP32 output.

    K_DIM is the actual compute extent. Every K32 group has an E8M0 scale.
    Scale storage packs two E8M0 bytes into uint16, with an identity scale
    in an unused final byte. Matrix data and compute K are never padded here.
    """
    NUM_BLOCKS = 32
    TILE_M = 256
    TILE_N = 256
    TILE_K = min(256, K_DIM)
    assert M_DIM > 0 and M_DIM % TILE_M == 0, "M must be a positive multiple of 256"
    assert N_DIM > 0 and N_DIM % TILE_N == 0, "N must be a positive multiple of 256"
    assert K_DIM > 0 and K_DIM % TILE_K == 0, "K must be divisible by the K tile"
    assert K_DIM % 32 == 0, "Input scale groups require K divisible by 32"
    SF_COLS = T.ceildiv(K_DIM, 64)
    TILE_SF_COLS = T.ceildiv(TILE_K, 64)
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    WINDOW = min(4, M_TILES)
    NUM_STAGES = 2
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW
    UF_2 = 2 if enable_unit_flag else 0
    UF_3 = 3 if enable_unit_flag else 0

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = tile_idx // WINDOW % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = tail_idx // TAIL_WIN % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile
        return (m_tile, n_tile)

    @T.prim_func
    def main(
        X: T.Tensor((M_DIM, K_DIM), "float8_e4m3fn"),
        W: T.Tensor((N_DIM, K_DIM), "float8_e4m3fn"),
        SFX: T.Tensor((SF_COLS, M_DIM), "uint16"),
        SFW: T.Tensor((SF_COLS, N_DIM), "uint16"),
        C: T.Tensor((M_DIM, N_DIM), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), "float8_e4m3fn")
            w_l1 = T.alloc_l1((TILE_N, TILE_K), "float8_e4m3fn")
            sx_l1 = T.alloc_l1((TILE_M, TILE_SF_COLS), "uint16")
            sw_l1 = T.alloc_l1((TILE_N, TILE_SF_COLS), "uint16")
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")
            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    T.copy(
                        SFX[kt * TILE_SF_COLS : (kt + 1) * TILE_SF_COLS, m_tile * TILE_M : (m_tile + 1) * TILE_M],
                        sx_l1,
                        transpose=True,
                    )
                    T.copy(
                        SFW[kt * TILE_SF_COLS : (kt + 1) * TILE_SF_COLS, n_tile * TILE_N : (n_tile + 1) * TILE_N],
                        sw_l1,
                        transpose=True,
                    )
                    T.gemm_blockscaled(
                        x_l1,
                        w_l1,
                        res,
                        sx_l1,
                        sw_l1,
                        transpose_B=True,
                        clear_accum=kt == 0,
                        unit_flag_ctrl=T.Select(kt == K_TILES - 1, UF_3, UF_2),
                    )
                T.dual_copy(res, temp, unit_flag_ctrl=UF_3)
                T.dual_copy(temp, C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N])

    return main


def make_inputs(M, K, N):
    generator = torch.Generator().manual_seed(20261119)
    x = (torch.randint(-4, 5, (M, K), generator=generator).float() / 8).to(torch.float8_e4m3fn)
    w = (torch.randint(-4, 5, (N, K), generator=generator).float() / 8).to(torch.float8_e4m3fn)
    scales = []
    for rows in (M, N):
        # Packing pads only scale storage. X and W still have exactly K columns.
        scale = torch.full((rows, 2 * ((K + 63) // 64)), 127, dtype=torch.uint8)
        scale[:, : K // 32] = torch.randint(125, 130, (rows, K // 32), generator=generator, dtype=torch.uint8)
        scales.append(scale.view(torch.uint16).T.contiguous())
    return x, w, *scales


def ref_program(x, w, sx, sw):
    sx = sx.T.contiguous().view(torch.uint8)
    sw = sw.T.contiguous().view(torch.uint8)
    x_scaled = x.float() * torch.exp2(sx.float() - 127).repeat_interleave(32, dim=1)[:, : x.shape[1]]
    w_scaled = w.float() * torch.exp2(sw.float() - 127).repeat_interleave(32, dim=1)[:, : w.shape[1]]
    return x_scaled @ w_scaled.T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=96)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    program = gemm(args.m, args.k, args.n)
    print("[Input IR]\n" + program.script(), flush=True)
    print(f"[Compile] M={args.m}, K={args.k}, N={args.n}", flush=True)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    print("[AscendC source]\n" + kernel.get_kernel_source(), flush=True)
    import torch_npu  # noqa: F401

    x, w, sx, sw = make_inputs(args.m, args.k, args.n)
    inputs = [value.npu() for value in (x, w, sx, sw)]
    print("[Execute] Launching block-scaled GEMM", flush=True)
    result = kernel(*inputs)
    torch.npu.synchronize()
    expected = ref_program(x, w, sx, sw)
    actual = result.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"block-scaled GEMM: correctness passed, max_diff={(actual - expected).abs().max().item()}")
    if args.bench:
        latency = do_bench(lambda: kernel(*inputs), backend="msprof", _n_warmup=30, _n_repeat=50)
        tflops = 2.0 * args.m * args.n * args.k / (latency * 1e9)
        print(f"block-scaled GEMM: {latency * 1000:.2f} us | {tflops:.1f} TFLOPS")
