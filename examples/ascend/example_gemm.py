"""Persistent GEMM with pipelining, swizzling and an optional mixed-core epilogue."""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def gemm(
    M_DIM=8192,
    K_DIM=8192,
    N_DIM=8192,
    dtype="bfloat16",
    out_dtype="float32",
    MIXED=None,
    hf32=None,
    enable_unit_flag=True,
    acc=False,
    *,
    block_m=256,
    block_n=256,
    block_k=None,
    num_blocks=32,
    num_stages=2,
    swizzle_window=4,
):
    """Persistent, auto-scheduled GEMM: C = X @ W.T.

    MIXED selects a vector-core epilogue (automatic for FP32 output from BF16/FP8).
    hf32 selects FP32 input rounding: None, nearest_zero or nearest_even.
    acc accumulates into a pre-initialized output via store-mode atomics.
    Keyword-only tuning parameters control tile sizes, active cores, pipeline
    depth and swizzle locality. Shapes must be divisible by the chosen tiles.
    Tile/buffer capacities must be checked on the selected Ascend device.
    """
    # Keep the baseline tile sizes by default; expose tuning knobs for
    # different shapes/devices without changing the accumulation order silently.
    for name, value in (
        ("M", M_DIM),
        ("N", N_DIM),
        ("K", K_DIM),
        ("block_m", block_m),
        ("block_n", block_n),
        ("num_blocks", num_blocks),
        ("num_stages", num_stages),
        ("swizzle_window", swizzle_window),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    is_fp32 = dtype == "float32"
    if MIXED is None:
        MIXED = not is_fp32 and out_dtype != "bfloat16"
    TILE_M = block_m
    TILE_N = block_n
    TILE_K = (128 if is_fp32 else 256) if block_k is None else block_k
    if not isinstance(TILE_K, int) or TILE_K <= 0:
        raise ValueError(f"block_k must be a positive integer, got {TILE_K}")
    if TILE_M % 16 or TILE_N % 16 or TILE_K % 32:
        raise ValueError("block_m/block_n must be multiples of 16 and block_k of 32")
    if M_DIM % TILE_M or N_DIM % TILE_N or K_DIM % TILE_K:
        raise ValueError("M, N and K must be divisible by their tile sizes; tail tiles are unsupported")
    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    # Avoid launching idle cores for small output grids.
    NUM_BLOCKS = min(num_blocks, OUT_TILES)
    WINDOW = min(swizzle_window, M_TILES)
    NUM_STAGES = min(num_stages, K_TILES)
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW
    if enable_unit_flag:
        UF_2 = 2
        UF_3 = 3
    else:
        UF_2 = 0
        UF_3 = 0

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = T.min(tile_idx // N_TILES // WINDOW, MAIN_ROW)
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
    def main(X: T.Tensor((M_DIM, K_DIM), dtype), W: T.Tensor((N_DIM, K_DIM), dtype), C: T.Tensor((M_DIM, N_DIM), out_dtype)):
        with T.Kernel(NUM_BLOCKS) as bx:
            if is_fp32:
                T.set_hf32_mode(hf32)
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            if MIXED:
                temp = T.alloc_shared((TILE_M // 2, TILE_N), out_dtype)
            if acc:
                T.set_atomic("add", out_dtype)
            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(X[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                    T.copy(W[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                    T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=kt == 0, unit_flag_ctrl=T.Select(kt == K_TILES - 1, UF_3, UF_2))
                if MIXED:
                    T.dual_copy(res, temp, unit_flag_ctrl=UF_3)
                    T.dual_copy(temp, C[m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N])
                else:
                    T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N], unit_flag_ctrl=UF_3)
            if acc:
                T.set_atomic_none()

    return main


def ref_program(x, w, c=None, out_dtype="float32"):
    out = (x @ w.T).to(torch.bfloat16) if out_dtype == "bfloat16" else x.float() @ w.float().T
    return out if c is None else out + c


def run_regression_perf(M=8192, K=8192, N=8192, dtype="bfloat16", hf32=None, target=None, out_dtype="float32", acc=False, **kernel_config):
    # A persistent block runs many tiles; over-launching can leave a long tail.
    if "num_blocks" not in kernel_config:
        kernel_config["num_blocks"] = torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num
    program = gemm(M, K, N, dtype=dtype, out_dtype=out_dtype, hf32=hf32, acc=acc, **kernel_config)
    kernel = tilelang.compile(program, target=target, out_idx=None if acc else -1)
    a = torch.randn(M, K, device="npu").to(getattr(torch, dtype))
    b = torch.randn(N, K, device="npu").to(getattr(torch, dtype))
    inputs = (a, b, torch.zeros(M, N, device="npu", dtype=getattr(torch, out_dtype))) if acc else (a, b)
    repeats = 100 if dtype == "bfloat16" else 50
    latency = do_bench(lambda: kernel(*inputs), backend="msprof", _n_warmup=30, _n_repeat=repeats)
    tflops = 2.0 * M * N * K / (latency * 1e9)
    print(
        f"gemm (M={M}, N={N}, K={K}, {dtype}, out={out_dtype}, hf32={hf32}, acc={acc}, blocks={kernel_config['num_blocks']}): {latency * 1000:.2f} us | {tflops:.1f} TFLOPS"
    )
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=True, help="benchmark after correctness checks")
    parser.add_argument("--num-blocks", type=int, default=None, help="persistent blocks (default: device Cube core count)")
    args = parser.parse_args()
    num_blocks = (
        args.num_blocks if args.num_blocks is not None else torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num
    )
    if num_blocks <= 0:
        parser.error("--num-blocks must be positive")
    print(f"Persistent GEMM blocks: {num_blocks}")
    M, K, N = 8192, 8192, 8192
    # dtype, output dtype, HF32 rounding, accumulation, maximum absolute error
    cases = [
        ("float8_e4m3fn", "float32", None, False, 1e-1),
        ("bfloat16", "float32", None, False, 1e-2),
        ("float32", "float32", None, False, 5e-3),
        ("bfloat16", "bfloat16", None, False, 1e-2),
        ("float32", "float32", "nearest_even", False, 2e-1),
        ("bfloat16", "float32", None, True, 5e-3),
        ("bfloat16", "bfloat16", None, True, 1e-2),
    ]
    for dtype, out_dtype, hf32, acc, tolerance in cases:
        a = torch.randn(M, K, device="npu").to(getattr(torch, dtype))
        b = torch.randn(N, K, device="npu").to(getattr(torch, dtype))
        initial = torch.randn(M, N, device="npu", dtype=getattr(torch, out_dtype)) if acc else None
        program = gemm(M, K, N, dtype=dtype, out_dtype=out_dtype, hf32=hf32, acc=acc, num_blocks=num_blocks)
        kernel = tilelang.compile(program, out_idx=None if acc else -1)
        if acc:
            result = initial.clone()
            kernel(a, b, result)
        else:
            result = kernel(a, b)
        expected = ref_program(a, b, initial, out_dtype)
        max_diff = (result - expected).abs().max().item()
        assert max_diff < tolerance, f"max_diff={max_diff:.2e}, tolerance={tolerance:.2e}"
        print(f"gemm ({dtype}, out={out_dtype}, hf32={hf32}, acc={acc}): correctness passed")
        if args.bench:
            run_regression_perf(dtype=dtype, out_dtype=out_dtype, hf32=hf32, acc=acc, num_blocks=num_blocks)
