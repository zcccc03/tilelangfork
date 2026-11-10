"""Persistent batched GEMM: C[b] = X[b] @ W[b].T."""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench


def batch_gemm(
    B_DIM=4,
    M_DIM=1024,
    K_DIM=1024,
    N_DIM=1024,
    dtype="float16",
    out_dtype="float32",
    MIXED=None,
    hf32=None,
    enable_unit_flag=True,
    acc=False,
    *,
    block_m=128,
    block_n=256,
    block_k=None,
    num_blocks=28,
    num_stages=2,
    swizzle_window=4,
    prefetch_k=False,
):
    """Persistent batched GEMM with independent X/W matrices per batch.

    MIXED selects a vector-core epilogue (automatic for FP32 output from FP16/BF16/FP8).
    hf32 selects FP32 input rounding: None, nearest_zero or nearest_even.
    acc accumulates into a pre-initialized output via store-mode atomics.
    Keyword-only tuning parameters control tile sizes, active cores, pipeline
    depth and swizzle locality. Shapes must be divisible by the chosen tiles.
    Tile/buffer capacities must be checked on the selected Ascend device.
    The default 128x256 tile is tuned for FP16, B=4 and M=N=K=1024 on 28 cores.
    Use block_m=256 to reproduce the original tile configuration.
    prefetch_k is an opt-in two-step K prefetch for FP16-to-FP32 overwrite.
    It keeps two L1 slots, falls back for one K tile, and does not tune outputs.
    """
    # Use FP16 reference-workload defaults; expose tuning knobs for
    # different shapes/devices without changing the accumulation order silently.
    for name, value in (
        ("B", B_DIM),
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
    TILES_PER_BATCH = M_TILES * N_TILES
    OUT_TILES = B_DIM * TILES_PER_BATCH
    # Avoid launching idle cores for small output grids.
    NUM_BLOCKS = min(num_blocks, OUT_TILES)
    WINDOW = min(swizzle_window, M_TILES)
    NUM_STAGES = min(num_stages, K_TILES)
    if prefetch_k and (dtype != "float16" or out_dtype != "float32" or acc):
        raise ValueError("prefetch_k currently supports FP16-to-FP32 overwrite only")
    PREFETCH_K = prefetch_k and K_TILES > 1
    if PREFETCH_K:
        if NUM_STAGES < 2:
            raise ValueError("prefetch_k requires at least two pipeline stages")
        # Two physical L1 slots; fixed version annotations must fit the target.
        if 2 * (TILE_M + TILE_N) * TILE_K * 2 > 512 * 1024:
            raise ValueError("prefetch_k operand slots exceed the 512 KiB L1 budget")

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
    def main(
        X: T.Tensor((B_DIM, M_DIM, K_DIM), dtype), W: T.Tensor((B_DIM, N_DIM, K_DIM), dtype), C: T.Tensor((B_DIM, M_DIM, N_DIM), out_dtype)
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            if is_fp32:
                T.set_hf32_mode(hf32)
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            if PREFETCH_K:
                T.annotate_buffer_versions({x_l1: 2, w_l1: 2})
            if MIXED:
                temp = T.alloc_shared((TILE_M // 2, TILE_N), out_dtype)
            if acc:
                T.set_atomic("add", out_dtype)
            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                batch_idx = tile_idx // TILES_PER_BATCH
                m_tile, n_tile = aswt_swizzle(tile_idx % TILES_PER_BATCH)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES, annotations={"enable_offset": PREFETCH_K}):
                    if PREFETCH_K:
                        with T.Stage(0):
                            T.copy(X[batch_idx, m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                            T.copy(W[batch_idx, n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                        with T.Stage(2):
                            T.gemm(
                                x_l1,
                                w_l1,
                                res,
                                transpose_B=True,
                                clear_accum=kt == 0,
                                unit_flag_ctrl=T.Select(kt == K_TILES - 1, UF_3, UF_2),
                            )
                    else:
                        T.copy(X[batch_idx, m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], x_l1)
                        T.copy(W[batch_idx, n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K], w_l1)
                        T.gemm(
                            x_l1, w_l1, res, transpose_B=True, clear_accum=kt == 0, unit_flag_ctrl=T.Select(kt == K_TILES - 1, UF_3, UF_2)
                        )
                if MIXED:
                    T.dual_copy(res, temp, unit_flag_ctrl=UF_3)
                    T.dual_copy(temp, C[batch_idx, m_tile * TILE_M : (m_tile + 1) * TILE_M, n_tile * TILE_N : (n_tile + 1) * TILE_N])
                else:
                    T.copy(res, C[batch_idx, m_tile * TILE_M, n_tile * TILE_N], unit_flag_ctrl=UF_3)
            if acc:
                T.set_atomic_none()

    return main


def ref_program(x, w, c=None, out_dtype="float32"):
    out = (x.float() @ w.float().transpose(-1, -2)).to(getattr(torch, out_dtype))
    return out if c is None else out + c


def run_regression_perf(
    B=4, M=1024, K=1024, N=1024, dtype="float16", out_dtype="float32", hf32=None, acc=False, target=None, **kernel_config
):
    if "num_blocks" not in kernel_config:
        kernel_config["num_blocks"] = torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num
    program = batch_gemm(B, M, K, N, dtype=dtype, out_dtype=out_dtype, hf32=hf32, acc=acc, **kernel_config)
    kernel = tilelang.compile(program, target=target, out_idx=None if acc else -1)
    a = torch.randn(B, M, K).to(getattr(torch, dtype)).npu()
    b = torch.randn(B, N, K).to(getattr(torch, dtype)).npu()
    inputs = (a, b, torch.zeros(B, M, N, device="npu", dtype=getattr(torch, out_dtype))) if acc else (a, b)
    repeats = 100 if dtype == "bfloat16" else 50
    latency = do_bench(lambda: kernel(*inputs), backend="msprof", _n_warmup=30, _n_repeat=repeats)
    tflops = 2.0 * B * M * N * K / (latency * 1e9)
    print(
        f"batch_gemm ({dtype}, out={out_dtype}, hf32={hf32}, acc={acc}, blocks={kernel_config['num_blocks']}): {latency * 1000:.2f} us | {tflops:.1f} TFLOPS"
    )
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-blocks", type=int, default=None, help="default: device Cube core count")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float8_e4m3fn", "float32"], default="float16")
    parser.add_argument("--all-dtypes", action="store_true", help="opt in to precision/accumulation regression cases")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--swizzle-window", type=int, default=4)
    parser.add_argument("--mixed", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefetch-k", action="store_true", help="opt-in FP16 K-load overlap (two L1 slots)")
    args = parser.parse_args()
    kernel_config = dict(
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_stages=args.num_stages,
        swizzle_window=args.swizzle_window,
        MIXED=args.mixed,
    )
    num_blocks = (
        args.num_blocks if args.num_blocks is not None else torch.npu.get_device_properties(torch.npu.current_device()).cube_core_num
    )
    if num_blocks <= 0:
        parser.error("--num-blocks must be positive")
    M, K, N = args.m, args.k, args.n
    B = args.batch
    torch.manual_seed(0)
    cases = [
        ("float16", "float32", None, False),
        ("float8_e4m3fn", "float32", None, False),
        ("bfloat16", "float32", None, False),
        ("float32", "float32", None, False),
        ("bfloat16", "bfloat16", None, False),
        ("float32", "float32", "nearest_even", False),
        ("bfloat16", "float32", None, True),
        ("bfloat16", "bfloat16", None, True),
    ]
    if not args.all_dtypes:
        cases = [(args.dtype, "float32", None, False)]
    print(f"CONFIG: FP16-first; blocks={num_blocks}; cases={cases}", flush=True)
    for dtype, out_dtype, hf32, acc in cases:
        # CPU setup avoids unrelated random/reference kernels in simulator captures.
        a = (torch.randint(-4, 5, (B, M, K), dtype=torch.int8).float() / 8).to(getattr(torch, dtype))
        b = (torch.randint(-4, 5, (B, N, K), dtype=torch.int8).float() / 8).to(getattr(torch, dtype))
        initial = torch.ones(B, M, N, dtype=getattr(torch, out_dtype)) if acc else None
        kernel_config["prefetch_k"] = args.prefetch_k and dtype == "float16" and out_dtype == "float32" and not acc
        program = batch_gemm(B, M, K, N, dtype=dtype, out_dtype=out_dtype, hf32=hf32, acc=acc, num_blocks=num_blocks, **kernel_config)
        kernel = tilelang.compile(program, out_idx=None if acc else -1)
        if acc:
            output = initial.npu()
            kernel(a.npu(), b.npu(), output)
            result = output.cpu()
        else:
            result = kernel(a.npu(), b.npu()).cpu()
        expected = ref_program(a, b, initial, out_dtype)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        print(f"batch_gemm ({dtype}, out={out_dtype}, hf32={hf32}, acc={acc}): correctness passed", flush=True)
        if args.bench:
            run_regression_perf(B, M, K, N, dtype=dtype, out_dtype=out_dtype, hf32=hf32, acc=acc, num_blocks=num_blocks, **kernel_config)
