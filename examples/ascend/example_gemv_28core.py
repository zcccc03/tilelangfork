"""28-Cube FP16 GEMV with configurable L1 buffering and FP32 output."""

import argparse
from pathlib import Path

import tilelang.ascend.language as T

from example_gemv import check_npu, export_ir


def gemv_28core(n=1792, k=6144, tile_n=64, tile_k=768, buffers=2):
    cores = 28
    if min(n, k, tile_n, tile_k) <= 0 or tile_n % 16 or tile_k % 256:
        raise ValueError("Positive tiles must align N to 16 and K to 256")
    if n % tile_n or k % tile_k:
        raise ValueError("N and K must be divisible by their tile sizes (no tails)")
    if buffers not in (1, 2, 4) or k // tile_k < buffers:
        raise ValueError("Use 1, 2 or 4 buffers and at least that many K tiles")
    if buffers * 2 * (tile_n + 1) * tile_k > 512 * 1024:
        raise ValueError("L1 slots exceed 512 KiB")
    if 2 * tile_n * 256 > 32 * 1024:
        raise ValueError("A 256-element GEMV sub-K must fit the current 32 KiB L0B tile budget")
    k_tiles = k // tile_k

    @T.prim_func
    def main(X: T.Tensor((1, k), "float16"), W: T.Tensor((n, k), "float16"), Y: T.Tensor((1, n), "float32")):
        with T.Kernel(cores) as bx:
            x = T.alloc_l1((1, tile_k), "float16")
            w = T.alloc_l1((tile_n, tile_k), "float16")
            acc = T.alloc_l0c((1, tile_n), "float32")
            # Each in-flight K tile owns one X slot and one W slot in L1.
            # Keep one FP32 accumulator for all K tiles of this output block.
            if buffers > 1:
                T.annotate_buffer_versions({x: buffers, w: buffers})
            for nt in T.Persistent([n // tile_n], cores, bx):
                if buffers == 1:
                    for kt in T.serial(k_tiles):
                        T.copy(X[0:1, kt * tile_k : (kt + 1) * tile_k], x)
                        T.copy(W[nt * tile_n : (nt + 1) * tile_n, kt * tile_k : (kt + 1) * tile_k], w)
                        T.gemm(x, w, acc, transpose_B=True, clear_accum=kt == 0)
                else:
                    for kt in T.Pipelined(k_tiles, num_stages=buffers, annotations={"enable_offset": True}):
                        with T.Stage(0):
                            T.copy(X[0:1, kt * tile_k : (kt + 1) * tile_k], x)
                            T.copy(W[nt * tile_n : (nt + 1) * tile_n, kt * tile_k : (kt + 1) * tile_k], w)
                        with T.Stage(2):
                            T.gemm(x, w, acc, transpose_B=True, clear_accum=kt == 0)
                T.copy(acc, Y[0:1, nt * tile_n : (nt + 1) * tile_n])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1792)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=768)
    parser.add_argument("--buffers", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--check-npu", action="store_true")
    args = parser.parse_args()
    program = gemv_28core(args.n, args.k, args.tile_n, args.tile_k, args.buffers)
    export_ir(program, args.out_dir)
    if args.check_npu:
        error = check_npu(program, args.n, args.k)
        print(f"NPU numerical PASS: max_error={error:g}")
