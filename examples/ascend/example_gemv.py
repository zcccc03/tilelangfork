"""Cube GEMV: Y = X @ W.T, with FP16 inputs and FP32 output."""

import argparse
import json
from pathlib import Path

import tilelang
import tilelang.ascend.language as T
from tilelang.env import TILELANG_TEMPLATE_PATH


def gemv(N_DIM=32, K_DIM=256):
    """Use one Cube tile and let the L1 GEMM backend split the reduction."""
    if N_DIM <= 0 or N_DIM % 16 or K_DIM <= 0 or K_DIM % 256:
        raise ValueError("GEMV requires positive N and K, aligned to 16 and 256 respectively")

    @T.prim_func
    def main(
        X: T.Tensor((1, K_DIM), "float16"),
        W: T.Tensor((N_DIM, K_DIM), "float16"),
        Y: T.Tensor((1, N_DIM), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((1, N_DIM), "float32")
            x_l1 = T.alloc_l1((1, K_DIM), "float16")
            w_l1 = T.alloc_l1((N_DIM, K_DIM), "float16")
            T.copy(X, x_l1)
            T.copy(W, w_l1)
            T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=True)
            T.copy(res, Y)

    return main


def ref_program(x, w):
    return x.float() @ w.float().T


def export_ir(program, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "01_input.tir").write_text(program.script(), encoding="utf-8")
    with tilelang.tvm.target.Target("ascend"):
        lowered = tilelang.lower(program, target="ascend", enable_host_codegen=False, enable_device_compile=False)
    (out_dir / "02_device.tir").write_text(lowered.device_mod.script(), encoding="utf-8")
    (out_dir / "03_kernel.asc").write_text(lowered.kernel_source, encoding="utf-8")
    template = Path(TILELANG_TEMPLATE_PATH) / "tl_templates" / "ascend" / "gemm.h"
    (out_dir / "04_gemm_template.h").write_bytes(template.read_bytes())
    return template


def check_npu(program, n, k):
    import torch

    generator = torch.Generator().manual_seed(0)
    x = (torch.randint(-4, 5, (1, k), generator=generator).half() / 8).contiguous()
    w = (torch.randint(-4, 5, (n, k), generator=generator).half() / 8).contiguous()
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    actual = kernel(x.npu(), w.npu()).cpu()
    torch.testing.assert_close(actual, ref_program(x, w), rtol=0, atol=0)
    return (actual - ref_program(x, w)).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--check-npu", action="store_true", help="also compare the result on a physical NPU")
    args = parser.parse_args()

    program = gemv(args.n, args.k)
    template = export_ir(program, args.out_dir)
    record = {
        "m": 1,
        "n": args.n,
        "k": args.k,
        "tilelang_path": str(Path(tilelang.__file__).resolve()),
        "template_path": str(template.resolve()),
        "npu_numerical_verified": False,
    }
    record_path = args.out_dir / "export.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if args.check_npu:
        max_error = check_npu(program, args.n, args.k)
        record.update(npu_numerical_verified=True, max_error=max_error)
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"NPU numerical PASS: max_error={max_error:g}")
    print(f"IR and Ascend source: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
