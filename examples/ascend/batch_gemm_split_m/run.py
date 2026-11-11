"""Validate the 28-batch, 512-cubed split-M GEMM on Ascend 950PR."""

import argparse
import sys
from pathlib import Path

import tilelang
import torch
from tilelang.ascend.callback import register_ascend_postproc_callback

BATCH = 28
DIM = 512
CORES = 28
KERNEL_PATH = Path(__file__).resolve().with_name("kernel.asc")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(KERNEL_PATH.parent.parent))
    from example_batch_gemm import batch_gemm

    kernel_source = KERNEL_PATH.read_text(encoding="utf-8")

    @register_ascend_postproc_callback
    def replace_kernel(source, target):
        signature = "void main_kernel(__gm__ float* C, __gm__ half* W, __gm__ half* X)"
        if signature not in source or "__cube__" not in source:
            raise RuntimeError("Unexpected Batch GEMM kernel ABI")
        return kernel_source

    kernel = tilelang.compile(
        batch_gemm(B_DIM=BATCH, M_DIM=DIM, N_DIM=DIM, K_DIM=DIM, MIXED=False, block_m=256, block_k=256, num_blocks=CORES), out_idx=-1
    )
    if kernel.get_kernel_source().strip() != kernel_source.strip():
        raise RuntimeError("The compiled kernel differs from kernel.asc")
    print("COMPILE_PASS", flush=True)
    if args.compile_only:
        return

    torch.set_num_threads(4)
    torch.manual_seed(0)
    x = torch.empty((BATCH, DIM, DIM), dtype=torch.float16)
    w = torch.empty_like(x)
    for operand in (x, w):
        for matrix in operand:
            matrix.copy_(torch.randint(-4, 5, (DIM, DIM), dtype=torch.int8))
            matrix.mul_(0.125)

    actual = kernel(x.npu(), w.npu()).cpu()
    for index in range(BATCH):
        expected = x[index].float() @ w[index].float().T
        torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
    print("CORRECTNESS_PASS max_error=0.0", flush=True)


if __name__ == "__main__":
    main()
