"""用例：O16
二级目录：semantics
三级特性：L0C 初始化与跨 K 分片累加
唯一场景：BF16两片K32逐片覆盖
验证约束：两片均clear；输出只等于最后一片
责任：调用方：选择clear；编译器：覆盖语义
实际输入：A(16, 64)/bfloat16，B(32, 64)/bfloat16；seed=20261119；L1/L0路径=l0
预期输入：两片均clear；输出只等于最后一片；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[16,32]/float32
预期输出及验收标准：只保留最后一个K分片的CPU乘积；atol=0, rtol=0
内核来源：testing/ascend/layout/test_ascend_l0_sliced_k.py（按本场景固定分片及类型）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/semantics/test_o16.py
正常结果只输出到终端；失败材料仅在指定 --failure-dir 时保存。
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
import tilelang.ascend.language as T
from testing.ascend.gemm_features._support import Case, pytest_case, standalone

SEED = 20261119
CASE = Case(
    id="O16",
    group="semantics",
    feature="L0C 初始化与跨 K 分片累加",
    scenario="BF16两片K32逐片覆盖",
    constraint="两片均clear；输出只等于最后一片",
    responsibility="调用方：选择clear；编译器：覆盖语义",
    inputs="A(16, 64)/bfloat16，B(32, 64)/bfloat16；seed=20261119；L1/L0路径=l0",
    expected_inputs="两片均clear；输出只等于最后一片；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[16,32]/float32",
    expected="只保留最后一个K分片的CPU乘积；atol=0, rtol=0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_ascend_l0_sliced_k.py（按本场景固定分片及类型）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "bfloat16"), B: T.Tensor((32, 64), "bfloat16"), C: T.Tensor((16, 32), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((16, 32), "bfloat16")
            b = T.alloc_l1((32, 32), "bfloat16")
            c = T.alloc_l0c((16, 32), "float32")
            a0 = T.alloc_l0a((16, 32), "bfloat16")
            b0 = T.alloc_l0b((32, 32), "bfloat16")
            for step in T.serial(2):
                T.copy(A[:, step * 32 : (step + 1) * 32], a)
                T.copy(B[:, step * 32 : (step + 1) * 32], b)
                T.copy(a, a0)
                T.copy(b, b0)
                T.gemm(a0, b0, c, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 64), "bfloat16", gen) for rows in (16, 32)]
    expected = a[:, 32:64].float() @ b[:, 32:64].float().T
    inputs = [a, b]
    return (inputs, expected, None)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
