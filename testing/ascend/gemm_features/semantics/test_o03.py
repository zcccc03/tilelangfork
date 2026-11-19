"""用例：O03
二级目录：semantics
三级特性：L1 操作数的 NT GEMM
唯一场景：BF16 L1 M16合法边界
验证约束：只验证M对齐的合法输入
责任：编译器：数值实现；调用方：合法输入
实际输入：A(16, 64)/bfloat16，B(32, 64)/bfloat16；seed=20261119；L1/L0路径=l1
预期输入：只验证M对齐的合法输入；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[16,32]/float32
预期输出及验收标准：CPU按逻辑A/B转换FP32后相乘；atol=0, rtol=0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/semantics/test_o03.py
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
    id="O03",
    group="semantics",
    feature="L1 操作数的 NT GEMM",
    scenario="BF16 L1 M16合法边界",
    constraint="只验证M对齐的合法输入",
    responsibility="编译器：数值实现；调用方：合法输入",
    inputs="A(16, 64)/bfloat16，B(32, 64)/bfloat16；seed=20261119；L1/L0路径=l1",
    expected_inputs="只验证M对齐的合法输入；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[16,32]/float32",
    expected="CPU按逻辑A/B转换FP32后相乘；atol=0, rtol=0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "bfloat16"), B: T.Tensor((32, 64), "bfloat16"), C: T.Tensor((16, 32), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((16, 64), "bfloat16")
            b = T.alloc_l1((32, 64), "bfloat16")
            c = T.alloc_l0c((16, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_A=False, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 64), "bfloat16", gen) for rows in (16, 32)]
    expected = a[:, 0:64].float() @ b[:, 0:64].float().T
    inputs = [a, b]
    return (inputs, expected, None)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
