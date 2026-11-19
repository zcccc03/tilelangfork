"""用例：D22
二级目录：data_types
三级特性：INT8 输入与 INT32 累加
唯一场景：INT8 L1 K32合法C0边界
验证约束：子K按32对齐且整除总K
责任：编译器：数值实现；调用方：合法输入
实际输入：A(32, 32)/int8，B(32, 32)/int8；seed=20261119；L1/L0路径=l1
预期输入：子K按32对齐且整除总K；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[32,32]/int32
预期输出及验收标准：CPU按逻辑A/B以INT64计算完整K的乘积；确认参考值在INT32范围内后转成INT32；逐元素误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/data_types/test_d22.py
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
    id="D22",
    group="data_types",
    feature="INT8 输入与 INT32 累加",
    scenario="INT8 L1 K32合法C0边界",
    constraint="子K按32对齐且整除总K",
    responsibility="编译器：数值实现；调用方：合法输入",
    inputs="A(32, 32)/int8，B(32, 32)/int8；seed=20261119；L1/L0路径=l1",
    expected_inputs="子K按32对齐且整除总K；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[32,32]/int32",
    expected="CPU按逻辑A/B以INT64计算完整K的乘积；确认参考值在INT32范围内后转成INT32；逐元素误差0",
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
    def main(A: T.Tensor((32, 32), "int8"), B: T.Tensor((32, 32), "int8"), C: T.Tensor((32, 32), "int32")):
        with T.Kernel(1):
            a = T.alloc_l1((32, 32), "int8")
            b = T.alloc_l1((32, 32), "int8")
            c = T.alloc_l0c((32, 32), "int32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_A=False, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a = torch.randint(-128, 128, (32, 32), dtype=torch.int8, generator=gen)
    b = torch.randint(-128, 128, (32, 32), dtype=torch.int8, generator=gen)
    a[0, :] = -128
    b[0, :], b[1, :] = (-128, 127)
    wide = a[:, 0:32].long() @ b[:, 0:32].long().T
    assert wide.min() >= torch.iinfo(torch.int32).min and wide.max() <= torch.iinfo(torch.int32).max, "caller INT32 range obligation"
    expected = wide.int()
    inputs = [a, b]
    return (inputs, expected, None)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
