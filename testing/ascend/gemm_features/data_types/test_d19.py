"""用例：D19
二级目录：data_types
三级特性：INT8 输入与 INT32 累加
唯一场景：INT8 L1仅M17非法
验证约束：M必须为16倍数
责任：设备编译器：模板static_assert
实际输入：INT8 A[17,64]/B[32,64]，INT32 C[17,32]
预期输入：M必须为16倍数；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：M must be a multiple of 16
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/data_types/test_d19.py
正常结果只输出到终端；失败材料仅在指定 --failure-dir 时保存。
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import tilelang.ascend.language as T
from testing.ascend.gemm_features._support import Case, pytest_case, standalone

SEED = 20261119
CASE = Case(
    id="D19",
    group="data_types",
    feature="INT8 输入与 INT32 累加",
    scenario="INT8 L1仅M17非法",
    constraint="M必须为16倍数",
    responsibility="设备编译器：模板static_assert",
    inputs="INT8 A[17,64]/B[32,64]，INT32 C[17,32]",
    expected_inputs="M必须为16倍数；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：M must be a multiple of 16",
    kind="reject",
    stage="device_compile",
    diagnostic="M must be a multiple of 16",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((17, 64), "int8"), B: T.Tensor((32, 64), "int8"), C: T.Tensor((17, 32), "int32")):
        with T.Kernel(1):
            a = T.alloc_l1((17, 64), "int8")
            b = T.alloc_l1((32, 64), "int8")
            c = T.alloc_l0c((17, 32), "int32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_A=False, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def make_reference():
    return None


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
