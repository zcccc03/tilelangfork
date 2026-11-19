"""用例：D05
二级目录：data_types
三级特性：FP16/BF16 输入与 FP32 累加
唯一场景：l1仅B类型与A不一致
验证约束：FP16/BF16不允许混合输入
责任：编译器：必须拦截
实际输入：A FP16[32,64]，B BF16[32,64]，C FP32[32,32]
预期输入：FP16/BF16不允许混合输入；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：Mixed-dtype GEMM
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/data_types/test_d05.py
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
    id="D05",
    group="data_types",
    feature="FP16/BF16 输入与 FP32 累加",
    scenario="l1仅B类型与A不一致",
    constraint="FP16/BF16不允许混合输入",
    responsibility="编译器：必须拦截",
    inputs="A FP16[32,64]，B BF16[32,64]，C FP32[32,32]",
    expected_inputs="FP16/BF16不允许混合输入；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：Mixed-dtype GEMM",
    kind="reject",
    stage="frontend",
    diagnostic="Mixed-dtype GEMM",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((32, 64), "float16"), B: T.Tensor((32, 64), "bfloat16"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((32, 64), "float16")
            b = T.alloc_l1((32, 64), "bfloat16")
            c = T.alloc_l0c((32, 32), "float32")
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
