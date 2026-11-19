"""用例：D27
二级目录：data_types
三级特性：FP32 GEMM 与 HF32 模式
唯一场景：仅模式字符串nearest_up非法
验证约束：只允许None/nearest_zero/nearest_even
责任：编译器：必须拦截
实际输入：set_hf32_mode('nearest_up')
预期输入：只允许None/nearest_zero/nearest_even；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：hf32 mode must be
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/data_types/test_d27.py
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
    id="D27",
    group="data_types",
    feature="FP32 GEMM 与 HF32 模式",
    scenario="仅模式字符串nearest_up非法",
    constraint="只允许None/nearest_zero/nearest_even",
    responsibility="编译器：必须拦截",
    inputs="set_hf32_mode('nearest_up')",
    expected_inputs="只允许None/nearest_zero/nearest_even；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：hf32 mode must be",
    kind="reject",
    stage="frontend",
    diagnostic="hf32 mode must be",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main():
        with T.Kernel(1):
            T.set_hf32_mode("nearest_up")

    return main


def make_reference():
    return None


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
