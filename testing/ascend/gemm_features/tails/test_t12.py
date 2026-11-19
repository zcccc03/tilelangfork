"""用例：T12
二级目录：tails
三级特性：显式 L0 的有效 M/N
唯一场景：仅A有效M16与C32不一致
验证约束：A/C有效M必须相等
责任：编译器：必须拦截
实际输入：L0A/B物理BF16[32,64]；A选16行，C32行
预期输入：A/C有效M必须相等；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：M shape check failed
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t12.py
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
    id="T12",
    group="tails",
    feature="显式 L0 的有效 M/N",
    scenario="仅A有效M16与C32不一致",
    constraint="A/C有效M必须相等",
    responsibility="编译器：必须拦截",
    inputs="L0A/B物理BF16[32,64]；A选16行，C32行",
    expected_inputs="A/C有效M必须相等；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：M shape check failed",
    kind="reject",
    stage="frontend",
    diagnostic="M shape check failed",
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
            a = T.alloc_l0a((32, 64), "bfloat16")
            b = T.alloc_l0b((32, 64), "bfloat16")
            c = T.alloc_l0c((32, 32), "float32")
            T.gemm(a[:16, :], b, c, transpose_B=True, clear_accum=True)

    return main


def make_reference():
    return None


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
