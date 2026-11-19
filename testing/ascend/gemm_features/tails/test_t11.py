"""用例：T11
二级目录：tails
三级特性：普通 GEMM 的有效 K 尾块
唯一场景：仅B的有效K64与A32不一致
验证约束：对应K必须相等
责任：编译器：必须拦截
实际输入：BF16 A/B物理[32,64]；有效A K32/B K64；C[32,32]
预期输入：对应K必须相等；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：K shape check failed
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t11.py
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
    id="T11",
    group="tails",
    feature="普通 GEMM 的有效 K 尾块",
    scenario="仅B的有效K64与A32不一致",
    constraint="对应K必须相等",
    responsibility="编译器：必须拦截",
    inputs="BF16 A/B物理[32,64]；有效A K32/B K64；C[32,32]",
    expected_inputs="对应K必须相等；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：K shape check failed",
    kind="reject",
    stage="frontend",
    diagnostic="K shape check failed",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((32, 64), "bfloat16"), B: T.Tensor((32, 64), "bfloat16"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((32, 64), "bfloat16")
            b = T.alloc_l1((32, 64), "bfloat16")
            c = T.alloc_l0c((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            a0 = T.alloc_l0a((32, 64), "bfloat16")
            b0 = T.alloc_l0b((32, 64), "bfloat16")
            T.copy(a, a0)
            T.copy(b, b0)
            T.gemm(a0[:, 0 : 0 + 32], b0[:, :64], c, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def make_reference():
    return None


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
