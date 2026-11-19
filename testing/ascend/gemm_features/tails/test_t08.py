"""用例：T08
二级目录：tails
三级特性：显式 L0 MAD 的运行时有效尺寸
唯一场景：仅有效M33超过物理M32（F3）
验证约束：调用方保证有效范围不超过物理分配；本单元只观察编译器是否提供额外静态诊断
责任：调用方：保证范围合法；编译器：静态防护观察，不承诺自动检查
实际输入：L0A/C物理32行，A/C region33行；B与K均合法
预期输入：合法调用应满足M<=32，本单元故意只把region M设为33来观察静态防护；不执行设备计算。
实际输出契约：不产生数值输出
预期输出及验收标准：记录接受或目标越界诊断；接受时明确范围由调用方保障，不判为合法数值通过。
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t08.py
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
    id="T08",
    group="tails",
    feature="显式 L0 MAD 的运行时有效尺寸",
    scenario="仅有效M33超过物理M32（F3）",
    constraint="调用方保证有效范围不超过物理分配；本单元只观察编译器是否提供额外静态诊断",
    responsibility="调用方：保证范围合法；编译器：静态防护观察，不承诺自动检查",
    inputs="L0A/C物理32行，A/C region33行；B与K均合法",
    expected_inputs="合法调用应满足M<=32，本单元故意只把region M设为33来观察静态防护；不执行设备计算。",
    outputs="不产生数值输出",
    expected="记录接受或目标越界诊断；接受时明确范围由调用方保障，不判为合法数值通过。",
    kind="caller",
    stage="device_compile",
    diagnostic="(?:L0|region|extent).*(?:bound|exceed|capacity)|out.of.bounds",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((32, 64), "float16"), B: T.Tensor((32, 64), "float16"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((32, 64), "float16")
            b = T.alloc_l1((32, 64), "float16")
            a0 = T.alloc_l0a((32, 64), "float16")
            b0 = T.alloc_l0b((32, 64), "float16")
            c = T.alloc_l0c((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(a, a0)
            T.copy(b, b0)
            T.gemm(a0[:33, :], b0, c[:33, :], transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def make_reference():
    return None


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
