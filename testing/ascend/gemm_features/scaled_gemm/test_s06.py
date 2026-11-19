"""用例：S06
二级目录：scaled_gemm
三级特性：L1 输入的块缩放 GEMM
唯一场景：仅transpose_B=False
验证约束：L1入口只支持NT
责任：编译器：必须拦截
实际输入：A[32,128], B[128,32] E4M3；scale匹配；M/N/K均合法
预期输入：L1入口只支持NT；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：only supports .*NT
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s06.py
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
    id="S06",
    group="scaled_gemm",
    feature="L1 输入的块缩放 GEMM",
    scenario="仅transpose_B=False",
    constraint="L1入口只支持NT",
    responsibility="编译器：必须拦截",
    inputs="A[32,128], B[128,32] E4M3；scale匹配；M/N/K均合法",
    expected_inputs="L1入口只支持NT；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：only supports .*NT",
    kind="reject",
    stage="lowering",
    diagnostic="only supports .*NT",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）",
)


def make_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((32, 128), "float8_e4m3fn"),
        B: T.Tensor((128, 32), "float8_e4m3fn"),
        SA: T.Tensor((2, 32), "uint16"),
        SB: T.Tensor((2, 32), "uint16"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((32, 128), "float8_e4m3fn")
            b = T.alloc_l1((128, 32), "float8_e4m3fn")
            sa = T.alloc_l1((32, 2), "uint16")
            sb = T.alloc_l1((32, 2), "uint16")
            c = T.alloc_l0c((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            T.gemm_blockscaled(a, b, c, sa, sb, transpose_A=False, transpose_B=False, clear_accum=True)
            T.copy(c, C)

    return main


def make_reference():
    return None


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
