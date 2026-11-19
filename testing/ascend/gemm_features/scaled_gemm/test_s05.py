"""用例：S05
二级目录：scaled_gemm
三级特性：L1 输入的块缩放 GEMM
唯一场景：仅N=17不对齐
验证约束：N必须为16倍数；M32、K128及其余条件合法
责任：设备编译器：模板static_assert
实际输入：A[32,128]、B[17,128] E4M3；uint16 SA[2,32]/SB[2,17]
预期输入：N必须为16倍数；M32、K128及其余条件合法；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：N must be a multiple of 16
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s05.py
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
    id="S05",
    group="scaled_gemm",
    feature="L1 输入的块缩放 GEMM",
    scenario="仅N=17不对齐",
    constraint="N必须为16倍数；M32、K128及其余条件合法",
    responsibility="设备编译器：模板static_assert",
    inputs="A[32,128]、B[17,128] E4M3；uint16 SA[2,32]/SB[2,17]",
    expected_inputs="N必须为16倍数；M32、K128及其余条件合法；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：N must be a multiple of 16",
    kind="reject",
    stage="device_compile",
    diagnostic="N must be a multiple of 16",
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
        B: T.Tensor((17, 128), "float8_e4m3fn"),
        SA: T.Tensor((2, 32), "uint16"),
        SB: T.Tensor((2, 17), "uint16"),
        C: T.Tensor((32, 17), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((32, 128), "float8_e4m3fn")
            b = T.alloc_l1((17, 128), "float8_e4m3fn")
            sa = T.alloc_l1((32, 2), "uint16")
            sb = T.alloc_l1((17, 2), "uint16")
            c = T.alloc_l0c((32, 17), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            T.gemm_blockscaled(a[:, :128], b[:, :128], c, sa[:, :2], sb[:, :2], transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def make_reference():
    return None


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
