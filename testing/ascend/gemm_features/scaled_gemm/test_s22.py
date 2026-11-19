"""用例：S22
二级目录：scaled_gemm
三级特性：显式 L0 块缩放 GEMM 与 scale 操作数校验
唯一场景：仅SF存储类型为float32
验证约束：SF只接受8位单个或16位成对存储
责任：编译器：必须拦截
实际输入：L0A[32,128]/E4M3；alloc_l0a_sf(sf_dtype='float32')
预期输入：SF只接受8位单个或16位成对存储；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：L0 SF storage must contain 8-bit scales or 16-bit scale pairs
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s22.py
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
    id="S22",
    group="scaled_gemm",
    feature="显式 L0 块缩放 GEMM 与 scale 操作数校验",
    scenario="仅SF存储类型为float32",
    constraint="SF只接受8位单个或16位成对存储",
    responsibility="编译器：必须拦截",
    inputs="L0A[32,128]/E4M3；alloc_l0a_sf(sf_dtype='float32')",
    expected_inputs="SF只接受8位单个或16位成对存储；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：L0 SF storage must contain 8-bit scales or 16-bit scale pairs",
    kind="reject",
    stage="frontend",
    diagnostic="L0 SF storage must contain 8-bit scales or 16-bit scale pairs",
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
            a = T.alloc_l0a((32, 128), "float8_e4m3fn")
            T.alloc_l0a_sf(a, sf_dtype="float32")

    return main


def make_reference():
    return None


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
