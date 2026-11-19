"""用例：T02
二级目录：tails
三级特性：显式 L0 的有效 M/N
唯一场景：float8_e4m3fn有效17×19写入GM切片
验证约束：物理L0为64×64；有效区乘积正确且GM外围181个哨兵不变
责任：编译器：有效尺寸/写回；调用方：紧凑布局和容量
实际输入：A[17,64]/B[19,64]/float8_e4m3fn；C[21,24]/FP32预填-12345
预期输入：物理L0为64×64；有效区乘积正确且GM外围181个哨兵不变；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：原地更新C[21,24]/FP32
预期输出及验收标准：C[2:19,3:22]=CPU乘积；其余=-12345；误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t02.py
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
    id="T02",
    group="tails",
    feature="显式 L0 的有效 M/N",
    scenario="float8_e4m3fn有效17×19写入GM切片",
    constraint="物理L0为64×64；有效区乘积正确且GM外围181个哨兵不变",
    responsibility="编译器：有效尺寸/写回；调用方：紧凑布局和容量",
    inputs="A[17,64]/B[19,64]/float8_e4m3fn；C[21,24]/FP32预填-12345",
    expected_inputs="物理L0为64×64；有效区乘积正确且GM外围181个哨兵不变；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="原地更新C[21,24]/FP32",
    expected="C[2:19,3:22]=CPU乘积；其余=-12345；误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="asc_mmad\\([^\\n]*, 17, 64, 19,",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((17, 64), "float8_e4m3fn"), B: T.Tensor((19, 64), "float8_e4m3fn"), C: T.Tensor((21, 24), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((17, 64), "float8_e4m3fn")
            b = T.alloc_l1((19, 64), "float8_e4m3fn")
            a0 = T.alloc_l0a((64, 64), "float8_e4m3fn")
            b0 = T.alloc_l0b((64, 64), "float8_e4m3fn")
            c = T.alloc_l0c((64, 64), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(a, a0[:17, :])
            T.copy(b, b0[:19, :])
            T.gemm(a0[:17, :], b0[:19, :], c[:17, :19], transpose_B=True, clear_accum=True)
            T.copy(c[:17, :19], C[2:19, 3:22])

    return main


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 64), "float8_e4m3fn", gen) for rows in (17, 19)]
    initial = torch.full((21, 24), -12345.0, dtype=torch.float32)
    expected = initial.clone()
    expected[2:19, 3:22] = a.float() @ b.float().T
    return ([a, b], expected, initial)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
