"""用例：T04
二级目录：tails
三级特性：MX K 尾块与计算 K 对齐
唯一场景：逻辑K96自动补零后计算K128
验证约束：主数据尾部补零，scale覆盖128；结果只来自96项
责任：编译器：MX运算；调用方：数据与scale同槽且值正确
实际输入：A(32, 96), B(32, 96)/float8_e4m3fn；SA[2,32],SB[2,32]/uint16；scale字节125..129；seed=20261119
预期输入：主数据尾部补零，scale覆盖128；结果只来自96项；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[32,32]/float32
预期输出及验收标准：CPU只计算逻辑前96项的E4M3/E8M0缩放乘积；源码必须含两侧补零和MAD MX K128；逐元素误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t04.py
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
    id="T04",
    group="tails",
    feature="MX K 尾块与计算 K 对齐",
    scenario="逻辑K96自动补零后计算K128",
    constraint="主数据尾部补零，scale覆盖128；结果只来自96项",
    responsibility="编译器：MX运算；调用方：数据与scale同槽且值正确",
    inputs="A(32, 96), B(32, 96)/float8_e4m3fn；SA[2,32],SB[2,32]/uint16；scale字节125..129；seed=20261119",
    expected_inputs="主数据尾部补零，scale覆盖128；结果只来自96项；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[32,32]/float32",
    expected="CPU只计算逻辑前96项的E4M3/E8M0缩放乘积；源码必须含两侧补零和MAD MX K128；逐元素误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="asc_mmad_mx\\(",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）",
)


def make_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((32, 96), "float8_e4m3fn"),
        B: T.Tensor((32, 96), "float8_e4m3fn"),
        SA: T.Tensor((2, 32), "uint16"),
        SB: T.Tensor((2, 32), "uint16"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((32, 128), "float8_e4m3fn")
            b = T.alloc_l1((32, 128), "float8_e4m3fn")
            sa = T.alloc_l1((32, 2), "uint16")
            sb = T.alloc_l1((32, 2), "uint16")
            c = T.alloc_l0c((32, 32), "float32")
            T.copy(A, a[:, :96])
            T.copy(B, b[:, :96])
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            a0 = T.alloc_l0a((32, 128), "float8_e4m3fn")
            b0 = T.alloc_l0b((32, 128), "float8_e4m3fn")
            sa0 = T.alloc_l0a_sf(a0, sf_dtype="uint16", sf_shape=(32, 2))
            sb0 = T.alloc_l0b_sf(b0, sf_dtype="uint16", sf_shape=(32, 2))
            T.copy(a, a0)
            T.copy(b, b0)
            T.copy(sa, sa0)
            T.copy(sb, sb0)
            T.gemm_blockscaled(a0[:, :128], b0[:, :128], c, sa0[:, :2], sb0[:, :2], transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 96), "float8_e4m3fn", gen) for rows in (32, 32)]
    aa, bb = (a, b)
    sa, sb = [torch.randint(125, 130, (rows, 128 // 32), generator=gen, dtype=torch.uint8) for rows in (32, 32)]
    af, bf = [v.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)[:, :96] for v, sf in ((a, sa), (b, sb))]
    expected = af @ bf.T
    inputs = [aa, bb, sa.view(torch.uint16).T.contiguous(), sb.view(torch.uint16).T.contiguous()]
    return (inputs, expected, None)


def check_source(source):
    import re

    assert source.count("asc_fill_l1(") == 2, "A/B的K补零必须实际生成"
    assert re.search("asc_mmad_mx\\([^\\n]*,\\s*32,\\s*128,\\s*32,", source), "MAD MX必须消费对齐后的K128"


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options, check_source=check_source)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference, check_source=check_source))
