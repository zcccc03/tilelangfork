"""用例：S09
二级目录：scaled_gemm
三级特性：L1 输入的块缩放 GEMM
唯一场景：K384自动选择可装入L0且整除总K的分段
验证约束：只验证自动子K的合法性与完整乘积
责任：编译器：MX运算；调用方：数据与scale同槽且值正确
实际输入：A(128, 384), B(128, 384)/float8_e4m3fn；SA[6,128],SB[6,128]/uint16；scale字节125..129；seed=20261119
预期输入：只验证自动子K的合法性与完整乘积；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[128,128]/float32
预期输出及验收标准：CPU将本单元实际E4M3输入转FP32，按逻辑E8M0每K32分组缩放后相乘；逐元素误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s09.py
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
    id="S09",
    group="scaled_gemm",
    feature="L1 输入的块缩放 GEMM",
    scenario="K384自动选择可装入L0且整除总K的分段",
    constraint="只验证自动子K的合法性与完整乘积",
    responsibility="编译器：MX运算；调用方：数据与scale同槽且值正确",
    inputs="A(128, 384), B(128, 384)/float8_e4m3fn；SA[6,128],SB[6,128]/uint16；scale字节125..129；seed=20261119",
    expected_inputs="只验证自动子K的合法性与完整乘积；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[128,128]/float32",
    expected="CPU将本单元实际E4M3输入转FP32，按逻辑E8M0每K32分组缩放后相乘；逐元素误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="ascend_blockscaled_gemm_l1<",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）",
)


def make_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((128, 384), "float8_e4m3fn"),
        B: T.Tensor((128, 384), "float8_e4m3fn"),
        SA: T.Tensor((6, 128), "uint16"),
        SB: T.Tensor((6, 128), "uint16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((128, 384), "float8_e4m3fn")
            b = T.alloc_l1((128, 384), "float8_e4m3fn")
            sa = T.alloc_l1((128, 6), "uint16")
            sb = T.alloc_l1((128, 6), "uint16")
            c = T.alloc_l0c((128, 128), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            T.gemm_blockscaled(a[:, :384], b[:, :384], c, sa[:, :6], sb[:, :6], transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 384), "float8_e4m3fn", gen) for rows in (128, 128)]
    aa, bb = (a, b)
    sa, sb = [torch.randint(125, 130, (rows, 384 // 32), generator=gen, dtype=torch.uint8) for rows in (128, 128)]
    af, bf = [v.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)[:, :384] for v, sf in ((a, sa), (b, sb))]
    expected = af @ bf.T
    inputs = [aa, bb, sa.view(torch.uint16).T.contiguous(), sb.view(torch.uint16).T.contiguous()]
    return (inputs, expected, None)


def check_source(source):
    import re

    match = re.search("ascend_blockscaled_gemm_l1<128, 384, 128, (\\d+),", source)
    assert match, source
    sub_k = int(match[1])
    assert sub_k % 64 == 0 and 384 % sub_k == 0
    assert 0 < sub_k < 384 and 2 * 128 * sub_k <= 65536


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options, check_source=check_source)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference, check_source=check_source))
