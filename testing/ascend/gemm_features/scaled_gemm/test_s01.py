"""用例：S01
二级目录：scaled_gemm
三级特性：L1 输入的块缩放 GEMM
唯一场景：四份操作数均在L1，NT正常计算
验证约束：合法L1入口应接受，并正确使用两侧非单位scale
责任：编译器：MX运算；调用方：数据与scale同槽且值正确
实际输入：A(32, 128), B(32, 128)/float8_e4m3fn；SA[2,32],SB[2,32]/uint16；scale字节125..129；seed=20261119
预期输入：合法L1入口应接受，并正确使用两侧非单位scale；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[32,32]/float32
预期输出及验收标准：CPU将本单元实际E4M3输入转FP32，按逻辑E8M0每K32分组缩放后相乘；逐元素误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s01.py
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
    id="S01",
    group="scaled_gemm",
    feature="L1 输入的块缩放 GEMM",
    scenario="四份操作数均在L1，NT正常计算",
    constraint="合法L1入口应接受，并正确使用两侧非单位scale",
    responsibility="编译器：MX运算；调用方：数据与scale同槽且值正确",
    inputs="A(32, 128), B(32, 128)/float8_e4m3fn；SA[2,32],SB[2,32]/uint16；scale字节125..129；seed=20261119",
    expected_inputs="合法L1入口应接受，并正确使用两侧非单位scale；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[32,32]/float32",
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
        A: T.Tensor((32, 128), "float8_e4m3fn"),
        B: T.Tensor((32, 128), "float8_e4m3fn"),
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
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            T.gemm_blockscaled(a[:, :128], b[:, :128], c, sa[:, :2], sb[:, :2], transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 128), "float8_e4m3fn", gen) for rows in (32, 32)]
    aa, bb = (a, b)
    sa, sb = [torch.randint(125, 130, (rows, 128 // 32), generator=gen, dtype=torch.uint8) for rows in (32, 32)]
    af, bf = [v.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)[:, :128] for v, sf in ((a, sa), (b, sb))]
    expected = af @ bf.T
    inputs = [aa, bb, sa.view(torch.uint16).T.contiguous(), sb.view(torch.uint16).T.contiguous()]
    return (inputs, expected, None)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
