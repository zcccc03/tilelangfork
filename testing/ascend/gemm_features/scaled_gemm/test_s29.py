"""用例：S29
二级目录：scaled_gemm
三级特性：显式 L0 块缩放 GEMM 与 scale 操作数校验
唯一场景：显式L0 E4M3原有边界形状M100/N100
验证约束：默认16位scale存储保持每K32一个系数
责任：编译器：MX运算；调用方：数据与scale同槽且值正确
实际输入：A[100,128],B[100,128]/E4M3；SA[2,100],SB[2,100]/uint16；seed=20261119
预期输入：默认16位scale存储保持每K32一个系数；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[100,100]/FP32
预期输出及验收标准：CPU将本单元实际E4M3输入转FP32，按逻辑E8M0每K32分组缩放后相乘；逐元素误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::test_blockscaled_l0_gemm / _blockscaled_kernel
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s29.py
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
    id="S29",
    group="scaled_gemm",
    feature="显式 L0 块缩放 GEMM 与 scale 操作数校验",
    scenario="显式L0 E4M3原有边界形状M100/N100",
    constraint="默认16位scale存储保持每K32一个系数",
    responsibility="编译器：MX运算；调用方：数据与scale同槽且值正确",
    inputs="A[100,128],B[100,128]/E4M3；SA[2,100],SB[2,100]/uint16；seed=20261119",
    expected_inputs="默认16位scale存储保持每K32一个系数；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[100,100]/FP32",
    expected="CPU将本单元实际E4M3输入转FP32，按逻辑E8M0每K32分组缩放后相乘；逐元素误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="asc_mmad_mx\\(",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::test_blockscaled_l0_gemm / _blockscaled_kernel",
    manual_schedule=False,
)


def make_kernel():
    @T.prim_func
    def gemm(
        a: T.Tensor((100, 128), "float8_e4m3fn"),
        b: T.Tensor((100, 128), "float8_e4m3fn"),
        sa: T.Tensor((2, 100), "uint16"),
        sb: T.Tensor((2, 100), "uint16"),
        out: T.Tensor((100, 100), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((100, 128), "float8_e4m3fn")
            b_l1 = T.alloc_l1((100, 128), "float8_e4m3fn")
            sa_l1 = T.alloc_l1((100, 2), "uint16")
            sb_l1 = T.alloc_l1((100, 2), "uint16")
            a_l0 = T.alloc_l0a((100, 128), "float8_e4m3fn")
            b_l0 = T.alloc_l0b((100, 128), "float8_e4m3fn")
            a_sf = T.alloc_l0a_sf(a_l0, sf_shape=(100, 2))
            b_sf = T.alloc_l0b_sf(b_l0, sf_shape=(100, 2))
            acc = T.alloc_l0c((100, 100), "float32")
            a_data = a
            b_data = b
            T.copy(a_data, a_l1)
            T.copy(b_data, b_l1)
            T.copy(sa, sa_l1, transpose=True)
            T.copy(sb, sb_l1, transpose=True)
            T.copy(a_l1, a_l0)
            T.copy(sa_l1, a_sf)
            T.copy(b_l1, b_l0)
            T.copy(sb_l1, b_sf)
            T.gemm_blockscaled(a_l0, b_l0, acc, a_sf, b_sf, transpose_A=False, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    return gemm


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 128), "float8_e4m3fn", gen) for rows in (100, 100)]
    aa, bb = (a, b)
    sa, sb = [torch.randint(125, 130, (rows, 128 // 32), generator=gen, dtype=torch.uint8) for rows in (100, 100)]
    af, bf = [v.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)[:, :128] for v, sf in ((a, sa), (b, sb))]
    expected = af @ bf.T
    inputs = [aa, bb, sa.view(torch.uint16).T.contiguous(), sb.view(torch.uint16).T.contiguous()]
    return (inputs, expected, None)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
