"""用例：S24
二级目录：scaled_gemm
三级特性：L1 输入的块缩放 GEMM
唯一场景：L1打包FP4的INT8字节视图输入
验证约束：每字节两个E2M1；每K32一个scale；CPU独立nibble查表
责任：编译器：MX运算；调用方：数据与scale同槽且值正确
实际输入：A/B[16,128]/int8，T.view为逻辑[16,256]/float4_e2m1fn；SA/SB[4,16]/uint16；seed=20261119
预期输入：每字节两个E2M1；每K32一个scale；CPU独立nibble查表；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[16,16]/float32
预期输出及验收标准：CPU独立nibble查表解码E2M1，再按逻辑E8M0每K32分组缩放后相乘；逐元素误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel(byte_view=True)
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s24.py
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
    id="S24",
    group="scaled_gemm",
    feature="L1 输入的块缩放 GEMM",
    scenario="L1打包FP4的INT8字节视图输入",
    constraint="每字节两个E2M1；每K32一个scale；CPU独立nibble查表",
    responsibility="编译器：MX运算；调用方：数据与scale同槽且值正确",
    inputs="A/B[16,128]/int8，T.view为逻辑[16,256]/float4_e2m1fn；SA/SB[4,16]/uint16；seed=20261119",
    expected_inputs="每字节两个E2M1；每K32一个scale；CPU独立nibble查表；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[16,16]/float32",
    expected="CPU独立nibble查表解码E2M1，再按逻辑E8M0每K32分组缩放后相乘；逐元素误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="ascend_blockscaled_gemm_l1<",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel(byte_view=True)",
    manual_schedule=False,
)


def make_kernel():
    @T.prim_func
    def gemm(
        a: T.Tensor((16, 256 // 2), "int8"),
        b: T.Tensor((16, 256 // 2), "int8"),
        sa: T.Tensor((4, 16), "uint16"),
        sb: T.Tensor((4, 16), "uint16"),
        out: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 256), "float4_e2m1fn")
            b_l1 = T.alloc_l1((16, 256), "float4_e2m1fn")
            sa_l1 = T.alloc_l1((16, 4), "uint16")
            sb_l1 = T.alloc_l1((16, 4), "uint16")
            acc = T.alloc_l0c((16, 16), "float32")
            a_data = T.view(a, (16, 256), "float4_e2m1fn")
            b_data = T.view(b, (16, 256), "float4_e2m1fn")
            T.copy(a_data, a_l1)
            T.copy(b_data, b_l1)
            T.copy(sa, sa_l1, transpose=True)
            T.copy(sb, sb_l1, transpose=True)
            T.gemm_blockscaled(a_l1, b_l1, acc, sa_l1, sb_l1, transpose_B=True, clear_accum=True)
            T.copy(acc, out)

    return gemm


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
    codes = [torch.randint(0, 16, (rows, 256), generator=gen, dtype=torch.uint8) for rows in (16, 16)]
    a, b = [table[code.long()] for code in codes]
    aa, bb = [(code[:, ::2] | code[:, 1::2] << 4).view(torch.int8) for code in codes]
    sa, sb = [torch.randint(125, 130, (rows, 256 // 32), generator=gen, dtype=torch.uint8) for rows in (16, 16)]
    af, bf = [v.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)[:, :256] for v, sf in ((a, sa), (b, sb))]
    expected = af @ bf.T
    inputs = [aa, bb, sa.view(torch.uint16).T.contiguous(), sb.view(torch.uint16).T.contiguous()]
    return (inputs, expected, None)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
