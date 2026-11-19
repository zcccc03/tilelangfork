"""用例：T03
二级目录：tails
三级特性：普通 GEMM 的有效 K 尾块
唯一场景：搬入K32，仅消费前K24，尾部非零
验证约束：K24之后A填2/B填3；计算不能包含这8项
责任：编译器：有效K；调用方：两侧有效K一致
实际输入：A[16,32]/B[32,16]/FP32；B转置搬入；seed固定
预期输入：K24之后A填2/B填3；计算不能包含这8项；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[16,16]/FP32
预期输出及验收标准：A[:,:24]@B[:24,:]，误差0
内核来源：testing/ascend/language/test_tilelang_ascend_l0_gemm_effective_k.py::_transposed_b_kernel
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t03.py
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
    id="T03",
    group="tails",
    feature="普通 GEMM 的有效 K 尾块",
    scenario="搬入K32，仅消费前K24，尾部非零",
    constraint="K24之后A填2/B填3；计算不能包含这8项",
    responsibility="编译器：有效K；调用方：两侧有效K一致",
    inputs="A[16,32]/B[32,16]/FP32；B转置搬入；seed固定",
    expected_inputs="K24之后A填2/B填3；计算不能包含这8项；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[16,16]/FP32",
    expected="A[:,:24]@B[:24,:]，误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="asc_mmad\\([^\\n]*, 16, 24, 16,",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/language/test_tilelang_ascend_l0_gemm_effective_k.py::_transposed_b_kernel",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((16, 32), "float32"), B: T.Tensor((32, 16), "float32"), C: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((16, 32), "float32")
            b = T.alloc_l1((32, 16), "float32")
            a0 = T.alloc_l0a((16, 32), "float32")
            b0 = T.alloc_l0b((16, 32), "float32")
            c = T.alloc_l0c((16, 16), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.set_hf32_mode("nearest_even")
            T.copy(a, a0)
            T.copy(b, b0, transpose=True)
            T.gemm(a0[:, :24], b0[:, :24], c, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_reference():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix(shape, "float32", gen) for shape in ((16, 32), (32, 16))]
    a[:, 24:], b[24:, :] = (2, 3)
    return ([a, b], a[:, :24] @ b[:24, :], None)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
