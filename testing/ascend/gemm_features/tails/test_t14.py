"""用例：T14
二级目录：tails
三级特性：显式 L0 的有效 M/N
唯一场景：普通GEMM有效区24×56，显式紧凑L0 pitch
验证约束：完整K64分两次K32搬入L0，l0_pitch匹配有效尺寸；有效区正确，GM外围哨兵不变
责任：编译器：有效尺寸/写回；调用方：紧凑布局和容量
实际输入：A/B[3,64,64]/BF16；A后K32加1；GM输出预填-12345；seed=20261119
预期输入：L0物理64×64；各部分搬运对齐至16且选同一有效pitch；手动同步并关闭自动调度
实际输出契约：C[3,64,64]/FP32，原地更新有效区
预期输出及验收标准：每轮有效区等于CPU完整K64乘积；其余GM哨兵保持；atol=rtol=1e-5
内核来源：testing/ascend/layout/test_ascend_partial_l0_copy.py::test_manual_partial_copies_with_branch_specific_gemm_pitch[ordinary]
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t14.py
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
    id="T14",
    group="tails",
    feature="显式 L0 的有效 M/N",
    scenario="普通GEMM有效区24×56，显式紧凑L0 pitch",
    constraint="完整K64分两次K32搬入L0，l0_pitch匹配有效尺寸；有效区正确，GM外围哨兵不变",
    responsibility="编译器：有效尺寸/写回；调用方：紧凑布局和容量",
    inputs="A/B[3,64,64]/BF16；A后K32加1；GM输出预填-12345；seed=20261119",
    expected_inputs="L0物理64×64；各部分搬运对齐至16且选同一有效pitch；手动同步并关闭自动调度",
    outputs="C[3,64,64]/FP32，原地更新有效区",
    expected="每轮有效区等于CPU完整K64乘积；其余GM哨兵保持；atol=rtol=1e-5",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="asc_mmad\\(",
    atol=1e-05,
    rtol=1e-05,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_ascend_partial_l0_copy.py::test_manual_partial_copies_with_branch_specific_gemm_pitch[ordinary]",
    manual_schedule=True,
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((3, 64, 64), "bfloat16"), B: T.Tensor((3, 64, 64), "bfloat16"), C: T.Tensor((3, 64, 64), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 64), "bfloat16")
            b_l1 = T.alloc_l1((64, 64), "bfloat16")
            a = T.alloc_l0a((64, 64), "bfloat16")
            b = T.alloc_l0b((64, 64), "bfloat16")
            acc = T.alloc_l0c((64, 64), "float32")
            with T.Cube():
                for e in T.serial(3):
                    m = 24
                    n = 56
                    padded_m = T.ceildiv(m, 16) * 16
                    padded_n = T.ceildiv(n, 16) * 16
                    T.copy(A[e, :, :], a_l1)
                    T.copy(B[e, :, :], b_l1)
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    for part in T.serial(2):
                        T.copy(
                            a_l1[:padded_m, part * 32 : (part + 1) * 32],
                            a[:padded_m, part * 32 : (part + 1) * 32],
                            l0_pitch=T.ceildiv(m, 16),
                        )
                        T.copy(
                            b_l1[:padded_n, part * 32 : (part + 1) * 32],
                            b[:padded_n, part * 32 : (part + 1) * 32],
                            l0_pitch=T.ceildiv(n, 16),
                        )
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(a[:24, :], b[:56, :], acc[:24, :56], transpose_B=True, clear_accum=True)
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(acc[:m, :n], C[e, :m, :n])
                    T.ascend_pipe_barrier("PIPE_ALL")

    return main


def make_reference():
    generator = torch.Generator().manual_seed(SEED)
    a = torch.randint(-4, 5, (3, 64, 64), generator=generator).to(torch.bfloat16) / 8
    b = torch.randint(-4, 5, (3, 64, 64), generator=generator).to(torch.bfloat16) / 8
    a[:, :, 32:] += 1
    product = a.float() @ b.float().transpose(-1, -2)
    initial = torch.full((3, 64, 64), -12345.0, dtype=torch.float32)
    expected = initial.clone()
    expected[:, :24, :56] = product[:, :24, :56]
    return [a, b], expected, initial


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference))
