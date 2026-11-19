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
import re
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
import tilelang
import tilelang.ascend.language as T
from testing.ascend.gemm_features._support import (
    Case,
    assert_result,
    execute,
    exit_code,
    parse_options,
    compiled_ir,
    record_case,
    check_signature,
)

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


def make_inputs():
    generator = torch.Generator().manual_seed(SEED)
    a = torch.randint(-4, 5, (3, 64, 64), generator=generator).to(torch.bfloat16) / 8
    b = torch.randint(-4, 5, (3, 64, 64), generator=generator).to(torch.bfloat16) / 8
    a[:, :, 32:] += 1
    initial = torch.full((3, 64, 64), -12345.0, dtype=torch.float32)
    return ([a, b], initial)


def ref_program(a, b, initial=None):
    product = a.float() @ b.float().transpose(-1, -2)
    expected = initial.clone()
    expected[:, :24, :56] = product[:, :24, :56]
    return expected


def make_reference():
    inputs, initial = make_inputs()
    return inputs, ref_program(*inputs, initial=initial), initial


def main(config=None):
    config = config or parse_options(CASE)
    if config.mode != "hardware":
        return execute(CASE, make_kernel, make_reference, config)
    with record_case(CASE, make_kernel, config) as record:
        record.phase = "reference"
        inputs, initial = make_inputs()
        expected = ref_program(*inputs, initial=initial)
        record.evidence["cpu_reference.pt"] = {"inputs": inputs, "expected": expected, "initial_output": initial}
        record.phase = "frontend"
        program = make_kernel()
        record.evidence["input.tir"] = program.script()
        print("[第一部分：program.script() / 输入 IR]\n" + program.script(), flush=True)
        check_signature(program, inputs, expected, initial)
        acc = initial is not None  # Also pass an explicit output when checking untouched GM sentinels.
        record.phase = "compile"
        configs = {tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False} if CASE.manual_schedule else None
        kernel = tilelang.compile(program, target="ascend", out_idx=None if acc else -1, pass_configs=configs)
        source = kernel.get_kernel_source()
        record.evidence["kernel.asc"] = source
        record.evidence["lowered.tir"] = compiled_ir(kernel, program, CASE)
        print("[编译后的 AscendC 源码]\n" + source, flush=True)
        print("[编译后的设备 IR]\n" + record.evidence["lowered.tir"], flush=True)
        assert re.search(CASE.source_pattern, source), "编译源码未满足本场景的目标断言"
        record.phase = "execution"
        try:
            import torch_npu  # noqa: F401
        except (ImportError, OSError) as error:
            record.row.update(status="SKIP", level="HARDWARE", actual="kernel编译通过；NPU运行环境不可用：" + str(error))
            return record.row
        if not torch.npu.is_available():
            record.row.update(status="SKIP", level="HARDWARE", actual="kernel编译通过；NPU不可用，数值尚未验收")
            return record.row
        torch.npu.set_device(config.device)
        device_inputs = [value.npu() for value in inputs]
        max_error = 0
        for launch in range(1, CASE.repeats + 1):
            if acc:
                result = initial.clone().npu()
                kernel(*device_inputs, result)
            else:
                result = kernel(*device_inputs)
            torch.npu.synchronize()
            actual = result.cpu()
            record.evidence[f"output_{launch}.pt"] = {"actual": actual, "expected": expected}
            unchanged = expected == initial
            assert torch.equal(actual[unchanged], initial[unchanged]), "GM外围哨兵必须逐元素保持"
            max_diff = (actual.double() - expected.double()).abs().max().item()
            torch.testing.assert_close(
                actual,
                expected,
                atol=CASE.atol,
                rtol=CASE.rtol,
                msg=lambda message, max_diff=max_diff: f"max_diff={max_diff:.8g}, atol={CASE.atol}, rtol={CASE.rtol}\n{message}",
            )
            max_error = max(max_error, max_diff)
            print(f"{CASE.id} ({CASE.scenario}): correctness passed / 运行结果正确; launch={launch}, max_diff={max_diff:.8g}", flush=True)
        record.row.update(
            status="PASS",
            level="HARDWARE",
            actual="运行结果正确：kernel输出已直接与本文件CPU参考比较",
            max_abs_error=max_error,
            launches=CASE.repeats,
        )
    return record.row


def test_case(gemm_options):
    assert_result(main(gemm_options))


if __name__ == "__main__":
    raise SystemExit(exit_code(main()))
