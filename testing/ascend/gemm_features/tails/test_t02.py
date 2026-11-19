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


def make_inputs():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 64), "float8_e4m3fn", gen) for rows in (17, 19)]
    initial = torch.full((21, 24), -12345.0, dtype=torch.float32)
    return ([a, b], initial)


def ref_program(a, b, initial=None):
    expected = initial.clone()
    expected[2:19, 3:22] = a.float() @ b.float().T
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
