"""用例：D11
二级目录：data_types
三级特性：INT8 输入与 INT32 累加
唯一场景：INT8 l1两片跨片累加
验证约束：第一轮初始化；仅改变第二轮clear设置；CPU保证INT32范围
责任：调用方：选择clear并保证INT32范围；编译器：整数累加语义
实际输入：A(16, 128)/int8，B(32, 128)/int8；seed=20261119；L1/L0路径=l1
预期输入：第一轮初始化；仅改变第二轮clear设置；CPU保证INT32范围；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[16,32]/int32
预期输出及验收标准：CPU按逻辑A/B以INT64计算完整K的乘积；确认参考值在INT32范围内后转成INT32；逐元素误差0
内核来源：testing/ascend/layout/test_ascend_l0_sliced_k.py（按本场景固定分片及类型）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/data_types/test_d11.py
正常结果只输出到终端；失败材料仅在指定 --failure-dir 时保存。
"""

import sys
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
    id="D11",
    group="data_types",
    feature="INT8 输入与 INT32 累加",
    scenario="INT8 l1两片跨片累加",
    constraint="第一轮初始化；仅改变第二轮clear设置；CPU保证INT32范围",
    responsibility="调用方：选择clear并保证INT32范围；编译器：整数累加语义",
    inputs="A(16, 128)/int8，B(32, 128)/int8；seed=20261119；L1/L0路径=l1",
    expected_inputs="第一轮初始化；仅改变第二轮clear设置；CPU保证INT32范围；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[16,32]/int32",
    expected="CPU按逻辑A/B以INT64计算完整K的乘积；确认参考值在INT32范围内后转成INT32；逐元素误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=2,
    note="",
    kernel_origin="testing/ascend/layout/test_ascend_l0_sliced_k.py（按本场景固定分片及类型）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((16, 128), "int8"), B: T.Tensor((32, 128), "int8"), C: T.Tensor((16, 32), "int32")):
        with T.Kernel(1):
            a = T.alloc_l1((16, 64), "int8")
            b = T.alloc_l1((32, 64), "int8")
            c = T.alloc_l0c((16, 32), "int32")
            for step in T.serial(2):
                T.copy(A[:, step * 64 : (step + 1) * 64], a)
                T.copy(B[:, step * 64 : (step + 1) * 64], b)
                T.gemm(a, b, c, transpose_B=True, clear_accum=step == 0)
            T.copy(c, C)

    return main


def make_inputs():
    gen = torch.Generator().manual_seed(SEED)
    a = torch.randint(-128, 128, (16, 128), dtype=torch.int8, generator=gen)
    b = torch.randint(-128, 128, (32, 128), dtype=torch.int8, generator=gen)
    a[0, :] = -128
    b[0, :], b[1, :] = (-128, 127)
    inputs = [a, b]
    return (inputs, None)


def ref_program(a, b, initial=None):
    wide = a[:, 0:128].long() @ b[:, 0:128].long().T
    assert wide.min() >= torch.iinfo(torch.int32).min and wide.max() <= torch.iinfo(torch.int32).max, "caller INT32 range obligation"
    return wide.int()


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
