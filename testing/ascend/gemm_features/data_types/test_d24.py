"""用例：D24
二级目录：data_types
三级特性：FP32 GEMM 与 HF32 模式
唯一场景：FP32全精度
验证约束：只验证本模式的生成状态及允许误差内的FP32参考结果
责任：编译器：数值实现；调用方：合法输入
实际输入：A(32, 64)/float32，B(32, 64)/float32；seed=20261119；L1/L0路径=l1
预期输入：只验证本模式的生成状态及允许误差内的FP32参考结果；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[32,32]/float32
预期输出及验收标准：CPU按逻辑A/B转换FP32后相乘；atol=0.001, rtol=0.001
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/data_types/test_d24.py
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
    id="D24",
    group="data_types",
    feature="FP32 GEMM 与 HF32 模式",
    scenario="FP32全精度",
    constraint="只验证本模式的生成状态及允许误差内的FP32参考结果",
    responsibility="编译器：数值实现；调用方：合法输入",
    inputs="A(32, 64)/float32，B(32, 64)/float32；seed=20261119；L1/L0路径=l1",
    expected_inputs="只验证本模式的生成状态及允许误差内的FP32参考结果；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[32,32]/float32",
    expected="CPU按逻辑A/B转换FP32后相乘；atol=0.001, rtol=0.001",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="asc_disable_hf32",
    atol=0.001,
    rtol=0.001,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((32, 64), "float32"), B: T.Tensor((32, 64), "float32"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((32, 64), "float32")
            b = T.alloc_l1((32, 64), "float32")
            c = T.alloc_l0c((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.set_hf32_mode(None)
            T.gemm(a, b, c, transpose_A=False, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def make_inputs():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [torch.rand((rows, 64), generator=gen, dtype=torch.float32) * 2 - 1 for rows in (32, 32)]
    inputs = [a, b]
    return (inputs, None)


def ref_program(a, b, initial=None):
    return a[:, 0:64].float() @ b[:, 0:64].float().T


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
