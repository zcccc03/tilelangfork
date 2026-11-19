"""用例：S12
二级目录：scaled_gemm
三级特性：显式 L0 块缩放 GEMM 与 scale 操作数校验
唯一场景：L0 uint8单字节scale正常计算（F1）
验证约束：8位存储必须正确消费相同逻辑scale，不依赖S11运行
责任：编译器：MX运算；调用方：数据与scale同槽且值正确
实际输入：A(32, 128), B(32, 128)/float8_e4m3fn；SA[4,32],SB[4,32]/uint8；scale字节125..129；seed=20261119
预期输入：8位存储必须正确消费相同逻辑scale，不依赖S11运行；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[32,32]/float32
预期输出及验收标准：CPU将本单元实际E4M3输入转FP32，按逻辑E8M0每K32分组缩放后相乘；逐元素误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s12.py
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
    id="S12",
    group="scaled_gemm",
    feature="显式 L0 块缩放 GEMM 与 scale 操作数校验",
    scenario="L0 uint8单字节scale正常计算（F1）",
    constraint="8位存储必须正确消费相同逻辑scale，不依赖S11运行",
    responsibility="编译器：MX运算；调用方：数据与scale同槽且值正确",
    inputs="A(32, 128), B(32, 128)/float8_e4m3fn；SA[4,32],SB[4,32]/uint8；scale字节125..129；seed=20261119",
    expected_inputs="8位存储必须正确消费相同逻辑scale，不依赖S11运行；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[32,32]/float32",
    expected="CPU将本单元实际E4M3输入转FP32，按逻辑E8M0每K32分组缩放后相乘；逐元素误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="asc_mmad_mx\\(",
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
        SA: T.Tensor((4, 32), "uint8"),
        SB: T.Tensor((4, 32), "uint8"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((32, 128), "float8_e4m3fn")
            b = T.alloc_l1((32, 128), "float8_e4m3fn")
            sa = T.alloc_l1((32, 4), "uint8")
            sb = T.alloc_l1((32, 4), "uint8")
            c = T.alloc_l0c((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            a0 = T.alloc_l0a((32, 128), "float8_e4m3fn")
            b0 = T.alloc_l0b((32, 128), "float8_e4m3fn")
            sa0 = T.alloc_l0a_sf(a0, sf_dtype="uint8", sf_shape=(32, 4))
            sb0 = T.alloc_l0b_sf(b0, sf_dtype="uint8", sf_shape=(32, 4))
            T.copy(a, a0)
            T.copy(b, b0)
            T.copy(sa, sa0)
            T.copy(sb, sb0)
            T.gemm_blockscaled(a0[:, :128], b0[:, :128], c, sa0[:, :4], sb0[:, :4], transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def matrix(shape, dtype, generator):
    return (torch.randint(-4, 5, shape, generator=generator).float() / 8).to(getattr(torch, dtype))


def make_inputs():
    gen = torch.Generator().manual_seed(SEED)
    a, b = [matrix((rows, 128), "float8_e4m3fn", gen) for rows in (32, 32)]
    aa, bb = (a, b)
    sa, sb = [torch.randint(125, 130, (rows, 128 // 32), generator=gen, dtype=torch.uint8) for rows in (32, 32)]
    inputs = [aa, bb, sa.view(torch.uint8).T.contiguous(), sb.view(torch.uint8).T.contiguous()]
    return (inputs, None)


def ref_program(a, b, sa, sb, initial=None):
    sa = sa.T.contiguous().view(torch.uint8)
    sb = sb.T.contiguous().view(torch.uint8)
    af = a.float() * torch.exp2(sa.float() - 127).repeat_interleave(32, dim=1)[:, : a.shape[1]]
    bf = b.float() * torch.exp2(sb.float() - 127).repeat_interleave(32, dim=1)[:, : b.shape[1]]
    return af @ bf.T


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
