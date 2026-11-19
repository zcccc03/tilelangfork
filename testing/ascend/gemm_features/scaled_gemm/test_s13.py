"""用例：S13
二级目录：scaled_gemm
三级特性：显式 L0 块缩放 GEMM 与 scale 操作数校验
唯一场景：L0原生打包FP4 E2M1输入
验证约束：显式L0数据与scale绑定且正常完成MX计算
责任：编译器：MX运算；调用方：数据与scale同槽且值正确
实际输入：A/B逻辑[16,256]/float4_e2m1fn；Torch物理[16,128]/float4_e2m1fn_x2；SA/SB[4,16]/uint16；seed=20261119
预期输入：显式L0数据与scale绑定且正常完成MX计算；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：C[16,16]/float32
预期输出及验收标准：CPU独立nibble查表解码E2M1，再按逻辑E8M0每K32分组缩放后相乘；逐元素误差0
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel / test_packed_fp4_blockscaled_gemm
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s13.py
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
    id="S13",
    group="scaled_gemm",
    feature="显式 L0 块缩放 GEMM 与 scale 操作数校验",
    constraint="显式L0数据与scale绑定且正常完成MX计算",
    responsibility="编译器：MX运算；调用方：数据与scale同槽且值正确",
    expected_inputs="显式L0数据与scale绑定且正常完成MX计算；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="C[16,16]/float32",
    expected="CPU独立nibble查表解码E2M1，再按逻辑E8M0每K32分组缩放后相乘；逐元素误差0",
    kind="numerical",
    stage="device_compile",
    diagnostic="",
    source_pattern="asc_mmad_mx\\(",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel / test_packed_fp4_blockscaled_gemm",
    scenario="L0原生打包FP4 E2M1输入",
    inputs="A/B逻辑[16,256]/float4_e2m1fn；Torch物理[16,128]/float4_e2m1fn_x2；SA/SB[4,16]/uint16；seed=20261119",
)


def make_kernel():
    @T.prim_func
    def gemm(
        a: T.Tensor((16, 256), "float4_e2m1fn"),
        b: T.Tensor((16, 256), "float4_e2m1fn"),
        sa: T.Tensor((4, 16), "uint16"),
        sb: T.Tensor((4, 16), "uint16"),
        out: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 256), "float4_e2m1fn")
            b_l1 = T.alloc_l1((16, 256), "float4_e2m1fn")
            sa_l1 = T.alloc_l1((16, 4), "uint16")
            sb_l1 = T.alloc_l1((16, 4), "uint16")
            a_l0 = T.alloc_l0a((16, 256), "float4_e2m1fn")
            b_l0 = T.alloc_l0b((16, 256), "float4_e2m1fn")
            a_sf = T.alloc_l0a_sf(a_l0, sf_shape=(16, 4))
            b_sf = T.alloc_l0b_sf(b_l0, sf_shape=(16, 4))
            acc = T.alloc_l0c((16, 16), "float32")
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


def make_inputs():
    gen = torch.Generator().manual_seed(SEED)
    codes = [torch.randint(0, 16, (rows, 256), generator=gen, dtype=torch.uint8) for rows in (16, 16)]
    aa, bb = [(code[:, ::2] | code[:, 1::2] << 4).view(torch.float4_e2m1fn_x2) for code in codes]
    sa, sb = [torch.randint(125, 130, (rows, 256 // 32), generator=gen, dtype=torch.uint8) for rows in (16, 16)]
    inputs = [aa, bb, sa.view(torch.uint16).T.contiguous(), sb.view(torch.uint16).T.contiguous()]
    return (inputs, None)


def ref_program(a, b, sa, sb, initial=None):
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)

    def decode(value):
        packed = value.contiguous().view(torch.uint8)
        codes = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8)
        codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
        return table[codes.long()]

    a, b = decode(a), decode(b)
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
