"""用例：S03
二级目录：scaled_gemm
三级特性：L1 输入的块缩放 GEMM
唯一场景：仅M=17不对齐
验证约束：M必须为16倍数；N32、K128及其余条件合法
责任：设备编译器：模板static_assert
实际输入：A[17,128]、B[32,128] E4M3；uint16 SA[2,17]/SB[2,32]
预期输入：M必须为16倍数；N32、K128及其余条件合法；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：M must be a multiple of 16
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s03.py
正常结果只输出到终端；失败材料仅在指定 --failure-dir 时保存。
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import tilelang
import tilelang.ascend.language as T
from testing.ascend.gemm_features._support import Case, assert_result, exit_code, parse_options, compiled_ir, record_case

SEED = 20261119
CASE = Case(
    id="S03",
    group="scaled_gemm",
    feature="L1 输入的块缩放 GEMM",
    scenario="仅M=17不对齐",
    constraint="M必须为16倍数；N32、K128及其余条件合法",
    responsibility="设备编译器：模板static_assert",
    inputs="A[17,128]、B[32,128] E4M3；uint16 SA[2,17]/SB[2,32]",
    expected_inputs="M必须为16倍数；N32、K128及其余条件合法；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：M must be a multiple of 16",
    kind="reject",
    stage="device_compile",
    diagnostic="M must be a multiple of 16",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::_blockscaled_kernel（按本场景固定）",
)


def make_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((17, 128), "float8_e4m3fn"),
        B: T.Tensor((32, 128), "float8_e4m3fn"),
        SA: T.Tensor((2, 17), "uint16"),
        SB: T.Tensor((2, 32), "uint16"),
        C: T.Tensor((17, 32), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((17, 128), "float8_e4m3fn")
            b = T.alloc_l1((32, 128), "float8_e4m3fn")
            sa = T.alloc_l1((17, 2), "uint16")
            sb = T.alloc_l1((32, 2), "uint16")
            c = T.alloc_l0c((17, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            T.gemm_blockscaled(a[:, :128], b[:, :128], c, sa[:, :2], sb[:, :2], transpose_B=True, clear_accum=True)
            T.copy(c, C)

    return main


def make_reference():
    return None


def main(config=None):
    config = config or parse_options(CASE)
    with record_case(CASE, make_kernel, config) as record:
        record.phase = "frontend"
        program = make_kernel()
        record.evidence["input.tir"] = program.script()
        print("[第一部分：program.script() / 输入 IR]\n" + program.script(), flush=True)
        record.phase = "compile"
        configs = {tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False} if CASE.manual_schedule else None
        kernel = tilelang.compile(program, target="ascend", out_idx=None, pass_configs=configs)
        source = kernel.get_kernel_source()
        record.evidence["kernel.asc"] = source
        record.evidence["lowered.tir"] = compiled_ir(kernel, program, CASE)
        print("[编译后的 AscendC 源码]\n" + source, flush=True)
        print("[编译后的设备 IR]\n" + record.evidence["lowered.tir"], flush=True)
        raise AssertionError("目标非法输入被 tilelang.compile 接受；预期约束没有拦截")
    return record.row


def test_case(gemm_options):
    assert_result(main(gemm_options))


if __name__ == "__main__":
    raise SystemExit(exit_code(main()))
