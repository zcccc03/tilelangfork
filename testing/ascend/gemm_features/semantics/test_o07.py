"""用例：O07
二级目录：semantics
三级特性：L1 操作数的 NT GEMM
唯一场景：BF16 L1仅K65无法按C0分段
验证约束：子K必须按16对齐且整除总K
责任：编译器：必须拦截
实际输入：BF16 A/B[32,65]，C[32,32]/FP32
预期输入：子K必须按16对齐且整除总K；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：K.*not divisible.*TILE_K_SUB
内核来源：testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/semantics/test_o07.py
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
    id="O07",
    group="semantics",
    feature="L1 操作数的 NT GEMM",
    scenario="BF16 L1仅K65无法按C0分段",
    constraint="子K必须按16对齐且整除总K",
    responsibility="编译器：必须拦截",
    inputs="BF16 A/B[32,65]，C[32,32]/FP32",
    expected_inputs="子K必须按16对齐且整除总K；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：K.*not divisible.*TILE_K_SUB",
    kind="reject",
    stage="lowering",
    diagnostic="K.*not divisible.*TILE_K_SUB",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/layout/test_tilelang_ascend_l0_gemm_matrix.py::make_l0_gemm_kernel（同类内核，固定本场景）",
)


def make_kernel():
    @T.prim_func
    def main(A: T.Tensor((32, 65), "bfloat16"), B: T.Tensor((32, 65), "bfloat16"), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((32, 65), "bfloat16")
            b = T.alloc_l1((32, 65), "bfloat16")
            c = T.alloc_l0c((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_A=False, transpose_B=True, clear_accum=True)
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
