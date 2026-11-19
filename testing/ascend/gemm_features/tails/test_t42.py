"""用例：T42
二级目录：tails
三级特性：MX K 尾块与计算 K 对齐
唯一场景：物理K96不能完整覆盖64元素scale组
验证约束：物理MX K分配必须为64的倍数
责任：编译器：物理scale组完整性检查
实际输入：E4M3；k-contiguous；实际K96/物理K96；MN32
预期输入：合法物理K应按64对齐；本单元仅改变物理K为96
实际输出契约：不产生数值输出
预期输出及验收标准：AscendInsertOOBPadding拒绝并诊断divisible by 64
内核来源：testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_padding_rejects_an_incomplete_physical_scale_group
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t42.py
正常结果只输出到终端；失败材料仅在指定 --failure-dir 时保存。
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from tilelang.layout import make_ascend_major_k_layout
from testing.ascend.gemm_features._support import Case, pytest_case, standalone

SEED = 20261119
CASE = Case(
    id="T42",
    group="tails",
    feature="MX K 尾块与计算 K 对齐",
    scenario="物理K96不能完整覆盖64元素scale组",
    constraint="物理MX K分配必须为64的倍数",
    responsibility="编译器：物理scale组完整性检查",
    inputs="E4M3；k-contiguous；实际K96/物理K96；MN32",
    expected_inputs="合法物理K应按64对齐；本单元仅改变物理K为96",
    outputs="不产生数值输出",
    expected="AscendInsertOOBPadding拒绝并诊断divisible by 64",
    kind="reject",
    stage="lowering",
    diagnostic="divisible by 64",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_padding_rejects_an_incomplete_physical_scale_group",
    manual_schedule=False,
)


def make_kernel():
    a = tirx.decl_buffer((32, 96), "float8_e4m3fn", name="A")
    l1 = tirx.decl_buffer((32, 96), "float8_e4m3fn", name="l1", scope="shared.l1")
    scale = tirx.decl_buffer((32, 96 // 64), "int16", name="scale", scope="shared.l1")
    l0a = tirx.decl_buffer((32, 96), "float8_e4m3fn", name="l0a", scope="shared.l0a")
    l0b = tirx.decl_buffer((32, 96), "float8_e4m3fn", name="l0b", scope="shared.l0b")
    sfa = tirx.decl_buffer((32, 96 // 64), "int16", name="sfa", scope="shared.l0a.sf")
    sfb = tirx.decl_buffer((32, 96 // 64), "int16", name="sfb", scope="shared.l0b.sf")
    acc = tirx.decl_buffer((32, 32), "float32", name="acc", scope="shared.l0c")

    def region(buffer, transposed):
        shape = (96, 32) if transposed else (32, 96)
        return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(0, extent) for extent in shape])

    body = tirx.SeqStmt(
        [
            tirx.Evaluate(T.copy(region(a, False), region(l1, False), transpose=False)),
            tirx.Evaluate(T.copy(l1, l0a)),
            tirx.Evaluate(T.copy(scale, sfa)),
            tirx.Evaluate(T.gemm_blockscaled(l0a, l0b, acc, sfa, sfb, transpose_A=False, transpose_B=True, clear_accum=True)),
        ]
    )
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=[l1, scale, l0a, l0b, acc, sfa, sfb],
        annotations={"tl.l0_sf_bindings": {sfa.data: l0a.data, sfb.data: l0b.data}, "layout_map": {l1: make_ascend_major_k_layout(l1)}},
    )
    body = tirx.SBlockRealize([], True, root)
    params = [a.data, *tirx.analysis.undefined_vars(body, [a.data])]
    return tvm.IRModule({"main": tirx.PrimFunc(params, body, buffer_map={a.data: a})})


def make_reference():
    return None


def lower_ir(module):
    return transform.AscendInsertOOBPadding()(module)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options, lower_ir=lower_ir)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference, lower_ir=lower_ir))
