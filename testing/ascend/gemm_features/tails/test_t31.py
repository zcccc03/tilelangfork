"""用例：T31
二级目录：tails
三级特性：MX K 尾块与计算 K 对齐
唯一场景：float8_e4m3fn，transpose-to-k-contiguous，有效K65，显式Task
验证约束：只补硬件GM→L1搬运未清零的K尾部，不重复清零或扩大补零范围
责任：编译器：AscendInsertOOBPadding补零范围与顺序
实际输入：物理K128，MN32，实际K65；transpose-to-k-contiguous；显式Task=True
预期输入：局部IR只用于标识MX的K轴；其他输入在本pass边界预先就绪，不能将此局部IR直接作为已初始化的完整数值内核运行
实际输出契约：经过AscendInsertOOBPadding后的IR
预期输出及验收标准：仅在K轴起点96补零32项，覆盖全部MN；在GM写后/L0读前
内核来源：testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t31.py
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
from testing.ascend._ir import calls, nodes, allocated_buffer
from testing.ascend.gemm_features._support import Case, pytest_case, standalone

SEED = 20261119
CASE = Case(
    id="T31",
    group="tails",
    feature="MX K 尾块与计算 K 对齐",
    scenario="float8_e4m3fn，transpose-to-k-contiguous，有效K65，显式Task",
    constraint="只补硬件GM→L1搬运未清零的K尾部，不重复清零或扩大补零范围",
    responsibility="编译器：AscendInsertOOBPadding补零范围与顺序",
    inputs="物理K128，MN32，实际K65；transpose-to-k-contiguous；显式Task=True",
    expected_inputs="局部IR只用于标识MX的K轴；其他输入在本pass边界预先就绪，不能将此局部IR直接作为已初始化的完整数值内核运行",
    outputs="经过AscendInsertOOBPadding后的IR",
    expected="仅在K轴起点96补零32项，覆盖全部MN；在GM写后/L0读前",
    kind="ir",
    stage="lowering",
    diagnostic="",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail",
    manual_schedule=False,
)


def make_kernel():
    a = tirx.decl_buffer((128, 32), "float8_e4m3fn", name="A")
    l1 = tirx.decl_buffer((32, 128), "float8_e4m3fn", name="l1", scope="shared.l1")
    scale = tirx.decl_buffer((32, 128 // 64), "int16", name="scale", scope="shared.l1")
    l0a = tirx.decl_buffer((32, 128), "float8_e4m3fn", name="l0a", scope="shared.l0a")
    l0b = tirx.decl_buffer((32, 128), "float8_e4m3fn", name="l0b", scope="shared.l0b")
    sfa = tirx.decl_buffer((32, 128 // 64), "int16", name="sfa", scope="shared.l0a.sf")
    sfb = tirx.decl_buffer((32, 128 // 64), "int16", name="sfb", scope="shared.l0b.sf")
    acc = tirx.decl_buffer((32, 32), "float32", name="acc", scope="shared.l0c")

    def region(buffer, transposed):
        shape = (65, 32) if transposed else (32, 65)
        return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(0, extent) for extent in shape])

    body = tirx.SeqStmt(
        [
            tirx.Evaluate(T.copy(region(a, True), region(l1, False), transpose=True)),
            tirx.Evaluate(T.copy(l1, l0a)),
            tirx.Evaluate(T.copy(scale, sfa)),
            tirx.Evaluate(T.gemm_blockscaled(l0a, l0b, acc, sfa, sfb, transpose_A=False, transpose_B=True, clear_accum=True)),
        ]
    )
    body = tirx.SeqStmt([tirx.AttrStmt(tirx.IntImm("int32", 0), "tl.ascend_task", 1, body.seq[0]), *body.seq[1:]])
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


def check_ir(module):
    fills = calls(module, "tl.tileop.fill")
    assert len(fills) == 1
    tasks = [node for node in nodes(module, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    (task,) = tasks
    assert [stmt.value.op.name for stmt in task.body.seq] == ["tl.tileop.ascend_copy", "tl.tileop.fill"]
    (fill,) = fills
    dst, value = fill.args
    assert dst.args[0].buffer.same_as(allocated_buffer(module, "l1"))
    assert float(value) == 0
    assert [int(i) for i in dst.args[0].indices] == [0, 96]
    assert [int(size) for size in dst.args[2:]] == [32, 32]
    copies = calls(module, "tl.tileop.ascend_copy")
    operations = nodes(module, tirx.Call)

    def position(call):
        return next(i for i, candidate in enumerate(operations) if candidate.same_as(call))

    assert position(copies[0]) < position(fill) < position(copies[1])


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options, check_ir=check_ir, lower_ir=lower_ir)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference, check_ir=check_ir, lower_ir=lower_ir))
