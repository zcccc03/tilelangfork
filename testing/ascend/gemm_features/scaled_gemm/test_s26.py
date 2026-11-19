"""用例：S26
二级目录：scaled_gemm
三级特性：显式 L0 块缩放 GEMM 与 scale 操作数校验
唯一场景：合法槽[1,2]与紧凑K64 scale切片
验证约束：数据与SF同槽；有效K64恰由一个uint16 scale对覆盖
责任：编译器：绑定及槽偏移
实际输入：物理L0A[2,3,32,128]与SF[2,3,32,2]；选[1,2]；有效A[17,64]，B[32,64]，SF各一列
预期输入：前导槽相同，矩阵行列零起点；SF覆盖有效K64而非整块物理K128
实际输出契约：下沉MAD MX IR
预期输出及验收标准：下沉生成MAD MX K64，A偏移=5*32*128；IR通过即可
内核来源：testing/ascend/transform/test_ascend_lower_gemm.py::_sf_gemm('compact') / test_matching_leading_indices_and_compact_regions
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s26.py
正常结果只输出到终端；失败材料仅在指定 --failure-dir 时保存。
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx
from tilelang.layout import make_ascend_major_k_layout, make_ascend_l0c_layout
from testing.ascend._ir import calls
from testing.ascend.gemm_features._support import Case, pytest_case, standalone

SEED = 20261119
CASE = Case(
    id="S26",
    group="scaled_gemm",
    feature="显式 L0 块缩放 GEMM 与 scale 操作数校验",
    scenario="合法槽[1,2]与紧凑K64 scale切片",
    constraint="数据与SF同槽；有效K64恰由一个uint16 scale对覆盖",
    responsibility="编译器：绑定及槽偏移",
    inputs="物理L0A[2,3,32,128]与SF[2,3,32,2]；选[1,2]；有效A[17,64]，B[32,64]，SF各一列",
    expected_inputs="前导槽相同，矩阵行列零起点；SF覆盖有效K64而非整块物理K128",
    outputs="下沉MAD MX IR",
    expected="下沉生成MAD MX K64，A偏移=5*32*128；IR通过即可",
    kind="ir",
    stage="lowering",
    diagnostic="",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/transform/test_ascend_lower_gemm.py::_sf_gemm('compact') / test_matching_leading_indices_and_compact_regions",
    manual_schedule=False,
)


def kernel(body, *, buffers=(), params=(), annotations=None):
    root = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=buffers, annotations=annotations or {})
    buffer_params = [param for param in params if isinstance(param, tirx.Buffer)]
    func = tirx.PrimFunc(
        [param.data if isinstance(param, tirx.Buffer) else param for param in params],
        tirx.SBlockRealize([], True, root),
        buffer_map={buffer.data: buffer for buffer in buffer_params},
    ).with_attr({"global_symbol": "main", "target": tvm.target.Target("ascend")})
    return tvm.IRModule({"main": func})


def make_kernel():
    a = tirx.decl_buffer((2, 3, 32, 128), "float8_e4m3fn", name="a", scope="shared.l0a")
    a2 = tirx.decl_buffer(a.shape, a.dtype, name="a2", scope="shared.l0a")
    b = tirx.decl_buffer((32, 128), a.dtype, name="b", scope="shared.l0b")
    c = tirx.decl_buffer((32, 32), "float32", name="c", scope="shared.l0c")
    sf = tirx.decl_buffer((2, 3, 32, 2), "uint16", name="sf", scope="shared.l0a.sf")
    sb = tirx.decl_buffer((32, 2), "uint16", name="sb", scope="shared.l0b.sf")
    bindings = {sf.data: a.data, sb.data: b.data}

    def region(buf, mins, extents):
        return tirx.BufferRegion(buf, [tvm.ir.Range.from_min_extent(m, e) for m, e in zip(mins, extents)])

    ar = region(a, [1, 2, 0, 0], [1, 1, 17, 64])
    sr = region(sf, [1, 2, 0, 0], [1, 1, 17, 1])
    br = region(b, [0, 0], [32, 64])
    cr = region(c, [0, 0], [17, 32])
    sbr = region(sb, [0, 0], [32, 1])
    call = T.gemm_blockscaled(ar, br, cr, sr, sbr, transpose_B=True, clear_accum=True)
    return kernel(
        tirx.Evaluate(call),
        buffers=[a, a2, b, c, sf, sb],
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )


def make_reference():
    return None


def check_ir(module):
    (mad,) = calls(module, "tl.ascend_mad_mx")
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(mad.args[4], 64)
    assert analyzer.can_prove_equal(mad.args[1].args[2], 5 * 32 * 128)


def test_case(gemm_options):
    pytest_case(CASE, make_kernel, make_reference, gemm_options, check_ir=check_ir)


if __name__ == "__main__":
    raise SystemExit(standalone(CASE, make_kernel, make_reference, check_ir=check_ir))
