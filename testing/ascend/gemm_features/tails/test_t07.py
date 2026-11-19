"""用例：T07
二级目录：tails
三级特性：显式 L0 MAD 的运行时有效尺寸
唯一场景：MX运行时m/n与静态物理分配
验证约束：只验证符号有效尺寸保留且分配不缩小
责任：编译器：保留符号；调用方：实际m/n不越界
实际输入：IR L0A/B[64,128]，L0C[64,64]；m/n运行时变量；K=128
预期输入：只验证符号有效尺寸保留且分配不缩小；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：MAD(m,K,n)的下沉IR
预期输出及验收标准：MAD参数保留m/n；A/B物理分配仍64*128；本单元只验IR
内核来源：testing/ascend/transform/test_ascend_lower_gemm.py（独立迁入本文件的IR构造）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/tails/test_t07.py
正常结果只输出到终端；失败材料仅在指定 --failure-dir 时保存。
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import tilelang.ascend.language as T
import math
from tilelang import tvm
from tvm import tirx
from tilelang.layout import make_ascend_major_k_layout, make_ascend_l0c_layout
from testing.ascend._ir import calls, allocated_buffer
from testing.ascend.gemm_features._support import Case, assert_result, execute, exit_code, parse_options

SEED = 20261119
CASE = Case(
    id="T07",
    group="tails",
    feature="显式 L0 MAD 的运行时有效尺寸",
    scenario="MX运行时m/n与静态物理分配",
    constraint="只验证符号有效尺寸保留且分配不缩小",
    responsibility="编译器：保留符号；调用方：实际m/n不越界",
    inputs="IR L0A/B[64,128]，L0C[64,64]；m/n运行时变量；K=128",
    expected_inputs="只验证符号有效尺寸保留且分配不缩小；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="MAD(m,K,n)的下沉IR",
    expected="MAD参数保留m/n；A/B物理分配仍64*128；本单元只验IR",
    kind="ir",
    stage="lowering",
    diagnostic="",
    source_pattern="",
    atol=0,
    rtol=0,
    repeats=1,
    note="",
    kernel_origin="testing/ascend/transform/test_ascend_lower_gemm.py（独立迁入本文件的IR构造）",
)


def region(buffer, mins, extents):
    return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(m, e) for m, e in zip(mins, extents)])


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
    m, n = (tirx.Var("m", "int32"), tirx.Var("n", "int32"))
    a = tirx.decl_buffer((64, 128), "float8_e4m3fn", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((64, 128), "float8_e4m3fn", name="b", scope="shared.l0b")
    c = tirx.decl_buffer((64, 64), "float32", name="c", scope="shared.l0c")
    ar, br, cr = (region(a, [0, 0], [m, 128]), region(b, [0, 0], [n, 128]), region(c, [0, 0], [m, n]))
    buffers, bindings = ([a, b, c], {})
    sa = tirx.decl_buffer((64, 2), "uint16", name="sa", scope="shared.l0a.sf")
    sb = tirx.decl_buffer((64, 2), "uint16", name="sb", scope="shared.l0b.sf")
    buffers += [sa, sb]
    bindings = {sa.data: a.data, sb.data: b.data}
    call = T.gemm_blockscaled(ar, br, cr, region(sa, [0, 0], [m, 2]), region(sb, [0, 0], [n, 2]), transpose_B=True, clear_accum=True)
    return kernel(
        tirx.Evaluate(call),
        buffers=buffers,
        params=[m, n],
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )


def make_reference():
    return None


def check_ir(module):
    (mad,) = calls(module, "tl.ascend_mad_mx")
    m, n = module["main"].params
    analyzer = tvm.arith.Analyzer()
    for actual, expected in zip(mad.args[3:6], [m, 128, n]):
        assert analyzer.can_prove_equal(actual, expected)
    for name in ("a", "b"):
        assert math.prod((int(dim) for dim in allocated_buffer(module, name).shape)) == 64 * 128


def main(config=None):
    config = config or parse_options(CASE)
    return execute(CASE, make_kernel, make_reference, config, check_ir=check_ir)


def test_case(gemm_options):
    assert_result(main(gemm_options))


if __name__ == "__main__":
    raise SystemExit(exit_code(main()))
