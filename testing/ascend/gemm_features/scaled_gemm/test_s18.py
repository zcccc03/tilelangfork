"""用例：S18
二级目录：scaled_gemm
三级特性：显式 L0 块缩放 GEMM 与 scale 操作数校验
唯一场景：仅SF行切片起点非零
验证约束：仅SF行切片起点非零必须拒绝，其他IR前提合法
责任：编译器：必须拦截
实际输入：L0A[2,3,32,128]/E4M3，选槽[1,2]有效17行；L0B[32,128]；uint16 SF[2,3,32,2]；仅改变本单元指定条件
预期输入：仅SF行切片起点非零必须拒绝，其他IR前提合法；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。
实际输出契约：不产生数值输出
预期输出及验收标准：拒绝目标非法条件，诊断须匹配：zero-origin
内核来源：testing/ascend/transform/test_ascend_lower_gemm.py（独立迁入本文件的IR构造）
完整内核定义及本场景CPU参考或IR/诊断断言均在本文件，不调用其他测试的kernel工厂。

单独执行：python testing/ascend/gemm_features/scaled_gemm/test_s18.py
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
from testing.ascend.gemm_features._support import Case, assert_result, execute, exit_code, parse_options

SEED = 20261119
CASE = Case(
    id="S18",
    group="scaled_gemm",
    feature="显式 L0 块缩放 GEMM 与 scale 操作数校验",
    scenario="仅SF行切片起点非零",
    constraint="仅SF行切片起点非零必须拒绝，其他IR前提合法",
    responsibility="编译器：必须拦截",
    inputs="L0A[2,3,32,128]/E4M3，选槽[1,2]有效17行；L0B[32,128]；uint16 SF[2,3,32,2]；仅改变本单元指定条件",
    expected_inputs="仅SF行切片起点非零必须拒绝，其他IR前提合法；其余输入遵守上一项的形状/类型，布局、绑定及初始化均保持合法。",
    outputs="不产生数值输出",
    expected="拒绝目标非法条件，诊断须匹配：zero-origin",
    kind="reject",
    stage="lowering",
    diagnostic="zero-origin",
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
    a = tirx.decl_buffer((2, 3, 32, 128), "float8_e4m3fn", name="a", scope="shared.l0a")
    other = tirx.decl_buffer(a.shape, a.dtype, name="other", scope="shared.l0a")
    b = tirx.decl_buffer((32, 128), a.dtype, name="b", scope="shared.l0b")
    c = tirx.decl_buffer((32, 32), "float32", name="c", scope="shared.l0c")
    sf = tirx.decl_buffer((2, 3, 32, 2), "uint16", name="sf", scope="shared.l0a.sf")
    sb = tirx.decl_buffer((32, 2), "uint16", name="sb", scope="shared.l0b.sf")
    bindings = {sf.data: a.data, sb.data: b.data}
    ar = region(a, [1, 2, 0, 0], [1, 1, 17, 128])
    sr = region(sf, [1, 2, 1, 0], [1, 1, 17, 2])
    call = T.gemm_blockscaled(
        ar, region(b, [0, 0], [32, 128]), region(c, [0, 0], [17, 32]), sr, region(sb, [0, 0], [32, 2]), transpose_B=True, clear_accum=True
    )
    return kernel(
        tirx.Evaluate(call),
        buffers=[a, other, b, c, sf, sb],
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )


def make_reference():
    return None


def main(config=None):
    config = config or parse_options(CASE)
    return execute(CASE, make_kernel, make_reference, config)


def test_case(gemm_options):
    assert_result(main(gemm_options))


if __name__ == "__main__":
    raise SystemExit(exit_code(main()))
