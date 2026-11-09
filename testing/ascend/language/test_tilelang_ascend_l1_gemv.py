"""Hardware GEMV needs compact L1 A; ordinary row-padded GEMM must not change."""
import pytest
import tilelang
import tilelang.ascend.language as T


def program(m=1, n=32, k=256, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((m, k), dtype), B: T.Tensor((n, k), dtype), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((m, k), dtype)
            b = T.alloc_l1((n, k), dtype)
            c = T.alloc_l0c((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, transpose_B=True, clear_accum=True)
            T.copy(c, C)
    return main


@pytest.mark.parametrize("k", [256, 1024])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_l1_gemv_compact_input(k, dtype):
    artifact = tilelang.lower(program(k=k, dtype=dtype), target="ascend")
    source = artifact.kernel_source
    assert f"ascend_gemm_l1<1, {k}, 32," in source
    # Only a single 32B row per K group: no ordinary 16-row NZ expansion.
    assert "asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(1), 0)" in source
    assert "__cube__" in source


def test_l1_gemm_retains_row_padded_layout():
    source = tilelang.lower(program(m=16), target="ascend").kernel_source
    assert "ascend_gemm_l1<16, 256, 32," in source
    assert "static_cast<uint16_t>(16)" in source
    assert "static_cast<uint16_t>(1)" not in source


@pytest.mark.parametrize("k,dtype", [(128, "float16"), (256, "float32")])
def test_l1_gemv_rejects_unsupported_geometry(k, dtype):
    with pytest.raises(Exception, match="L1 GEMV"):
        tilelang.lower(program(k=k, dtype=dtype), target="ascend")
