"""Persistent batched/grouped GEMM mapping, precision and accumulation."""

import pytest
import torch
import tilelang
import tilelang.testing
from example_batch_gemm import batch_gemm, ref_program as batch_ref
from example_group_gemm import group_gemm, make_group_offsets, ref_program as group_ref


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("kind", ["batch", "group"])
@pytest.mark.parametrize(
    "dtype,out_dtype,mixed,hf32,acc",
    [
        ("bfloat16", "float32", False, None, False),
        ("float8_e4m3fn", "float32", True, None, False),
        ("float32", "float32", False, "nearest_even", False),
        ("bfloat16", "bfloat16", False, None, False),
        ("bfloat16", "float32", True, None, True),
    ],
)
def test_batch_group_gemm(kind, dtype, out_dtype, mixed, hf32, acc):
    config = dict(
        dtype=dtype,
        out_dtype=out_dtype,
        MIXED=mixed,
        hf32=hf32,
        acc=acc,
        block_m=128,
        block_n=128,
        block_k=128,
        num_blocks=2,
        swizzle_window=2,
    )
    if kind == "batch":
        program = batch_gemm(3, 128, 256, 128, **config)
        a_shape, b_shape, c_shape = (3, 128, 256), (3, 128, 256), (3, 128, 128)
    else:
        # Leading/interior/trailing empty groups and uneven nonempty sizes.
        offsets = make_group_offsets([0, 128, 0, 384, 0], 128)
        program = group_gemm(5, 512, 256, 128, **config)
        a_shape, b_shape, c_shape = (512, 256), (5, 128, 256), (512, 128)
    generator = torch.Generator().manual_seed(20261109)
    a = (torch.randint(-4, 5, a_shape, generator=generator).float() / 8).to(getattr(torch, dtype))
    b = (torch.randint(-4, 5, b_shape, generator=generator).float() / 8).to(getattr(torch, dtype))
    initial = torch.ones(c_shape, dtype=getattr(torch, out_dtype)) if acc else None
    expected = batch_ref(a, b, initial, out_dtype) if kind == "batch" else group_ref(a, b, offsets, initial, out_dtype)
    kernel = tilelang.compile(program, out_idx=None if acc else (-1 if kind == "batch" else 2))
    args = (a.npu(), b.npu())
    extra = () if kind == "batch" else (offsets.npu(),)
    if acc:
        result = initial.npu()
        kernel(*args, result, *extra)
    else:
        result = kernel(*args, *extra)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (batch_gemm, {"B_DIM": 0}),
        (batch_gemm, {"M_DIM": 257}),
        (batch_gemm, {"num_blocks": 0}),
        (group_gemm, {"GROUPS": 0}),
        (group_gemm, {"K_DIM": 129}),
        (group_gemm, {"num_stages": 0}),
    ],
)
def test_multi_gemm_invalid_config(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize("sizes", [[], [0, 0], [-256, 512], [257], [2**31]])
def test_group_offsets_invalid(sizes):
    with pytest.raises(ValueError):
        make_group_offsets(sizes)


def test_group_offsets_empty_groups():
    assert make_group_offsets([0, 256, 0, 512, 0]).tolist() == [0, 0, 256, 256, 768, 768]
