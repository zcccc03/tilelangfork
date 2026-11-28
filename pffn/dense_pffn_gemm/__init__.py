"""Optimized Ascend 950DT fused-cast batched GEMM for DensePFFN (_dense_pffn_gemm_kernel)."""

from .kernel import (
    _fc_bmm,
    build_v4_l2_pingpong_bmm,
    build_v5_ring_bmm,
    dense_pffn_gemm,
)
from .reference import CALLSITES, make_callsite_inputs, reference_fc_bmm

__all__ = [
    "CALLSITES",
    "_fc_bmm",
    "build_v4_l2_pingpong_bmm",
    "build_v5_ring_bmm",
    "dense_pffn_gemm",
    "make_callsite_inputs",
    "reference_fc_bmm",
]
