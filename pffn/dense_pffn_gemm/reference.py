"""Reference PyTorch implementation and DensePFFN 6-callsite tensor generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class CallsiteSpec:
    name: str
    batch: int
    M: int
    N: int
    K: int
    a_dtype: str
    w_dtype: str
    o_dtype: str
    a_layout: str
    w_layout: str

    @property
    def flops(self) -> float:
        return 2.0 * self.batch * self.M * self.N * self.K

    def ideal_ms(self, peak_tflops: float = 432.0) -> float:
        return self.flops / (peak_tflops * 1e12) * 1e3


# Default DensePFFN SwiGLU FFN dimensions: B=1024, T=56, D=1232, P=2464, 2P=4928.
CALLSITES = (
    CallsiteSpec("c1_fwd_up", 56, 1024, 4928, 1232, "float32", "float32", "bfloat16", "perm(1,0,2) sak=1", "contig swn=1"),
    CallsiteSpec("c2_fwd_down", 56, 1024, 1232, 2464, "bfloat16", "float32", "bfloat16", "contig sak=1", "contig swn=1"),
    CallsiteSpec("c3_bwd_dw2", 56, 2464, 1232, 1024, "float32", "float32", "float32", "T(1,2) sam=1", "perm(1,0,2) swn=1"),
    CallsiteSpec("c4_bwd_dswi", 56, 1024, 2464, 1232, "float32", "float32", "float32", "perm(1,0,2) sak=1", "T(1,2) swk=1"),
    CallsiteSpec("c5_bwd_dw1", 56, 1232, 4928, 1024, "bfloat16", "float32", "float32", "perm(1,2,0) sam=1", "perm(1,0,2) swn=1"),
    CallsiteSpec("c6_bwd_dx", 56, 1024, 1232, 4928, "float32", "float32", "float32", "perm(1,0,2) sak=1", "T(1,2) swk=1"),
)


def make_callsite_inputs(
    name: str,
    device: str | torch.device = "npu",
    seed: int = 42,
    B: int = 1024,
    T: int = 56,
    D: int = 1232,
    P: int = 2464,
) -> Tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Construct strided (a, w, out_dtype) tensors matching origin_gemm.py call sites."""
    P2 = 2 * P
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def _randn(*shape, dtype):
        return (torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.1).to(device=device, dtype=dtype)

    if name == "c1_fwd_up":
        x = _randn(B, T, D, dtype=torch.float32)
        w1 = _randn(T, D, P2, dtype=torch.float32)
        return x.permute(1, 0, 2), w1, torch.bfloat16
    if name == "c2_fwd_down":
        swiglu = _randn(T, B, P, dtype=torch.bfloat16)
        w2 = _randn(T, P, D, dtype=torch.float32)
        return swiglu, w2, torch.bfloat16
    if name == "c3_bwd_dw2":
        swiglu_f32 = _randn(T, B, P, dtype=torch.float32)
        dout = _randn(B, T, D, dtype=torch.float32)
        return swiglu_f32.transpose(1, 2), dout.permute(1, 0, 2), torch.float32
    if name == "c4_bwd_dswi":
        dout = _randn(B, T, D, dtype=torch.float32)
        w2 = _randn(T, P, D, dtype=torch.float32)
        return dout.permute(1, 0, 2), w2.transpose(1, 2), torch.float32
    if name == "c5_bwd_dw1":
        x_bf = _randn(B, T, D, dtype=torch.bfloat16)
        dbuf = _randn(B, T, P2, dtype=torch.float32)
        return x_bf.permute(1, 2, 0), dbuf.permute(1, 0, 2), torch.float32
    if name == "c6_bwd_dx":
        dbuf = _randn(B, T, P2, dtype=torch.float32)
        w1 = _randn(T, D, P2, dtype=torch.float32)
        return dbuf.permute(1, 0, 2), w1.transpose(1, 2), torch.float32
    raise ValueError(f"Unknown callsite: {name}")


def reference_fc_bmm(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    use_bf16: bool = True,
) -> torch.Tensor:
    """Exact semantic reference for _fc_bmm: optional BF16 cast + FP32 accumulation + out_dtype cast."""
    if use_bf16:
        a_c = a.to(torch.bfloat16).float()
        w_c = w.to(torch.bfloat16).float()
    else:
        a_c = a.float()
        w_c = w.float()
    out = torch.bmm(a_c, w_c)
    return out.to(out_dtype)
