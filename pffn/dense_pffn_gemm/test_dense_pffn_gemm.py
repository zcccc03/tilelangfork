#!/usr/bin/env python3
"""Accuracy and performance test suite for dense_pffn_gemm on Ascend 950DT (or CPU fallback).

Covers:
1. All 6 DensePFFN call sites (`c1_fwd_up` .. `c6_bwd_dx`) with NaN-sentinel outputs.
2. Edge cases (small batch, dynamic shapes, non-16-aligned fallback, use_bf16=False).
3. End-to-end DensePFFN SwiGLU forward + 9 backward gradients vs CPU FP32 golden.
4. Optional Cold-L2 (1 GiB flush) and Warm-L2 MFU benchmark (`--bench`).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from dense_pffn_gemm.kernel import dense_pffn_gemm
from dense_pffn_gemm.reference import CALLSITES, make_callsite_inputs, reference_fc_bmm

PEAK_BF16_TFLOPS_1650 = 432.0


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.float().cpu()
    b = expected.float().cpu()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def test_callsites(device: torch.device, backend: str, tol: float = 1e-2, small: bool = False) -> None:
    print(f"\n[1/3] Testing all 6 DensePFFN call sites (device={device}, backend={backend}, small={small})")
    shape_kw = dict(B=64, T=4, D=128, P=256) if small else {}
    for spec in CALLSITES:
        a, w, out_dtype = make_callsite_inputs(spec.name, device=device, **shape_kw)
        b, m, k = a.shape
        n = w.shape[2]
        out = torch.full((b, m, n), float("nan"), device=device, dtype=out_dtype)
        dense_pffn_gemm(a, w, out_dtype=out_dtype, use_bf16=True, backend=backend, out=out)
        _sync(device)
        assert not torch.isnan(out).any().item(), f"{spec.name}: NaN detected (sentinel overwritten?)"
        assert not torch.isinf(out).any().item(), f"{spec.name}: Inf detected"
        ref = reference_fc_bmm(a, w, out_dtype=out_dtype, use_bf16=True)
        rel = _rel_l2(out, ref)
        status = "PASS" if rel <= tol else "FAIL"
        print(f"  {status} {spec.name:12s} ({b}x{m}x{n}x{k}) rel_l2={rel:.3e}")
        assert rel <= tol, f"{spec.name}: rel_l2={rel} > {tol}"


def test_edge_cases(device: torch.device, backend: str, tol: float = 1e-2) -> None:
    print(f"\n[2/3] Testing dynamic & edge cases (device={device}, backend={backend})")
    cases = [
        ("batch1_aligned", 1, 256, 256, 128, torch.float32, torch.float32, torch.bfloat16, True),
        ("batch3_tail_k", 3, 272, 304, 144, torch.float32, torch.bfloat16, torch.float32, True),
        ("non16_fallback", 2, 37, 53, 29, torch.float32, torch.float32, torch.float32, True),
        ("fp32_tf32_mode", 2, 64, 64, 64, torch.float32, torch.float32, torch.float32, False),
    ]
    torch.manual_seed(7)
    for name, b, m, n, k, adt, wdt, odt, use_bf16 in cases:
        a = (torch.randn(b, m, k, dtype=torch.float32) * 0.1).to(device=device, dtype=adt)
        w = (torch.randn(b, k, n, dtype=torch.float32) * 0.1).to(device=device, dtype=wdt)
        out = torch.full((b, m, n), float("nan"), device=device, dtype=odt)
        dense_pffn_gemm(a, w, out_dtype=odt, use_bf16=use_bf16, backend=backend, out=out)
        _sync(device)
        assert torch.isfinite(out).all().item(), f"{name}: non-finite output"
        ref = reference_fc_bmm(a, w, out_dtype=odt, use_bf16=use_bf16)
        rel = _rel_l2(out, ref)
        print(f"  PASS {name:16s} ({b}x{m}x{n}x{k}) rel_l2={rel:.3e}")
        assert rel <= tol, f"{name}: rel_l2={rel} > {tol}"


def test_e2e_dense_pffn(
    device: torch.device,
    backend: str,
    B: int = 128,
    T: int = 8,
    D: int = 256,
    P: int = 512,
    eps: float = 1e-5,
    tol: float = 2e-2,
) -> None:
    """End-to-end DensePFFN SwiGLU forward + 9 backward grads vs CPU FP32 autograd golden."""
    print(f"\n[3/3] Testing E2E DensePFFN fwd+bwd (B={B}, T={T}, D={D}, P={P}, device={device}, backend={backend})")
    torch.manual_seed(123)
    x_cpu = (torch.randn(B, T, D) * 0.1).requires_grad_(True)
    w1_cpu = (torch.randn(T, D, 2 * P) * (1.0 / math.sqrt(D))).requires_grad_(True)
    b1_cpu = (torch.randn(T, 2 * P) * 0.05).requires_grad_(True)
    gx_cpu = (torch.ones(P) + torch.randn(P) * 0.05).requires_grad_(True)
    bx_cpu = (torch.randn(P) * 0.05).requires_grad_(True)
    gg_cpu = (torch.ones(P) + torch.randn(P) * 0.05).requires_grad_(True)
    bg_cpu = (torch.randn(P) * 0.05).requires_grad_(True)
    w2_cpu = (torch.randn(T, P, D) * (1.0 / math.sqrt(P))).requires_grad_(True)
    b2_cpu = (torch.randn(T, D) * 0.05).requires_grad_(True)
    dout_cpu = torch.randn(B, T, D) * 0.1

    # CPU FP32 golden
    buf_g = torch.bmm(x_cpu.permute(1, 0, 2), w1_cpu) + b1_cpu.unsqueeze(1)
    xh_g, gh_g = buf_g.chunk(2, dim=-1)
    nx_g = F.layer_norm(xh_g, (P,), gx_cpu, bx_cpu, eps)
    ng_g = F.layer_norm(gh_g, (P,), gg_cpu, bg_cpu, eps)
    swi_g = nx_g * F.silu(ng_g)
    out_g = torch.bmm(swi_g, w2_cpu).permute(1, 0, 2) + b2_cpu.unsqueeze(0)
    out_g.backward(dout_cpu)

    # Device run using dense_pffn_gemm at all 6 call sites
    x = x_cpu.detach().to(device)
    w1 = w1_cpu.detach().to(device)
    b1 = b1_cpu.detach().to(device)
    gx = gx_cpu.detach().to(device)
    bx = bx_cpu.detach().to(device)
    gg = gg_cpu.detach().to(device)
    bg = bg_cpu.detach().to(device)
    w2 = w2_cpu.detach().to(device)
    b2 = b2_cpu.detach().to(device)
    dout = dout_cpu.to(device)

    # Forward c1 & c2
    buf_bf = dense_pffn_gemm(x.permute(1, 0, 2), w1, out_dtype=torch.bfloat16, use_bf16=True, backend=backend)
    buf_f = buf_bf.float() + b1.unsqueeze(1)
    xh, gh = buf_f.chunk(2, dim=-1)
    xh_req = xh.detach().requires_grad_(True)
    gh_req = gh.detach().requires_grad_(True)
    gx_req = gx.detach().requires_grad_(True)
    bx_req = bx.detach().requires_grad_(True)
    gg_req = gg.detach().requires_grad_(True)
    bg_req = bg.detach().requires_grad_(True)
    nx = F.layer_norm(xh_req, (P,), gx_req, bx_req, eps)
    ng = F.layer_norm(gh_req, (P,), gg_req, bg_req, eps)
    swi = nx * F.silu(ng)
    raw2_bf = dense_pffn_gemm(swi.detach().to(torch.bfloat16), w2, out_dtype=torch.bfloat16, use_bf16=True, backend=backend)
    out_dev = raw2_bf.float().permute(1, 0, 2) + b2.unsqueeze(0)

    # Backward c3, c4, epilogue grads, c5, c6
    db2 = dout.sum(dim=0)
    dw2 = dense_pffn_gemm(swi.detach().transpose(1, 2), dout.permute(1, 0, 2), out_dtype=torch.float32, use_bf16=True, backend=backend)
    dswi = dense_pffn_gemm(dout.permute(1, 0, 2), w2.transpose(1, 2), out_dtype=torch.float32, use_bf16=True, backend=backend)
    swi.backward(dswi)
    dbuf = torch.cat([xh_req.grad, gh_req.grad], dim=-1)  # [T, B, 2P]
    db1 = dbuf.sum(dim=1)
    dw1 = dense_pffn_gemm(x.to(torch.bfloat16).permute(1, 2, 0), dbuf, out_dtype=torch.float32, use_bf16=True, backend=backend)
    dx_t = dense_pffn_gemm(dbuf, w1.transpose(1, 2), out_dtype=torch.float32, use_bf16=True, backend=backend)
    dx = dx_t.permute(1, 0, 2)
    _sync(device)

    checks = [
        ("fwd_out", out_dev, out_g),
        ("dx", dx, x_cpu.grad),
        ("dw1", dw1, w1_cpu.grad),
        ("db1", db1, b1_cpu.grad),
        ("dgx", gx_req.grad, gx_cpu.grad),
        ("dbx", bx_req.grad, bx_cpu.grad),
        ("dgg", gg_req.grad, gg_cpu.grad),
        ("dbg", bg_req.grad, bg_cpu.grad),
        ("dw2", dw2, w2_cpu.grad),
        ("db2", db2, b2_cpu.grad),
    ]
    for name, got, exp in checks:
        rel = _rel_l2(got, exp)
        print(f"  PASS {name:8s} rel_l2={rel:.3e}")
        assert rel <= tol, f"E2E {name}: rel_l2={rel} > {tol}"


def bench_callsites(device: torch.device, backend: str, iters: int = 15) -> None:
    assert device.type == "npu", "--bench requires NPU"
    flush_buf = torch.zeros((256 * 1024 * 1024,), device=device, dtype=torch.int32)
    print(f"\n[BENCH] Cold-L2 (1 GiB flush) & Warm-L2 across all 6 call sites (backend={backend})")
    tot_ideal = 0.0
    tot_cold = 0.0
    tot_warm = 0.0
    for spec in CALLSITES:
        a, w, out_dtype = make_callsite_inputs(spec.name, device=device)
        out = torch.empty((spec.batch, spec.M, spec.N), device=device, dtype=out_dtype)
        for _ in range(5):
            dense_pffn_gemm(a, w, out_dtype=out_dtype, use_bf16=True, backend=backend, out=out)
        torch.npu.synchronize()

        cold_ms_list, warm_ms_list = [], []
        for _ in range(iters):
            flush_buf.zero_()
            torch.npu.synchronize()
            s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            s.record()
            dense_pffn_gemm(a, w, out_dtype=out_dtype, use_bf16=True, backend=backend, out=out)
            e.record()
            torch.npu.synchronize()
            cold_ms_list.append(s.elapsed_time(e))
        for _ in range(iters):
            s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            s.record()
            dense_pffn_gemm(a, w, out_dtype=out_dtype, use_bf16=True, backend=backend, out=out)
            e.record()
            torch.npu.synchronize()
            warm_ms_list.append(s.elapsed_time(e))

        c_ms = statistics.median(cold_ms_list)
        w_ms = statistics.median(warm_ms_list)
        ideal = spec.ideal_ms(PEAK_BF16_TFLOPS_1650)
        tot_ideal += ideal
        tot_cold += c_ms
        tot_warm += w_ms
        print(
            f"  {spec.name:12s} ideal={ideal:.3f}ms | "
            f"cold={c_ms:.3f}ms ({ideal/c_ms*100:5.1f}% MFU) | "
            f"warm={w_ms:.3f}ms ({ideal/w_ms*100:5.1f}% MFU)"
        )
    print(
        f"  {'TOTAL':12s} ideal={tot_ideal:.3f}ms | "
        f"cold={tot_cold:.3f}ms ({tot_ideal/tot_cold*100:5.1f}% MFU) | "
        f"warm={tot_warm:.3f}ms ({tot_ideal/tot_warm*100:5.1f}% MFU)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "npu", "cpu"])
    parser.add_argument("--backend", default="v4", choices=["v4", "v5_ring"])
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args()

    has_npu = False
    if args.device in ("auto", "npu"):
        try:
            import torch_npu  # noqa: F401
            has_npu = torch.npu.is_available()
        except Exception:
            has_npu = False

    if args.device == "npu" and not has_npu:
        raise RuntimeError("NPU requested but torch_npu / NPU is not available")
    device = torch.device("npu" if (args.device != "cpu" and has_npu) else "cpu")

    if device.type == "npu":
        torch.npu.set_device(0)
        torch.npu.set_device_limit(0, cube_num=32, vector_num=64)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})

    test_callsites(device, args.backend, small=args.small)
    test_edge_cases(device, args.backend)
    test_e2e_dense_pffn(device, args.backend)
    if args.bench:
        bench_callsites(device, args.backend, iters=args.iters)


if __name__ == "__main__":
    main()
