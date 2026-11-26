#!/usr/bin/env python3
"""Accuracy + performance test for a fused DensePFFN SwiGLU FFN op.

Whole FFN in one fused pass: up-GEMM + bias, chunk, dual LayerNorm, SiLU gate
and multiply (SwiGLU), down-GEMM + bias. GEMMs are fused-cast bf16 Triton
kernels (fp32 operands cast to bf16 in-register, fp32 accumulate); norm stats
run fp32; output is fp32. CPU golden (fixed seed, fp32) via autograd through a
pure-torch forward; device runs the fused fwd + bwd.

Shape: B=1024, T=56, D=1232, P=2464 (up width 2P=4928), fp32 in/out, bf16
internal. Single fixed config (no autotune). Path: layer_norm, gate_first=False.

Usage: 
python testcase_dense_pffn.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import triton
import triton.language as tl

# Fixed launch configs (no autotuning).
_GEMM_BM, _GEMM_BN, _GEMM_BK, _GEMM_GROUP_M = 128, 256, 64, 8
_GEMM_WARPS, _GEMM_STAGES = 8, 3
_UP_XBLOCK, _UP_WARPS = 1, 8

_WIDTH_FACTOR = 2
_LN_GRAD_MASK = [True, True, True]


# ===========================================================================
# Fused Triton kernels
# ===========================================================================
@triton.jit
def _dense_pffn_gemm_kernel(  # pylint: disable=invalid-name
    a_ptr, w_ptr, o_ptr, M, N, K,
    sab, sam, sak, swb, swk, swn, sob, som, son,
    USE_BF16: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
) -> None:
    """Fused-cast batched GEMM: a [batch, M, K] @ w [batch, K, N] -> [batch, M, N]."""
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GROUP_M * grid_n
    gid = pid // width
    first_m = gid * GROUP_M
    group_sz = tl.minimum(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_sz)
    pid_n = (pid % width) // group_sz
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + bid * sab + (rm[:, None] * sam + rk[None, :] * sak)
    w_ptrs = w_ptr + bid * swb + (rk[:, None] * swk + rn[None, :] * swn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a_mask = (rm[:, None] < M) & (rk[None, :] + k0 < K)
        w_mask = (rk[:, None] + k0 < K) & (rn[None, :] < N)
        a_val = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w_val = tl.load(w_ptrs, mask=w_mask, other=0.0)
        if USE_BF16:
            acc += tl.dot(a_val.to(tl.bfloat16), w_val.to(tl.bfloat16))
        else:
            acc += tl.dot(a_val, w_val, input_precision="tf32")
        a_ptrs += BK * sak
        w_ptrs += BK * swk
    o_ptrs = o_ptr + bid * sob + (rm[:, None] * som + rn[None, :] * son)
    o_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_mask)


@triton.jit
def _dense_pffn_up_epilogue_kernel(  # pylint: disable=invalid-name
    buf_ptr, bias_ptr, gamma_x_ptr, beta_x_ptr, gamma_gate_ptr, beta_gate_ptr,
    out_ptr,
    B, T, P: tl.constexpr, eps,
    BLOCK_P: tl.constexpr, GATE_FIRST: tl.constexpr, XBLOCK: tl.constexpr,
) -> None:
    """Fused bias-add, chunk, dual LayerNorm, SiLU-gate -> SwiGLU [T, B, P]."""
    width = 2 * P
    x_off = P if GATE_FIRST else 0
    gate_off = 0 if GATE_FIRST else P
    row = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    row_mask = row < (T * B)
    t = row // B
    cols = tl.arange(0, BLOCK_P)
    col_mask = cols < P
    mask = row_mask[:, None] & col_mask[None, :]
    buf_base = row[:, None] * width + cols[None, :]
    bias_base = t[:, None] * width + cols[None, :]
    inv_p = 1.0 / P
    x_half = tl.load(buf_ptr + buf_base + x_off, mask=mask, other=0.0).to(
        tl.float32
    ) + tl.load(bias_ptr + bias_base + x_off, mask=mask, other=0.0)
    mean_x = tl.sum(tl.where(mask, x_half, 0.0), axis=1)[:, None] * inv_p
    centered_x = tl.where(mask, x_half - mean_x, 0.0)
    var_x = tl.sum(centered_x * centered_x, axis=1)[:, None] * inv_p
    gamma_x = tl.load(gamma_x_ptr + cols, mask=col_mask, other=0.0)[None, :]
    normed_x = (x_half - mean_x) * tl.math.rsqrt(var_x + eps) * gamma_x
    beta_x = tl.load(beta_x_ptr + cols, mask=col_mask, other=0.0)[None, :]
    normed_x = normed_x + beta_x
    gate = tl.load(buf_ptr + buf_base + gate_off, mask=mask, other=0.0).to(tl.float32)
    gate += tl.load(bias_ptr + bias_base + gate_off, mask=mask, other=0.0)
    mean_g = tl.sum(tl.where(mask, gate, 0.0), axis=1)[:, None] * inv_p
    centered_g = tl.where(mask, gate - mean_g, 0.0)
    var_g = tl.sum(centered_g * centered_g, axis=1)[:, None] * inv_p
    gamma_gate = tl.load(gamma_gate_ptr + cols, mask=col_mask, other=0.0)[None, :]
    normed_gate = (gate - mean_g) * tl.math.rsqrt(var_g + eps) * gamma_gate
    beta_gate = tl.load(beta_gate_ptr + cols, mask=col_mask, other=0.0)[None, :]
    normed_gate = normed_gate + beta_gate
    silu_gate = normed_gate * tl.sigmoid(normed_gate)
    result = normed_x * silu_gate
    out_base = row[:, None] * P + cols[None, :]
    tl.store(out_ptr + out_base, result.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dense_pffn_down_epilogue_kernel(  # pylint: disable=invalid-name
    raw_ptr, bias_ptr, out_ptr,
    B, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
) -> None:
    """Transpose-fused bias-add: raw [T, B, D] -> out2 [B, T, D] fp32."""
    row = tl.program_id(0)
    b_idx = row // T
    t = row % T
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    v = tl.load(raw_ptr + (t * B + b_idx) * D + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    bias = tl.load(bias_ptr + t * D + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * D + cols, v + bias, mask=mask)


# ===========================================================================
# Launchers
# ===========================================================================
def _fc_bmm(a: torch.Tensor, w: torch.Tensor, *, out_dtype: torch.dtype,
            use_bf16: bool) -> torch.Tensor:
    """Fused-cast batched GEMM a @ w over the batch axis -> [batch, M, N]."""
    n_batch, m_rows, k_dim = a.shape
    n_cols = w.shape[2]
    out = torch.empty((n_batch, m_rows, n_cols), device=a.device, dtype=out_dtype)
    grid = (triton.cdiv(m_rows, _GEMM_BM) * triton.cdiv(n_cols, _GEMM_BN), n_batch)
    _dense_pffn_gemm_kernel[grid](
        a, w, out, m_rows, n_cols, k_dim,
        a.stride(0), a.stride(1), a.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        use_bf16,
        BM=_GEMM_BM, BN=_GEMM_BN, BK=_GEMM_BK, GROUP_M=_GEMM_GROUP_M,
        num_warps=_GEMM_WARPS, num_stages=_GEMM_STAGES,
    )
    return out


def launch(x, weight_1, bias_1, gamma_x, beta_x, gamma_gate, beta_gate,
           weight_2, bias_2, eps):
    """Fused DensePFFN forward -> (out2 [B,T,D] fp32, buf [T,B,2P] bf16)."""
    n_batch, n_tokens, token_dim = x.shape
    proj_dim = weight_1.shape[2] // _WIDTH_FACTOR
    dt = torch.bfloat16

    buf = _fc_bmm(x.permute(1, 0, 2), weight_1, out_dtype=dt, use_bf16=True)

    swiglu = torch.empty((n_tokens, n_batch, proj_dim), device=x.device, dtype=dt)
    block_p = triton.next_power_of_2(proj_dim)
    n_rows = n_tokens * n_batch
    _dense_pffn_up_epilogue_kernel[(triton.cdiv(n_rows, _UP_XBLOCK),)](
        buf, bias_1, gamma_x, beta_x, gamma_gate, beta_gate, swiglu,
        n_batch, n_tokens, proj_dim, eps, block_p, False,
        XBLOCK=_UP_XBLOCK, num_warps=_UP_WARPS,
    )

    raw = _fc_bmm(swiglu, weight_2, out_dtype=dt, use_bf16=True)

    out2 = torch.empty((n_batch, n_tokens, token_dim), device=x.device,
                       dtype=x.dtype)
    block_d = triton.next_power_of_2(token_dim)
    _dense_pffn_down_epilogue_kernel[(n_batch * n_tokens,)](
        raw, bias_2, out2, n_batch, n_tokens, token_dim, block_d,
    )
    return out2, buf


def backward_launch(grad_out2, x, weight_1, bias_1, gamma_x, beta_x, gamma_gate,
                    beta_gate, weight_2, buf, eps):
    """Fused DensePFFN backward (LayerNorm path) -> the 9 input grads.

    Reconstructs the forward chain from the saved pre-bias ``buf`` (no up-GEMM
    recompute), walks the SwiGLU / dual-LayerNorm chain backward with aten ops,
    and runs the four grad GEMMs through the fused-cast bf16 kernel.
    """
    normalized_shape = [gamma_x.shape[0]]
    out1 = buf.permute(1, 0, 2).float() + bias_1  # [B, T, 2P]
    x_half, gate = out1.chunk(_WIDTH_FACTOR, dim=-1)  # gate_first=False
    ax, mean_x, rstd_x = torch.ops.aten.native_layer_norm(
        x_half, normalized_shape, gamma_x, beta_x, eps)
    ag, mean_g, rstd_g = torch.ops.aten.native_layer_norm(
        gate, normalized_shape, gamma_gate, beta_gate, eps)
    sig_ag = torch.sigmoid(ag)
    silu_ag = ag * sig_ag

    grad_raw = grad_out2.permute(1, 0, 2)  # [T, B, D]
    grad_bias_2 = grad_out2.sum(dim=0)

    swiglu = ax * silu_ag  # [B, T, P]
    grad_weight_2 = _fc_bmm(
        swiglu.permute(1, 0, 2).transpose(1, 2), grad_raw,
        out_dtype=torch.float32, use_bf16=True)  # [T, P, D]
    grad_sw_p = _fc_bmm(
        grad_raw, weight_2.transpose(1, 2),
        out_dtype=torch.float32, use_bf16=True)  # [T, B, P]
    grad_swiglu = grad_sw_p.permute(1, 0, 2)  # [B, T, P]

    # SwiGLU product rule through swiglu = ax * silu(ag).
    grad_ax = grad_swiglu * silu_ag
    grad_ag = grad_swiglu * ax
    deriv = sig_ag * (1.0 + ag * (1.0 - sig_ag))
    grad_ag = grad_ag * deriv
    grad_x_half, dgamma_x, dbeta_x = torch.ops.aten.native_layer_norm_backward(
        grad_ax, x_half, normalized_shape, mean_x, rstd_x, gamma_x, beta_x,
        _LN_GRAD_MASK)
    grad_gate, dgamma_gate, dbeta_gate = torch.ops.aten.native_layer_norm_backward(
        grad_ag, gate, normalized_shape, mean_g, rstd_g, gamma_gate, beta_gate,
        _LN_GRAD_MASK)
    grad_out1 = torch.cat([grad_x_half, grad_gate], dim=-1)  # [B, T, 2P]
    grad_bias_1 = grad_out1.sum(dim=0)

    grad_buf_p = grad_out1.permute(1, 0, 2)  # [T, B, 2P]
    grad_weight_1 = _fc_bmm(
        x.permute(1, 0, 2).transpose(1, 2), grad_buf_p,
        out_dtype=torch.float32, use_bf16=True)  # [T, D, 2P]
    grad_x = _fc_bmm(
        grad_buf_p, weight_1.transpose(1, 2),
        out_dtype=torch.float32, use_bf16=True).permute(1, 0, 2)  # [B, T, D]

    return (grad_x, grad_weight_1, grad_bias_1, dgamma_x, dbeta_x,
            dgamma_gate, dbeta_gate, grad_weight_2, grad_bias_2)


# ===========================================================================
# Pure-PyTorch reference oracle
# ===========================================================================
def torch_ref(x, weight_1, bias_1, gamma_x, beta_x, gamma_gate, beta_gate,
              weight_2, bias_2, eps):
    import torch.nn.functional as F
    buf = torch.matmul(x.permute(1, 0, 2), weight_1)
    out1 = buf.permute(1, 0, 2) + bias_1
    x_half, gate_half = out1.chunk(_WIDTH_FACTOR, dim=-1)
    proj_dim = x_half.shape[-1]
    x_half = F.layer_norm(x_half, (proj_dim,), gamma_x, beta_x, eps)
    gate_half = F.layer_norm(gate_half, (proj_dim,), gamma_gate, beta_gate, eps)
    swiglu = x_half * F.silu(gate_half)
    out2 = torch.matmul(swiglu.permute(1, 0, 2), weight_2).permute(1, 0, 2) + bias_2
    return out2


def bench(fn, device: torch.device, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1024, help="B (token/batch count)")
    ap.add_argument("--tokens", type=int, default=56, help="T")
    ap.add_argument("--token-dim", type=int, default=1232, help="D")
    ap.add_argument("--proj-dim", type=int, default=2464, help="P (up width 2P)")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise SystemExit("No CUDA/ROCm device available — Triton kernels require a GPU.")

    B, T, D, P = args.batch, args.tokens, args.token_dim, args.proj_dim
    eps = args.eps
    rel_gate = 1e-2

    print("=" * 72)
    print(f"dense_pffn | device={device} dtype=fp32-in/bf16-internal "
          f"B={B} T={T} D={D} P={P} seed={args.seed}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)} | torch={torch.__version__} "
              f"| triton={triton.__version__}")
    print("=" * 72)

    # raw data on CPU (fp32, fixed seed)
    gen = torch.Generator().manual_seed(args.seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.float32)

    x = randn(B, T, D)
    # 1/sqrt(fan_in) scaled weights keep the GEMM outputs O(1).
    weight_1 = randn(T, D, 2 * P, scale=D ** -0.5)
    bias_1 = randn(T, 2 * P, scale=0.1)
    gamma_x = randn(P, scale=0.0) + 1.0
    beta_x = randn(P, scale=0.1)
    gamma_gate = randn(P, scale=0.0) + 1.0
    beta_gate = randn(P, scale=0.1)
    weight_2 = randn(T, P, D, scale=P ** -0.5)
    bias_2 = randn(T, D, scale=0.1)
    grad_out2 = randn(B, T, D)

    # CPU golden (out2 + all input grads via autograd through torch_ref)
    leaves = {
        "x": x, "weight_1": weight_1, "bias_1": bias_1, "gamma_x": gamma_x,
        "beta_x": beta_x, "gamma_gate": gamma_gate, "beta_gate": beta_gate,
        "weight_2": weight_2, "bias_2": bias_2,
    }
    req = {k: v.detach().clone().requires_grad_(True) for k, v in leaves.items()}
    golden_out = torch_ref(
        req["x"], req["weight_1"], req["bias_1"], req["gamma_x"], req["beta_x"],
        req["gamma_gate"], req["beta_gate"], req["weight_2"], req["bias_2"], eps)
    golden_grads = torch.autograd.grad(golden_out, list(req.values()), grad_out2)
    golden_out = golden_out.detach()
    golden = dict(zip(req.keys(), golden_grads))

    d = {k: v.to(device) for k, v in leaves.items()}
    god = grad_out2.to(device)

    # device fused fwd + bwd
    out_dev, buf_dev = launch(
        d["x"], d["weight_1"], d["bias_1"], d["gamma_x"], d["beta_x"],
        d["gamma_gate"], d["beta_gate"], d["weight_2"], d["bias_2"], eps)
    (gx, gw1, gb1, dgx, dbx, dgg, dbg, gw2, gb2) = backward_launch(
        god, d["x"].to(torch.bfloat16), d["weight_1"], d["bias_1"], d["gamma_x"],
        d["beta_x"], d["gamma_gate"], d["beta_gate"], d["weight_2"], buf_dev, eps)
    dev_grads = {
        "x": gx, "weight_1": gw1, "bias_1": gb1, "gamma_x": dgx, "beta_x": dbx,
        "gamma_gate": dgg, "beta_gate": dbg, "weight_2": gw2, "bias_2": gb2,
    }

    # accuracy vs CPU golden (relative L2, bf16 gate)
    def rel_l2(dev_t, ref_t):
        diff = (dev_t.float() - ref_t.float()).norm()
        return (diff / (ref_t.float().norm() + 1e-12)).item()

    def report(name, dev_t, ref_t):
        m = (dev_t.cpu().float() - ref_t.float()).abs().max().item()
        r = rel_l2(dev_t.cpu(), ref_t)
        print(f"[{name}] max_abs_err={m:.3e}  rel_l2={r:.3e}  acc_ok={r <= rel_gate}")
        return r <= rel_gate

    ok = report("forward  out2 ", out_dev, golden_out)
    for name in leaves:
        ok &= report(f"backward g_{name:<9}", dev_grads[name], golden[name])
    print(f"ACCURACY: {'PASS' if ok else 'FAIL'}")

    # performance: device latency only
    xb = d["x"].to(torch.bfloat16)
    fwd_ms = bench(
        lambda: launch(d["x"], d["weight_1"], d["bias_1"], d["gamma_x"],
                       d["beta_x"], d["gamma_gate"], d["beta_gate"],
                       d["weight_2"], d["bias_2"], eps),
        device, args.iters, 10)
    bwd_ms = bench(
        lambda: backward_launch(god, xb, d["weight_1"], d["bias_1"], d["gamma_x"],
                                d["beta_x"], d["gamma_gate"], d["beta_gate"],
                                d["weight_2"], buf_dev, eps),
        device, args.iters, 10)
    print(f"[forward ] {fwd_ms:.4f} ms")
    print(f"[backward] {bwd_ms:.4f} ms")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
