"""TileLang-Ascend fused-cast batched GEMM kernels for DensePFFN on Ascend 950DT.

Provides two backends sharing the exact `_fc_bmm(a, w, *, out_dtype, use_bf16)` semantics:
1. `backend="v4"` (DEFAULT, on-board verified 6/6 PASS at 1650 MHz on Ascend 950DT):
   L2-Ping-Pong Super-Batch CV kernel (`256x256x128` 2-step `BK_L0=64`, 256B SimdVF FP32->BF16 cast,
   L2-pinned `NORMAL_LAST_VICTIM` workspace, streaming `NOTALLOC_DROP` input / `NOTALLOC_PW` output,
   proven deadlock-free `mode=0` + `mode=4` cross-core synchronization).
2. `backend="v5_ring"` (OPT-IN, CPU-lowered cleanly; Pure Cube verified 88.8%/90.1% MFU; full CV ring
   pending on-board verification after 950DT device 0 reset):
   Pure-Hardware Zero-AIC-Barrier Persistent Ring (`L1_SLOTS=3`, hardware `unit_flag_ctrl=2/3`,
   batch-rotated tail-tile swizzle, single continuous AIC persistent loop across all batches with
   zero AIC barriers and zero GM polling).
"""

import os
from typing import Dict, Optional, Tuple

import torch

_CXX_FLAGS = []
_cxx_root = "/usr/include/c++"
if os.path.isdir(_cxx_root):
    _versions = sorted(name for name in os.listdir(_cxx_root) if name[:1].isdigit())
    if _versions:
        _base = os.path.join(_cxx_root, _versions[-1])
        _CXX_FLAGS.append("-I" + _base)
        _triples = sorted(
            [n for n in os.listdir(_base) if os.path.isfile(os.path.join(_base, n, "bits", "c++config.h"))],
            key=lambda n: ("linux" not in n, n),
        )
        if _triples:
            _CXX_FLAGS.append("-I" + os.path.join(_base, _triples[0]))

_WS_CACHE: Dict[Tuple, torch.Tensor] = {}
_KERNEL_CACHE: Dict[Tuple, object] = {}


def _get_ws(device: torch.device, shape: Tuple[int, ...], tag: str) -> torch.Tensor:
    key = (str(device), tag, shape)
    buf = _WS_CACHE.get(key)
    if buf is None or buf.shape != shape or buf.device != device:
        buf = torch.empty(shape, device=device, dtype=torch.bfloat16)
        _WS_CACHE[key] = buf
    return buf


def _make_v4_prim_func(
    a_dtype: str = "float32",
    w_dtype: str = "float32",
    o_dtype: str = "bfloat16",
    trans_a: bool = False,
    trans_b: bool = False,
    BM: int = 256,
    BN: int = 256,
    BK: int = 128,
    BK_L0: int = 64,
    L1_SLOTS: int = 2,
    GROUP_M: int = 4,
    G_B: int = 2,
    RB: int = 8,
    CB: int = 1280,
    ws_store_l2: int = 1,
    ws_load_l2: int = 1,
    core_count: int = 32,
):
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    assert BK in (64, 128) and BK_L0 in (64, 128) and BK % BK_L0 == 0
    sub_k_steps = BK // BK_L0

    @T.macro
    def _pair_set(pipe, flag):
        T.ascend_cross_core_set_flag(4, pipe, flag)
        T.ascend_cross_core_set_flag(4, pipe, flag + 16)

    @T.macro
    def _pair_wait(pipe, flag):
        T.ascend_cross_core_wait_flag(4, pipe, flag)
        T.ascend_cross_core_wait_flag(4, pipe, flag + 16)

    @T.macro
    def _manual_sync(event, index):
        T.ascend_set_flag(event, index)
        T.ascend_wait_flag(event, index)

    @T.macro
    def _simdvf_cast_f32_bf16(src_ub, dst_ub, slot_idx, rows: int, cols: int):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            b16_128 = S.pset(16, "PAT_VL128")
            n_chunks = cols // 128
            for r in range(rows):
                for c in T.unroll(n_chunks):
                    v_even, v_odd = S.vld2(src_ub[slot_idx, r, c * 128], "DINTLV_B32")
                    b_even = S.vcvt(v_even, "bfloat16", full, round="ROUND_R", sat=False, part=0, mode="MODE_ZEROING")
                    b_odd = S.vcvt(v_odd, "bfloat16", full, round="ROUND_R", sat=False, part=1, mode="MODE_ZEROING")
                    merged = S.vor(T.reinterpret(b_even, "uint16x128"), T.reinterpret(b_odd, "uint16x128"), b16_128)
                    S.vsts(dst_ub[slot_idx, r, c * 128], T.reinterpret(merged, "bfloat16x128"), b16_128, "NORM_B16")

    @T.macro
    def _aiv_cast_3d_to_ws(
        src_gm, ws_bf16, f32_ub, bf16_ub,
        slot, b_start, cur_gb, R_dim, C_dim, vid,
        rb: int, cb: int, n_aiv: int, st_l2: int,
    ):
        r_blocks = T.alloc_var("int32", init=T.ceildiv(R_dim, rb))
        c_blocks = T.alloc_var("int32", init=T.ceildiv(C_dim, cb))
        rc_blocks = T.alloc_var("int32", init=r_blocks * c_blocks)
        total_tasks = T.alloc_var("int32", init=cur_gb * rc_blocks)
        my_tasks = T.alloc_var("int32", init=T.max((total_tasks - vid + n_aiv - 1) // n_aiv, 0))

        pf_lb = T.alloc_var("int32", init=0)
        pf_r0 = T.alloc_var("int32", init=0)
        pf_c0 = T.alloc_var("int32", init=0)
        pf_ar = T.alloc_var("int32", init=0)
        pf_ac = T.alloc_var("int32", init=0)
        cur_lb = T.alloc_var("int32", init=0)
        cur_r0 = T.alloc_var("int32", init=0)
        cur_c0 = T.alloc_var("int32", init=0)
        cur_ar = T.alloc_var("int32", init=0)
        cur_ac = T.alloc_var("int32", init=0)

        if my_tasks > 0:
            t0 = vid
            pf_lb = t0 // rc_blocks
            rem0 = t0 - pf_lb * rc_blocks
            pf_r0 = (rem0 // c_blocks) * rb
            pf_c0 = (rem0 % c_blocks) * cb
            pf_ar = T.min(R_dim - pf_r0, rb)
            pf_ac = T.min(C_dim - pf_c0, cb)
            T.ascend_wait_flag("V_MTE2", 0)
            T.copy(
                src_gm[b_start + pf_lb, pf_r0 : pf_r0 + pf_ar, pf_c0 : pf_c0 + pf_ac],
                f32_ub[0, 0:pf_ar, 0:pf_ac],
                l2_cache_ctrl="NOTALLOC_DROP",
            )
            T.ascend_set_flag("MTE2_V", 0)

            for tk in range(my_tasks):
                u = tk % 2
                nu = (tk + 1) % 2
                cur_lb = pf_lb
                cur_r0 = pf_r0
                cur_c0 = pf_c0
                cur_ar = pf_ar
                cur_ac = pf_ac

                if tk + 1 < my_tasks:
                    tn = vid + (tk + 1) * n_aiv
                    pf_lb = tn // rc_blocks
                    remn = tn - pf_lb * rc_blocks
                    pf_r0 = (remn // c_blocks) * rb
                    pf_c0 = (remn % c_blocks) * cb
                    pf_ar = T.min(R_dim - pf_r0, rb)
                    pf_ac = T.min(C_dim - pf_c0, cb)
                    T.ascend_wait_flag("V_MTE2", nu)
                    T.copy(
                        src_gm[b_start + pf_lb, pf_r0 : pf_r0 + pf_ar, pf_c0 : pf_c0 + pf_ac],
                        f32_ub[nu, 0:pf_ar, 0:pf_ac],
                        l2_cache_ctrl="NOTALLOC_DROP",
                    )
                    T.ascend_set_flag("MTE2_V", nu)

                T.ascend_wait_flag("MTE2_V", u)
                T.ascend_wait_flag("MTE3_V", u)
                _simdvf_cast_f32_bf16(f32_ub, bf16_ub, u, rb, cb)
                T.ascend_set_flag("V_MTE2", u)
                T.ascend_set_flag("V_MTE3", u)

                T.ascend_wait_flag("V_MTE3", u)
                T.copy(
                    bf16_ub[u, 0:cur_ar, 0:cur_ac],
                    ws_bf16[slot, cur_lb, cur_r0 : cur_r0 + cur_ar, cur_c0 : cur_c0 + cur_ac],
                    l2_cache_ctrl=st_l2,
                )
                T.ascend_set_flag("MTE3_V", u)

    @T.macro
    def _load_cube_ab(
        A_gm, W_gm, ws_a, ws_w, a_l1, b_l1,
        s, slot, b_start, lb, m0, n0, k0, am, an, ak,
        a_f32: bool, w_f32: bool, ta: bool, tb: bool, ld_l2: int,
    ):
        if a_f32:
            if ta:
                T.copy(ws_a[slot, lb, k0 : k0 + ak, m0 : m0 + am], a_l1[s, 0:ak, 0:am], l2_cache_ctrl=ld_l2)
            else:
                T.copy(ws_a[slot, lb, m0 : m0 + am, k0 : k0 + ak], a_l1[s, 0:am, 0:ak], l2_cache_ctrl=ld_l2)
        else:
            if ta:
                T.copy(A_gm[b_start + lb, k0 : k0 + ak, m0 : m0 + am], a_l1[s, 0:ak, 0:am], l2_cache_ctrl=ld_l2)
            else:
                T.copy(A_gm[b_start + lb, m0 : m0 + am, k0 : k0 + ak], a_l1[s, 0:am, 0:ak], l2_cache_ctrl=ld_l2)

        if w_f32:
            if tb:
                T.copy(ws_w[slot, lb, n0 : n0 + an, k0 : k0 + ak], b_l1[s, 0:an, 0:ak], l2_cache_ctrl=ld_l2)
            else:
                T.copy(ws_w[slot, lb, k0 : k0 + ak, n0 : n0 + an], b_l1[s, 0:ak, 0:an], l2_cache_ctrl=ld_l2)
        else:
            if tb:
                T.copy(W_gm[b_start + lb, n0 : n0 + an, k0 : k0 + ak], b_l1[s, 0:an, 0:ak], l2_cache_ctrl=ld_l2)
            else:
                T.copy(W_gm[b_start + lb, k0 : k0 + ak, n0 : n0 + an], b_l1[s, 0:ak, 0:an], l2_cache_ctrl=ld_l2)

    batch = T.dynamic("batch")
    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")
    sab = T.dynamic("sab")
    sam = T.dynamic("sam")
    swb = T.dynamic("swb")
    swk = T.dynamic("swk")

    a_is_f32 = (a_dtype == "float32")
    w_is_f32 = (w_dtype == "float32")
    need_aiv = a_is_f32 or w_is_f32
    c_slots = 1 if (BM * BN * 4 > 128 * 1024) else 2
    num_aiv = core_count * 2

    a_rows_l1 = BK if trans_a else BM
    a_cols_l1 = BM if trans_a else BK
    b_rows_l1 = BN if trans_b else BK
    b_cols_l1 = BK if trans_b else BN

    a_shape = (batch, K, M) if trans_a else (batch, M, K)
    w_shape = (batch, N, K) if trans_b else (batch, K, N)
    ws_a_shape = ((2, G_B, K, M) if trans_a else (2, G_B, M, K)) if a_is_f32 else (1, 1, 1, 1)
    ws_w_shape = ((2, G_B, N, K) if trans_b else (2, G_B, K, N)) if w_is_f32 else (1, 1, 1, 1)

    @T.prim_func
    def dense_pffn_gemm_v4(
        A: T.StridedTensor(a_shape, (sab, sam, 1), a_dtype),
        W: T.StridedTensor(w_shape, (swb, swk, 1), w_dtype),
        ws_a: T.Tensor(ws_a_shape, "bfloat16"),
        ws_w: T.Tensor(ws_w_shape, "bfloat16"),
        O: T.Tensor((batch, M, N), o_dtype),
    ):
        with T.Kernel(core_count) as cid:
            a_l1 = T.alloc_l1((L1_SLOTS, a_rows_l1, a_cols_l1), "bfloat16")
            b_l1 = T.alloc_l1((L1_SLOTS, b_rows_l1, b_cols_l1), "bfloat16")
            a_l0 = T.alloc_l0a((2, BM, BK_L0), "bfloat16")
            b_l0 = T.alloc_l0b((2, BN, BK_L0), "bfloat16")
            c_l0c = T.alloc_l0c((c_slots, BM, BN), "float32")

            if need_aiv:
                f32_ub = T.alloc_shared((2, RB, CB), "float32")
                bf16_ub = T.alloc_shared((2, RB, CB), "bfloat16")

            m_tiles = T.alloc_var("int32", init=T.ceildiv(M, BM))
            n_tiles = T.alloc_var("int32", init=T.ceildiv(N, BN))
            k_tiles = T.alloc_var("int32", init=T.ceildiv(K, BK))
            tiles_per_b = T.alloc_var("int32", init=m_tiles * n_tiles)
            width = T.alloc_var("int32", init=GROUP_M * n_tiles)
            num_sb = T.alloc_var("int32", init=T.ceildiv(batch, G_B))

            with T.Cube():
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 1)
                for cs in T.unroll(c_slots):
                    T.ascend_set_flag("FIX_M", cs)
                for s in T.unroll(L1_SLOTS):
                    T.ascend_set_flag("MTE1_MTE2", s)
                if need_aiv:
                    _pair_set("PIPE_MTE2", 2)
                    _pair_set("PIPE_MTE2", 3)

                lb_v = T.alloc_var("int32", init=0)
                m0_v = T.alloc_var("int32", init=0)
                n0_v = T.alloc_var("int32", init=0)
                am_v = T.alloc_var("int32", init=0)
                an_v = T.alloc_var("int32", init=0)
                ak_v = T.alloc_var("int32", init=0)

                pf_ti = T.alloc_var("int32", init=0)
                pf_kt = T.alloc_var("int32", init=0)
                pf_lb = T.alloc_var("int32", init=0)
                pf_m0 = T.alloc_var("int32", init=0)
                pf_n0 = T.alloc_var("int32", init=0)
                pf_am = T.alloc_var("int32", init=0)
                pf_an = T.alloc_var("int32", init=0)
                pf_ak = T.alloc_var("int32", init=0)
                sb_my_tiles = T.alloc_var("int32", init=0)
                sb_steps = T.alloc_var("int32", init=0)

                for sb in range(num_sb):
                    slot = sb % 2
                    b_start = sb * G_B
                    cur_gb = T.min(batch - b_start, G_B)
                    sb_total_tiles = cur_gb * tiles_per_b
                    sb_my_tiles = T.max((sb_total_tiles - cid + core_count - 1) // core_count, 0)
                    sb_steps = sb_my_tiles * k_tiles

                    if need_aiv:
                        _pair_wait("PIPE_MTE2", slot)

                    if sb_steps > 0:
                        pf_ti = 0
                        pf_kt = 0
                        t_idx0 = cid
                        pf_lb = t_idx0 // tiles_per_b
                        in_b0 = t_idx0 - pf_lb * tiles_per_b
                        gid0 = in_b0 // width
                        fm0 = gid0 * GROUP_M
                        gsz0 = T.min(m_tiles - fm0, GROUP_M)
                        w0 = in_b0 - gid0 * width
                        pm0 = fm0 + (w0 % gsz0)
                        pn0_r = w0 // gsz0
                        pn0 = T.if_then_else((gid0 & 1) == 1, n_tiles - 1 - pn0_r, pn0_r)
                        pf_m0 = pm0 * BM
                        pf_n0 = pn0 * BN
                        pf_am = T.min(M - pf_m0, BM)
                        pf_an = T.min(N - pf_n0, BN)
                        pf_ak = T.min(K, BK)
                        lb0 = pf_lb
                        m0_0 = pf_m0
                        n0_0 = pf_n0
                        am0 = pf_am
                        an0 = pf_an
                        ak0 = pf_ak
                        T.assume(lb0 >= 0 and lb0 < G_B and b_start + lb0 < batch)
                        T.assume(m0_0 >= 0 and am0 > 0 and am0 <= BM and m0_0 + am0 <= M)
                        T.assume(n0_0 >= 0 and an0 > 0 and an0 <= BN and n0_0 + an0 <= N)
                        T.assume(ak0 > 0 and ak0 <= BK and ak0 <= K)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        _load_cube_ab(
                            A, W, ws_a, ws_w, a_l1, b_l1,
                            0, slot, b_start, lb0, m0_0, n0_0, 0, am0, an0, ak0,
                            a_is_f32, w_is_f32, trans_a, trans_b, ws_load_l2,
                        )
                        T.ascend_set_flag("MTE2_MTE1", 0)
                        pf_kt = 1
                        if pf_kt == k_tiles:
                            pf_kt = 0
                            pf_ti = 1

                        for ti in range(sb_my_tiles):
                            t_idx = cid + ti * core_count
                            lb_v = t_idx // tiles_per_b
                            in_b = t_idx - lb_v * tiles_per_b
                            gid = in_b // width
                            first_m = gid * GROUP_M
                            group_sz = T.min(m_tiles - first_m, GROUP_M)
                            within = in_b - gid * width
                            pid_m = first_m + (within % group_sz)
                            pid_n_raw = within // group_sz
                            pid_n = T.if_then_else((gid & 1) == 1, n_tiles - 1 - pid_n_raw, pid_n_raw)
                            m0_v = pid_m * BM
                            n0_v = pid_n * BN
                            am_v = T.min(M - m0_v, BM)
                            an_v = T.min(N - n0_v, BN)
                            lb_r = lb_v
                            m0_r = m0_v
                            n0_r = n0_v
                            am = am_v
                            an = an_v
                            T.assume(lb_r >= 0 and lb_r < G_B and b_start + lb_r < batch)
                            T.assume(m0_r >= 0 and am > 0 and am <= BM and m0_r + am <= M)
                            T.assume(n0_r >= 0 and an > 0 and an <= BN and n0_r + an <= N)
                            c_slot = 0 if c_slots == 1 else (ti % 2)
                            T.ascend_wait_flag("FIX_M", c_slot)

                            for kt in range(k_tiles):
                                st_cur = ti * k_tiles + kt
                                s = st_cur % L1_SLOTS
                                ak_v = T.min(K - kt * BK, BK)
                                ak = ak_v
                                T.assume(ak > 0 and ak <= BK and kt * BK + ak <= K)

                                if st_cur + 1 < sb_steps:
                                    ns = (st_cur + 1) % L1_SLOTS
                                    if pf_kt == 0:
                                        nt_idx = cid + pf_ti * core_count
                                        pf_lb = nt_idx // tiles_per_b
                                        nin_b = nt_idx - pf_lb * tiles_per_b
                                        ngid = nin_b // width
                                        nfm = ngid * GROUP_M
                                        ngsz = T.min(m_tiles - nfm, GROUP_M)
                                        nw = nin_b - ngid * width
                                        npm = nfm + (nw % ngsz)
                                        npn_r = nw // ngsz
                                        npn = T.if_then_else((ngid & 1) == 1, n_tiles - 1 - npn_r, npn_r)
                                        pf_m0 = npm * BM
                                        pf_n0 = npn * BN
                                        pf_am = T.min(M - pf_m0, BM)
                                        pf_an = T.min(N - pf_n0, BN)
                                    pf_k0 = pf_kt * BK
                                    pf_ak = T.min(K - pf_k0, BK)
                                    plb = pf_lb
                                    pm0_r = pf_m0
                                    pn0_r = pf_n0
                                    pk0_r = pf_k0
                                    pam = pf_am
                                    pan = pf_an
                                    pak = pf_ak
                                    T.assume(plb >= 0 and plb < G_B and b_start + plb < batch)
                                    T.assume(pm0_r >= 0 and pam > 0 and pam <= BM and pm0_r + pam <= M)
                                    T.assume(pn0_r >= 0 and pan > 0 and pan <= BN and pn0_r + pan <= N)
                                    T.assume(pk0_r >= 0 and pak > 0 and pak <= BK and pk0_r + pak <= K)
                                    T.ascend_wait_flag("MTE1_MTE2", ns)
                                    _load_cube_ab(
                                        A, W, ws_a, ws_w, a_l1, b_l1,
                                        ns, slot, b_start, plb, pm0_r, pn0_r, pk0_r, pam, pan, pak,
                                        a_is_f32, w_is_f32, trans_a, trans_b, ws_load_l2,
                                    )
                                    T.ascend_set_flag("MTE2_MTE1", ns)
                                    pf_kt = pf_kt + 1
                                    if pf_kt == k_tiles:
                                        pf_kt = 0
                                        pf_ti = pf_ti + 1

                                T.ascend_wait_flag("MTE2_MTE1", s)
                                if sub_k_steps == 1:
                                    p = st_cur % 2
                                    T.ascend_wait_flag("M_MTE1", p)
                                    if trans_a:
                                        T.copy(a_l1[s, 0:ak, 0:am], a_l0[p, 0:am, 0:ak], transpose=True)
                                    else:
                                        T.copy(a_l1[s, 0:am, 0:ak], a_l0[p, 0:am, 0:ak])
                                    if trans_b:
                                        T.copy(b_l1[s, 0:an, 0:ak], b_l0[p, 0:an, 0:ak])
                                    else:
                                        T.copy(b_l1[s, 0:ak, 0:an], b_l0[p, 0:an, 0:ak], transpose=True)
                                    T.ascend_set_flag("MTE1_MTE2", s)
                                    _manual_sync("MTE1_M", p)
                                    T.gemm(
                                        a_l0[p, 0:am, 0:ak],
                                        b_l0[p, 0:an, 0:ak],
                                        c_l0c[c_slot, 0:am, 0:an],
                                        transpose_A=False,
                                        transpose_B=True,
                                        clear_accum=(kt == 0),
                                    )
                                    T.ascend_set_flag("M_MTE1", p)
                                else:
                                    ak_sub0 = T.min(ak, BK_L0)
                                    T.assume(ak_sub0 > 0 and ak_sub0 <= BK_L0)
                                    T.ascend_wait_flag("M_MTE1", 0)
                                    if trans_a:
                                        T.copy(a_l1[s, 0:ak_sub0, 0:am], a_l0[0, 0:am, 0:ak_sub0], transpose=True)
                                    else:
                                        T.copy(a_l1[s, 0:am, 0:ak_sub0], a_l0[0, 0:am, 0:ak_sub0])
                                    if trans_b:
                                        T.copy(b_l1[s, 0:an, 0:ak_sub0], b_l0[0, 0:an, 0:ak_sub0])
                                    else:
                                        T.copy(b_l1[s, 0:ak_sub0, 0:an], b_l0[0, 0:an, 0:ak_sub0], transpose=True)
                                    if ak <= BK_L0:
                                        T.ascend_set_flag("MTE1_MTE2", s)
                                    _manual_sync("MTE1_M", 0)
                                    T.gemm(
                                        a_l0[0, 0:am, 0:ak_sub0],
                                        b_l0[0, 0:an, 0:ak_sub0],
                                        c_l0c[c_slot, 0:am, 0:an],
                                        transpose_A=False,
                                        transpose_B=True,
                                        clear_accum=(kt == 0),
                                    )
                                    T.ascend_set_flag("M_MTE1", 0)

                                    if ak > BK_L0:
                                        ak_sub1 = ak - BK_L0
                                        T.assume(ak_sub1 > 0 and ak_sub1 <= BK_L0)
                                        T.ascend_wait_flag("M_MTE1", 1)
                                        if trans_a:
                                            T.copy(a_l1[s, BK_L0 : BK_L0 + ak_sub1, 0:am], a_l0[1, 0:am, 0:ak_sub1], transpose=True)
                                        else:
                                            T.copy(a_l1[s, 0:am, BK_L0 : BK_L0 + ak_sub1], a_l0[1, 0:am, 0:ak_sub1])
                                        if trans_b:
                                            T.copy(b_l1[s, 0:an, BK_L0 : BK_L0 + ak_sub1], b_l0[1, 0:an, 0:ak_sub1])
                                        else:
                                            T.copy(b_l1[s, BK_L0 : BK_L0 + ak_sub1, 0:an], b_l0[1, 0:an, 0:ak_sub1], transpose=True)
                                        T.ascend_set_flag("MTE1_MTE2", s)
                                        _manual_sync("MTE1_M", 1)
                                        T.gemm(
                                            a_l0[1, 0:am, 0:ak_sub1],
                                            b_l0[1, 0:an, 0:ak_sub1],
                                            c_l0c[c_slot, 0:am, 0:an],
                                            transpose_A=False,
                                            transpose_B=True,
                                            clear_accum=False,
                                        )
                                        T.ascend_set_flag("M_MTE1", 1)

                            _manual_sync("M_FIX", c_slot)
                            T.copy(
                                c_l0c[c_slot, 0:am, 0:an],
                                O[b_start + lb_r, m0_r : m0_r + am, n0_r : n0_r + an],
                                l2_cache_ctrl="NOTALLOC_PW",
                            )
                            T.ascend_set_flag("FIX_M", c_slot)

                    if need_aiv:
                        T.ascend_cross_core_set_flag(0, "PIPE_MTE2", 10 + slot)
                        T.ascend_cross_core_wait_flag(0, "PIPE_MTE2", 10 + slot)
                        _pair_set("PIPE_MTE2", 2 + slot)

                for cs in T.unroll(c_slots):
                    T.ascend_wait_flag("FIX_M", cs)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("M_MTE1", 1)
                for s in T.unroll(L1_SLOTS):
                    T.ascend_wait_flag("MTE1_MTE2", s)

            with T.Vector() as sid:
                if need_aiv:
                    vid = cid * 2 + sid
                    T.ascend_set_flag("V_MTE2", 0)
                    T.ascend_set_flag("V_MTE2", 1)
                    T.ascend_set_flag("MTE3_V", 0)
                    T.ascend_set_flag("MTE3_V", 1)

                    for sb in range(num_sb):
                        slot = sb % 2
                        b_start = sb * G_B
                        cur_gb = T.min(batch - b_start, G_B)
                        T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 2 + slot)
                        if a_is_f32:
                            R_a = K if trans_a else M
                            C_a = M if trans_a else K
                            _aiv_cast_3d_to_ws(A, ws_a, f32_ub, bf16_ub, slot, b_start, cur_gb, R_a, C_a, vid, RB, CB, num_aiv, ws_store_l2)
                        if w_is_f32:
                            R_w = N if trans_b else K
                            C_w = K if trans_b else N
                            _aiv_cast_3d_to_ws(W, ws_w, f32_ub, bf16_ub, slot, b_start, cur_gb, R_w, C_w, vid, RB, CB, num_aiv, ws_store_l2)
                        T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 8 + slot)
                        T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 8 + slot)
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE3", slot)

                    T.ascend_wait_flag("V_MTE2", 0)
                    T.ascend_wait_flag("V_MTE2", 1)
                    T.ascend_wait_flag("MTE3_V", 0)
                    T.ascend_wait_flag("MTE3_V", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 2)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 3)

    return dense_pffn_gemm_v4


def _make_v5_prim_func(
    a_dtype: str = "float32",
    w_dtype: str = "float32",
    o_dtype: str = "bfloat16",
    trans_a: bool = False,
    trans_b: bool = False,
    BM: int = 256,
    BN: int = 256,
    BK: int = 128,
    BK_L0: int = 64,
    L1_SLOTS: int = 3,
    GROUP_M: int = 4,
    RING_R: int = 4,
    RB: int = 8,
    CB: int = 1280,
    ws_store_l2: int = 1,
    ws_load_l2: int = 1,
    use_unit_flag: bool = True,
    use_dn2nz_b: bool = False,
    use_batch_rot: bool = True,
    core_count: int = 32,
):
    """V5.1 Pure-Hardware Zero-AIC-Barrier Persistent Ring (CPU-lowered cleanly; opt-in)."""
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    assert BK in (64, 128) and BK_L0 in (64, 128) and BK % BK_L0 == 0
    assert 1 <= RING_R <= 4
    sub_k_steps = BK // BK_L0

    @T.macro
    def _pair_set(pipe, flag):
        T.ascend_cross_core_set_flag(4, pipe, flag)
        T.ascend_cross_core_set_flag(4, pipe, flag + 16)

    @T.macro
    def _pair_wait(pipe, flag):
        T.ascend_cross_core_wait_flag(4, pipe, flag)
        T.ascend_cross_core_wait_flag(4, pipe, flag + 16)

    @T.macro
    def _manual_sync(event, index):
        T.ascend_set_flag(event, index)
        T.ascend_wait_flag(event, index)

    @T.macro
    def _simdvf_cast_f32_bf16(src_ub, dst_ub, slot_idx, rows: int, cols: int):
        with T.SimdVF():
            full = S.pset(32, "PAT_ALL")
            b16_128 = S.pset(16, "PAT_VL128")
            n_chunks = cols // 128
            for r in range(rows):
                for c in T.unroll(n_chunks):
                    v_even, v_odd = S.vld2(src_ub[slot_idx, r, c * 128], "DINTLV_B32")
                    b_even = S.vcvt(v_even, "bfloat16", full, round="ROUND_R", sat=False, part=0, mode="MODE_ZEROING")
                    b_odd = S.vcvt(v_odd, "bfloat16", full, round="ROUND_R", sat=False, part=1, mode="MODE_ZEROING")
                    merged = S.vor(T.reinterpret(b_even, "uint16x128"), T.reinterpret(b_odd, "uint16x128"), b16_128)
                    S.vsts(dst_ub[slot_idx, r, c * 128], T.reinterpret(merged, "bfloat16x128"), b16_128, "NORM_B16")

    @T.macro
    def _aiv_cast_single_batch(
        src_gm, ws_bf16, f32_ub, bf16_ub,
        slot, b_idx, R_dim, C_dim, vid,
        rb: int, cb: int, n_aiv: int, st_l2: int,
    ):
        r_blocks = T.alloc_var("int32", init=T.ceildiv(R_dim, rb))
        c_blocks = T.alloc_var("int32", init=T.ceildiv(C_dim, cb))
        total_tasks = T.alloc_var("int32", init=r_blocks * c_blocks)
        my_tasks = T.alloc_var("int32", init=T.max((total_tasks - vid + n_aiv - 1) // n_aiv, 0))

        pf_r0 = T.alloc_var("int32", init=0)
        pf_c0 = T.alloc_var("int32", init=0)
        pf_ar = T.alloc_var("int32", init=0)
        pf_ac = T.alloc_var("int32", init=0)
        cur_r0 = T.alloc_var("int32", init=0)
        cur_c0 = T.alloc_var("int32", init=0)
        cur_ar = T.alloc_var("int32", init=0)
        cur_ac = T.alloc_var("int32", init=0)

        if my_tasks > 0:
            t0 = vid
            pf_r0 = (t0 // c_blocks) * rb
            pf_c0 = (t0 % c_blocks) * cb
            pf_ar = T.min(R_dim - pf_r0, rb)
            pf_ac = T.min(C_dim - pf_c0, cb)
            T.ascend_wait_flag("V_MTE2", 0)
            T.copy(
                src_gm[b_idx, pf_r0 : pf_r0 + pf_ar, pf_c0 : pf_c0 + pf_ac],
                f32_ub[0, 0:pf_ar, 0:pf_ac],
                l2_cache_ctrl="NOTALLOC_DROP",
            )
            T.ascend_set_flag("MTE2_V", 0)

            for tk in range(my_tasks):
                u = tk % 2
                nu = (tk + 1) % 2
                cur_r0 = pf_r0
                cur_c0 = pf_c0
                cur_ar = pf_ar
                cur_ac = pf_ac

                if tk + 1 < my_tasks:
                    tn = vid + (tk + 1) * n_aiv
                    pf_r0 = (tn // c_blocks) * rb
                    pf_c0 = (tn % c_blocks) * cb
                    pf_ar = T.min(R_dim - pf_r0, rb)
                    pf_ac = T.min(C_dim - pf_c0, cb)
                    T.ascend_wait_flag("V_MTE2", nu)
                    T.copy(
                        src_gm[b_idx, pf_r0 : pf_r0 + pf_ar, pf_c0 : pf_c0 + pf_ac],
                        f32_ub[nu, 0:pf_ar, 0:pf_ac],
                        l2_cache_ctrl="NOTALLOC_DROP",
                    )
                    T.ascend_set_flag("MTE2_V", nu)

                T.ascend_wait_flag("MTE2_V", u)
                T.ascend_wait_flag("MTE3_V", u)
                _simdvf_cast_f32_bf16(f32_ub, bf16_ub, u, rb, cb)
                T.ascend_set_flag("V_MTE2", u)
                T.ascend_set_flag("V_MTE3", u)

                T.ascend_wait_flag("V_MTE3", u)
                T.copy(
                    bf16_ub[u, 0:cur_ar, 0:cur_ac],
                    ws_bf16[slot, cur_r0 : cur_r0 + cur_ar, cur_c0 : cur_c0 + cur_ac],
                    l2_cache_ctrl=st_l2,
                )
                T.ascend_set_flag("MTE3_V", u)

    @T.macro
    def _load_cube_ab_ring(
        A_gm, W_gm, ws_a, ws_w, a_l1, b_l1,
        s, bid, r_slot, m0, n0, k0, am, an, ak,
        a_f32: bool, w_f32: bool, ta: bool, tb: bool, dn2nz: bool, ld_l2: int,
    ):
        if a_f32:
            if ta:
                T.copy(ws_a[r_slot, k0 : k0 + ak, m0 : m0 + am], a_l1[s, 0:ak, 0:am], l2_cache_ctrl=ld_l2)
            else:
                T.copy(ws_a[r_slot, m0 : m0 + am, k0 : k0 + ak], a_l1[s, 0:am, 0:ak], l2_cache_ctrl=ld_l2)
        else:
            if ta:
                T.copy(A_gm[bid, k0 : k0 + ak, m0 : m0 + am], a_l1[s, 0:ak, 0:am], l2_cache_ctrl=ld_l2)
            else:
                T.copy(A_gm[bid, m0 : m0 + am, k0 : k0 + ak], a_l1[s, 0:am, 0:ak], l2_cache_ctrl=ld_l2)

        if w_f32:
            if tb:
                T.copy(ws_w[r_slot, n0 : n0 + an, k0 : k0 + ak], b_l1[s, 0:an, 0:ak], l2_cache_ctrl=ld_l2)
            elif dn2nz:
                T.copy(ws_w[r_slot, k0 : k0 + ak, n0 : n0 + an], b_l1[s, 0:an, 0:ak], transpose=True, l2_cache_ctrl=ld_l2)
            else:
                T.copy(ws_w[r_slot, k0 : k0 + ak, n0 : n0 + an], b_l1[s, 0:ak, 0:an], l2_cache_ctrl=ld_l2)
        else:
            if tb:
                T.copy(W_gm[bid, n0 : n0 + an, k0 : k0 + ak], b_l1[s, 0:an, 0:ak], l2_cache_ctrl=ld_l2)
            elif dn2nz:
                T.copy(W_gm[bid, k0 : k0 + ak, n0 : n0 + an], b_l1[s, 0:an, 0:ak], transpose=True, l2_cache_ctrl=ld_l2)
            else:
                T.copy(W_gm[bid, k0 : k0 + ak, n0 : n0 + an], b_l1[s, 0:ak, 0:an], l2_cache_ctrl=ld_l2)

    @T.macro
    def _load_l0_and_mmad(
        a_l1, b_l1, a_l0, b_l0, c_l0c,
        s, p, c_slot, k_off, am, an, ak_sub,
        ta: bool, b_nk: bool, is_clear, uf_mode, release_l1, use_uf: bool,
    ):
        T.ascend_wait_flag("M_MTE1", p)
        if ta:
            T.copy(a_l1[s, k_off : k_off + ak_sub, 0:am], a_l0[p, 0:am, 0:ak_sub], transpose=True)
        else:
            T.copy(a_l1[s, 0:am, k_off : k_off + ak_sub], a_l0[p, 0:am, 0:ak_sub])
        if b_nk:
            T.copy(b_l1[s, 0:an, k_off : k_off + ak_sub], b_l0[p, 0:an, 0:ak_sub])
        else:
            T.copy(b_l1[s, k_off : k_off + ak_sub, 0:an], b_l0[p, 0:an, 0:ak_sub], transpose=True)
        if release_l1:
            T.ascend_set_flag("MTE1_MTE2", s)
        _manual_sync("MTE1_M", p)
        if use_uf:
            T.gemm(
                a_l0[p, 0:am, 0:ak_sub],
                b_l0[p, 0:an, 0:ak_sub],
                c_l0c[c_slot, 0:am, 0:an],
                transpose_A=False,
                transpose_B=True,
                clear_accum=is_clear,
                unit_flag_ctrl=uf_mode,
            )
        else:
            T.gemm(
                a_l0[p, 0:am, 0:ak_sub],
                b_l0[p, 0:an, 0:ak_sub],
                c_l0c[c_slot, 0:am, 0:an],
                transpose_A=False,
                transpose_B=True,
                clear_accum=is_clear,
            )
        T.ascend_set_flag("M_MTE1", p)

    batch = T.dynamic("batch")
    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")
    sab = T.dynamic("sab")
    sam = T.dynamic("sam")
    swb = T.dynamic("swb")
    swk = T.dynamic("swk")

    a_is_f32 = (a_dtype == "float32")
    w_is_f32 = (w_dtype == "float32")
    need_aiv = a_is_f32 or w_is_f32
    c_slots = 1 if (BM * BN * 4 > 128 * 1024) else 2
    eff_unit_flag = use_unit_flag and (c_slots == 1)
    b_l1_is_nk = trans_b or use_dn2nz_b
    num_aiv = core_count * 2

    a_rows_l1 = BK if trans_a else BM
    a_cols_l1 = BM if trans_a else BK
    b_rows_l1 = BN if b_l1_is_nk else BK
    b_cols_l1 = BK if b_l1_is_nk else BN

    a_shape = (batch, K, M) if trans_a else (batch, M, K)
    w_shape = (batch, N, K) if trans_b else (batch, K, N)
    ws_a_shape = ((RING_R, K, M) if trans_a else (RING_R, M, K)) if a_is_f32 else (1, 1, 1)
    ws_w_shape = ((RING_R, N, K) if trans_b else (RING_R, K, N)) if w_is_f32 else (1, 1, 1)

    @T.prim_func
    def dense_pffn_gemm_v5(
        A: T.StridedTensor(a_shape, (sab, sam, 1), a_dtype),
        W: T.StridedTensor(w_shape, (swb, swk, 1), w_dtype),
        ws_a: T.Tensor(ws_a_shape, "bfloat16"),
        ws_w: T.Tensor(ws_w_shape, "bfloat16"),
        O: T.Tensor((batch, M, N), o_dtype),
    ):
        with T.Kernel(core_count) as cid:
            a_l1 = T.alloc_l1((L1_SLOTS, a_rows_l1, a_cols_l1), "bfloat16")
            b_l1 = T.alloc_l1((L1_SLOTS, b_rows_l1, b_cols_l1), "bfloat16")
            a_l0 = T.alloc_l0a((2, BM, BK_L0), "bfloat16")
            b_l0 = T.alloc_l0b((2, BN, BK_L0), "bfloat16")
            c_l0c = T.alloc_l0c((c_slots, BM, BN), "float32")

            if need_aiv:
                f32_ub = T.alloc_shared((2, RB, CB), "float32")
                bf16_ub = T.alloc_shared((2, RB, CB), "bfloat16")

            m_tiles = T.alloc_var("int32", init=T.ceildiv(M, BM))
            n_tiles = T.alloc_var("int32", init=T.ceildiv(N, BN))
            k_tiles = T.alloc_var("int32", init=T.ceildiv(K, BK))
            tiles_per_b = T.alloc_var("int32", init=m_tiles * n_tiles)
            width = T.alloc_var("int32", init=GROUP_M * n_tiles)
            total_tiles = T.alloc_var("int32", init=batch * tiles_per_b)

            with T.Cube():
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 1)
                if not eff_unit_flag:
                    for cs in T.unroll(c_slots):
                        T.ascend_set_flag("FIX_M", cs)
                for s in T.unroll(L1_SLOTS):
                    T.ascend_set_flag("MTE1_MTE2", s)

                my_tiles = T.alloc_var("int32", init=T.max((total_tiles - cid + core_count - 1) // core_count, 0))
                my_steps = T.alloc_var("int32", init=my_tiles * k_tiles)

                next_wait_b = T.alloc_var("int32", init=0)
                next_rel_b = T.alloc_var("int32", init=0)

                bid_v = T.alloc_var("int32", init=0)
                m0_v = T.alloc_var("int32", init=0)
                n0_v = T.alloc_var("int32", init=0)
                am_v = T.alloc_var("int32", init=0)
                an_v = T.alloc_var("int32", init=0)
                ak_v = T.alloc_var("int32", init=0)

                pf_ti = T.alloc_var("int32", init=0)
                pf_kt = T.alloc_var("int32", init=0)
                pf_bid = T.alloc_var("int32", init=0)
                pf_m0 = T.alloc_var("int32", init=0)
                pf_n0 = T.alloc_var("int32", init=0)
                pf_am = T.alloc_var("int32", init=0)
                pf_an = T.alloc_var("int32", init=0)
                pf_ak = T.alloc_var("int32", init=0)

                if my_steps > 0:
                    t_idx0 = cid
                    pf_bid = t_idx0 // tiles_per_b
                    if need_aiv:
                        while next_rel_b < pf_bid:
                            if next_wait_b <= next_rel_b:
                                _pair_wait("PIPE_MTE2", next_wait_b % RING_R)
                                next_wait_b = next_wait_b + 1
                            _pair_set("PIPE_MTE2", 4 + (next_rel_b % RING_R))
                            next_rel_b = next_rel_b + 1
                        if next_wait_b <= pf_bid:
                            _pair_wait("PIPE_MTE2", next_wait_b % RING_R)
                            next_wait_b = next_wait_b + 1

                    in_b0_raw = t_idx0 - pf_bid * tiles_per_b
                    if use_batch_rot:
                        sh0 = (pf_bid * GROUP_M) % tiles_per_b
                        in_b0_s = in_b0_raw + sh0
                        in_b0 = T.if_then_else(in_b0_s >= tiles_per_b, in_b0_s - tiles_per_b, in_b0_s)
                    else:
                        in_b0 = in_b0_raw
                    gid0 = in_b0 // width
                    fm0 = gid0 * GROUP_M
                    gsz0 = T.min(m_tiles - fm0, GROUP_M)
                    w0 = in_b0 - gid0 * width
                    pm0 = fm0 + (w0 % gsz0)
                    pn0_r = w0 // gsz0
                    pn0 = T.if_then_else((gid0 & 1) == 1, n_tiles - 1 - pn0_r, pn0_r)
                    pf_m0 = pm0 * BM
                    pf_n0 = pn0 * BN
                    pf_am = T.min(M - pf_m0, BM)
                    pf_an = T.min(N - pf_n0, BN)
                    pf_ak = T.min(K, BK)

                    b0 = pf_bid
                    rs0 = b0 % RING_R
                    m0_0 = pf_m0
                    n0_0 = pf_n0
                    am0 = pf_am
                    an0 = pf_an
                    ak0 = pf_ak
                    T.assume(b0 >= 0 and b0 < batch and rs0 >= 0 and rs0 < RING_R)
                    T.assume(m0_0 >= 0 and am0 > 0 and am0 <= BM and m0_0 + am0 <= M)
                    T.assume(n0_0 >= 0 and an0 > 0 and an0 <= BN and n0_0 + an0 <= N)
                    T.assume(ak0 > 0 and ak0 <= BK and ak0 <= K)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    _load_cube_ab_ring(
                        A, W, ws_a, ws_w, a_l1, b_l1,
                        0, b0, rs0, m0_0, n0_0, 0, am0, an0, ak0,
                        a_is_f32, w_is_f32, trans_a, trans_b, use_dn2nz_b, ws_load_l2,
                    )
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    pf_kt = 1
                    if pf_kt == k_tiles:
                        pf_kt = 0
                        pf_ti = 1

                    for ti in range(my_tiles):
                        t_idx = cid + ti * core_count
                        bid_v = t_idx // tiles_per_b
                        in_b_raw = t_idx - bid_v * tiles_per_b
                        if use_batch_rot:
                            sh = (bid_v * GROUP_M) % tiles_per_b
                            in_b_s = in_b_raw + sh
                            in_b = T.if_then_else(in_b_s >= tiles_per_b, in_b_s - tiles_per_b, in_b_s)
                        else:
                            in_b = in_b_raw
                        gid = in_b // width
                        first_m = gid * GROUP_M
                        group_sz = T.min(m_tiles - first_m, GROUP_M)
                        within = in_b - gid * width
                        pid_m = first_m + (within % group_sz)
                        pid_n_raw = within // group_sz
                        pid_n = T.if_then_else((gid & 1) == 1, n_tiles - 1 - pid_n_raw, pid_n_raw)
                        m0_v = pid_m * BM
                        n0_v = pid_n * BN
                        am_v = T.min(M - m0_v, BM)
                        an_v = T.min(N - n0_v, BN)
                        bid_r = bid_v
                        m0_r = m0_v
                        n0_r = n0_v
                        am = am_v
                        an = an_v
                        T.assume(bid_r >= 0 and bid_r < batch)
                        T.assume(m0_r >= 0 and am > 0 and am <= BM and m0_r + am <= M)
                        T.assume(n0_r >= 0 and an > 0 and an <= BN and n0_r + an <= N)
                        c_slot = 0 if c_slots == 1 else (ti % 2)
                        if not eff_unit_flag:
                            T.ascend_wait_flag("FIX_M", c_slot)

                        for kt in range(k_tiles):
                            st_cur = ti * k_tiles + kt
                            s = st_cur % L1_SLOTS
                            ak_v = T.min(K - kt * BK, BK)
                            ak = ak_v
                            T.assume(ak > 0 and ak <= BK and kt * BK + ak <= K)

                            if st_cur + 1 < my_steps:
                                ns = (st_cur + 1) % L1_SLOTS
                                if pf_kt == 0:
                                    nt_idx = cid + pf_ti * core_count
                                    new_bid = nt_idx // tiles_per_b
                                    if need_aiv:
                                        while next_rel_b < new_bid:
                                            if next_wait_b <= next_rel_b:
                                                _pair_wait("PIPE_MTE2", next_wait_b % RING_R)
                                                next_wait_b = next_wait_b + 1
                                            _pair_set("PIPE_MTE2", 4 + (next_rel_b % RING_R))
                                            next_rel_b = next_rel_b + 1
                                        if next_wait_b <= new_bid:
                                            _pair_wait("PIPE_MTE2", next_wait_b % RING_R)
                                            next_wait_b = next_wait_b + 1
                                    pf_bid = new_bid
                                    nin_b_raw = nt_idx - pf_bid * tiles_per_b
                                    if use_batch_rot:
                                        nsh = (pf_bid * GROUP_M) % tiles_per_b
                                        nin_b_s = nin_b_raw + nsh
                                        nin_b = T.if_then_else(nin_b_s >= tiles_per_b, nin_b_s - tiles_per_b, nin_b_s)
                                    else:
                                        nin_b = nin_b_raw
                                    ngid = nin_b // width
                                    nfm = ngid * GROUP_M
                                    ngsz = T.min(m_tiles - nfm, GROUP_M)
                                    nw = nin_b - ngid * width
                                    npm = nfm + (nw % ngsz)
                                    npn_r = nw // ngsz
                                    npn = T.if_then_else((ngid & 1) == 1, n_tiles - 1 - npn_r, npn_r)
                                    pf_m0 = npm * BM
                                    pf_n0 = npn * BN
                                    pf_am = T.min(M - pf_m0, BM)
                                    pf_an = T.min(N - pf_n0, BN)
                                pf_k0 = pf_kt * BK
                                pf_ak = T.min(K - pf_k0, BK)
                                pbid = pf_bid
                                prs = pbid % RING_R
                                pm0_r = pf_m0
                                pn0_r = pf_n0
                                pk0_r = pf_k0
                                pam = pf_am
                                pan = pf_an
                                pak = pf_ak
                                T.assume(pbid >= 0 and pbid < batch and prs >= 0 and prs < RING_R)
                                T.assume(pm0_r >= 0 and pam > 0 and pam <= BM and pm0_r + pam <= M)
                                T.assume(pn0_r >= 0 and pan > 0 and pan <= BN and pn0_r + pan <= N)
                                T.assume(pk0_r >= 0 and pak > 0 and pak <= BK and pk0_r + pak <= K)
                                T.ascend_wait_flag("MTE1_MTE2", ns)
                                _load_cube_ab_ring(
                                    A, W, ws_a, ws_w, a_l1, b_l1,
                                    ns, pbid, prs, pm0_r, pn0_r, pk0_r, pam, pan, pak,
                                    a_is_f32, w_is_f32, trans_a, trans_b, use_dn2nz_b, ws_load_l2,
                                )
                                T.ascend_set_flag("MTE2_MTE1", ns)
                                pf_kt = pf_kt + 1
                                if pf_kt == k_tiles:
                                    pf_kt = 0
                                    pf_ti = pf_ti + 1

                            T.ascend_wait_flag("MTE2_MTE1", s)
                            is_last_kt = (kt == k_tiles - 1)
                            if sub_k_steps == 1:
                                p = st_cur % 2
                                uf = T.if_then_else(is_last_kt, 3, T.if_then_else(kt == 0, 2, 0))
                                _load_l0_and_mmad(
                                    a_l1, b_l1, a_l0, b_l0, c_l0c,
                                    s, p, c_slot, 0, am, an, ak,
                                    trans_a, b_l1_is_nk, kt == 0, uf, True, eff_unit_flag,
                                )
                            else:
                                ak_sub0 = T.min(ak, BK_L0)
                                T.assume(ak_sub0 > 0 and ak_sub0 <= BK_L0)
                                uf0 = T.if_then_else(is_last_kt and (ak <= BK_L0), 3, T.if_then_else(kt == 0, 2, 0))
                                _load_l0_and_mmad(
                                    a_l1, b_l1, a_l0, b_l0, c_l0c,
                                    s, 0, c_slot, 0, am, an, ak_sub0,
                                    trans_a, b_l1_is_nk, kt == 0, uf0, ak <= BK_L0, eff_unit_flag,
                                )
                                if ak > BK_L0:
                                    ak_sub1 = ak - BK_L0
                                    T.assume(ak_sub1 > 0 and ak_sub1 <= BK_L0)
                                    uf1 = T.if_then_else(is_last_kt, 3, 0)
                                    _load_l0_and_mmad(
                                        a_l1, b_l1, a_l0, b_l0, c_l0c,
                                        s, 1, c_slot, BK_L0, am, an, ak_sub1,
                                        trans_a, b_l1_is_nk, False, uf1, True, eff_unit_flag,
                                    )

                        if eff_unit_flag:
                            T.copy(
                                c_l0c[0, 0:am, 0:an],
                                O[bid_r, m0_r : m0_r + am, n0_r : n0_r + an],
                                unit_flag_ctrl=3,
                                l2_cache_ctrl="NOTALLOC_PW",
                            )
                        else:
                            _manual_sync("M_FIX", c_slot)
                            T.copy(
                                c_l0c[c_slot, 0:am, 0:an],
                                O[bid_r, m0_r : m0_r + am, n0_r : n0_r + an],
                                l2_cache_ctrl="NOTALLOC_PW",
                            )
                            T.ascend_set_flag("FIX_M", c_slot)

                if need_aiv:
                    while next_rel_b < batch:
                        if next_wait_b <= next_rel_b:
                            _pair_wait("PIPE_MTE2", next_wait_b % RING_R)
                            next_wait_b = next_wait_b + 1
                        _pair_set("PIPE_MTE2", 4 + (next_rel_b % RING_R))
                        next_rel_b = next_rel_b + 1

                if eff_unit_flag:
                    T.ascend_pipe_barrier("PIPE_ALL")
                else:
                    for cs in T.unroll(c_slots):
                        T.ascend_wait_flag("FIX_M", cs)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("M_MTE1", 1)
                for s in T.unroll(L1_SLOTS):
                    T.ascend_wait_flag("MTE1_MTE2", s)

            with T.Vector() as sid:
                if need_aiv:
                    vid = cid * 2 + sid
                    T.ascend_set_flag("V_MTE2", 0)
                    T.ascend_set_flag("V_MTE2", 1)
                    T.ascend_set_flag("MTE3_V", 0)
                    T.ascend_set_flag("MTE3_V", 1)

                    for b in range(batch):
                        r_slot = b % RING_R
                        if b >= RING_R:
                            T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 4 + r_slot)
                            T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 10 + (b % 2))
                            T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 10 + (b % 2))

                        if a_is_f32:
                            R_a = K if trans_a else M
                            C_a = M if trans_a else K
                            _aiv_cast_single_batch(A, ws_a, f32_ub, bf16_ub, r_slot, b, R_a, C_a, vid, RB, CB, num_aiv, ws_store_l2)
                        if w_is_f32:
                            R_w = N if trans_b else K
                            C_w = K if trans_b else N
                            _aiv_cast_single_batch(W, ws_w, f32_ub, bf16_ub, r_slot, b, R_w, C_w, vid, RB, CB, num_aiv, ws_store_l2)

                        T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 8 + (b % 2))
                        T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 8 + (b % 2))
                        T.ascend_cross_core_set_flag(4, "PIPE_MTE3", r_slot)

                    drain_start = T.max(0, batch - RING_R)
                    for k in range(drain_start, batch):
                        T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 4 + (k % RING_R))

                    T.ascend_wait_flag("V_MTE2", 0)
                    T.ascend_wait_flag("V_MTE2", 1)
                    T.ascend_wait_flag("MTE3_V", 0)
                    T.ascend_wait_flag("MTE3_V", 1)

    return dense_pffn_gemm_v5


def build_v4_l2_pingpong_bmm(**kwargs):
    """Compile (and cache) the on-board verified V4 L2-Ping-Pong Super-Batch CV kernel."""
    import tilelang

    key = ("v4",) + tuple(sorted(kwargs.items()))
    if key not in _KERNEL_CACHE:
        fn = _make_v4_prim_func(**kwargs)
        _KERNEL_CACHE[key] = tilelang.compile(
            fn,
            target="ascend",
            compile_flags=_CXX_FLAGS,
            pass_configs={
                "tl.enable_auto_schedule": False,
                "tl.disable_shared_memory_reuse": True,
            },
        )
    return _KERNEL_CACHE[key]


def build_v5_ring_bmm(**kwargs):
    """Compile (and cache) the V5.1 Pure-Hardware Zero-AIC-Barrier Persistent Ring kernel."""
    import tilelang

    key = ("v5_ring",) + tuple(sorted(kwargs.items()))
    if key not in _KERNEL_CACHE:
        fn = _make_v5_prim_func(**kwargs)
        _KERNEL_CACHE[key] = tilelang.compile(
            fn,
            target="ascend",
            compile_flags=_CXX_FLAGS,
            pass_configs={
                "tl.enable_auto_schedule": False,
                "tl.disable_shared_memory_reuse": True,
            },
        )
    return _KERNEL_CACHE[key]


def _normalize_operand(t: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Ensure trailing dim has stride 1, either natively (trans=False) or via transpose(1,2) (trans=True)."""
    if t.stride(2) == 1:
        return t, False
    if t.stride(1) == 1:
        return t.transpose(1, 2), True
    return t.contiguous(), False


def _select_v4_gb(batch: int, M: int, N: int, K: int, a_is_f32: bool, w_is_f32: bool) -> int:
    """Pick super-batch size G_B keeping 2-slot ping-pong workspace within ~75 MiB of 128 MiB L2."""
    bytes_per_b = 0
    if a_is_f32:
        bytes_per_b += M * K * 2
    if w_is_f32:
        bytes_per_b += K * N * 2
    if bytes_per_b == 0:
        return min(batch, 8)
    max_gb = max(1, (80 * 1024 * 1024) // (2 * bytes_per_b))
    for cand in (8, 4, 2, 1):
        if cand <= max_gb and cand <= batch:
            return cand
    return 1


def _select_v5_ring_r(batch: int, M: int, N: int, K: int, a_is_f32: bool, w_is_f32: bool) -> int:
    """Pick ring depth R in {2, 3, 4} keeping R-slot workspace within ~80 MiB of 128 MiB L2."""
    bytes_per_b = (M * K * 2 if a_is_f32 else 0) + (K * N * 2 if w_is_f32 else 0)
    if bytes_per_b == 0:
        return 1
    max_r = max(1, min(4, (80 * 1024 * 1024) // bytes_per_b))
    return max(1, min(max_r, batch))


def dense_pffn_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    use_bf16: bool = True,
    core_count: int = 32,
    backend: str = "v4",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused-cast batched GEMM: a [batch, M, K] @ w [batch, K, N] -> [batch, M, N]."""
    assert a.ndim == 3 and w.ndim == 3, f"Expected 3D tensors, got {a.shape} and {w.shape}"
    batch, M, K = a.shape
    w_batch, w_K, N = w.shape
    assert batch == w_batch and K == w_K, f"Shape mismatch: a={a.shape}, w={w.shape}"

    if out is None:
        out = torch.empty((batch, M, N), device=a.device, dtype=out_dtype)

    if batch == 0 or M == 0 or N == 0 or K == 0:
        if K == 0:
            out.zero_()
        return out

    # Fallback for non-NPU, TF32/FP32 mode, unsupported dtypes, or non-16-aligned dimensions
    supported_in = {torch.float32, torch.bfloat16}
    supported_out = {torch.float32, torch.bfloat16}
    if (
        (not use_bf16)
        or a.device.type != "npu"
        or a.dtype not in supported_in
        or w.dtype not in supported_in
        or out_dtype not in supported_out
        or (M % 16 != 0)
        or (N % 16 != 0)
        or (K % 16 != 0)
    ):
        a_c = a.to(torch.bfloat16).contiguous() if use_bf16 else a.float().contiguous()
        w_c = w.to(torch.bfloat16).contiguous() if use_bf16 else w.float().contiguous()
        if use_bf16 and out_dtype == torch.bfloat16:
            res = torch.bmm(a_c, w_c)
        else:
            res = torch.bmm(a_c.float(), w_c.float()).to(out_dtype)
        out.copy_(res)
        return out

    a_view, trans_a = _normalize_operand(a)
    w_view, trans_b = _normalize_operand(w)
    a_dt = "float32" if a_view.dtype == torch.float32 else "bfloat16"
    w_dt = "float32" if w_view.dtype == torch.float32 else "bfloat16"
    o_dt = "bfloat16" if out_dtype == torch.bfloat16 else "float32"
    a_is_f32 = (a_dt == "float32")
    w_is_f32 = (w_dt == "float32")

    if backend == "v4":
        gb = _select_v4_gb(batch, M, N, K, a_is_f32, w_is_f32)
        # On c6-like small-tile-count shapes (M*N <= 1024*1232 with huge K), 128x256 slightly edges 256x256 in V4
        bm = 128 if (M <= 1024 and N <= 1280 and K >= 4096) else 256
        kernel = build_v4_l2_pingpong_bmm(
            a_dtype=a_dt,
            w_dtype=w_dt,
            o_dtype=o_dt,
            trans_a=trans_a,
            trans_b=trans_b,
            BM=bm,
            BN=256,
            BK=128,
            BK_L0=64,
            L1_SLOTS=2,
            GROUP_M=4,
            G_B=gb,
            ws_store_l2=1,
            ws_load_l2=1,
            core_count=core_count,
        )
        ws_a_shape = ((2, gb, K, M) if trans_a else (2, gb, M, K)) if a_is_f32 else (1, 1, 1, 1)
        ws_w_shape = ((2, gb, N, K) if trans_b else (2, gb, K, N)) if w_is_f32 else (1, 1, 1, 1)
        ws_a = _get_ws(a.device, ws_a_shape, "v4_a")
        ws_w = _get_ws(a.device, ws_w_shape, "v4_w")
        kernel(a_view, w_view, ws_a, ws_w, out)
        return out

    if backend == "v5_ring":
        ring_r = _select_v5_ring_r(batch, M, N, K, a_is_f32, w_is_f32)
        kernel = build_v5_ring_bmm(
            a_dtype=a_dt,
            w_dtype=w_dt,
            o_dtype=o_dt,
            trans_a=trans_a,
            trans_b=trans_b,
            BM=256,
            BN=256,
            BK=128,
            BK_L0=64,
            L1_SLOTS=3,
            GROUP_M=4,
            RING_R=ring_r,
            ws_store_l2=1,
            ws_load_l2=1,
            use_unit_flag=True,
            use_dn2nz_b=False,
            use_batch_rot=True,
            core_count=core_count,
        )
        ws_a_shape = ((ring_r, K, M) if trans_a else (ring_r, M, K)) if a_is_f32 else (1, 1, 1)
        ws_w_shape = ((ring_r, N, K) if trans_b else (ring_r, K, N)) if w_is_f32 else (1, 1, 1)
        ws_a = _get_ws(a.device, ws_a_shape, "v5_a")
        ws_w = _get_ws(a.device, ws_w_shape, "v5_w")
        kernel(a_view, w_view, ws_a, ws_w, out)
        return out

    raise ValueError(f"Unknown backend={backend!r}; expected 'v4' or 'v5_ring'")


def _fc_bmm(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    use_bf16: bool = True,
    core_count: int = 32,
    backend: str = "v4",
) -> torch.Tensor:
    """Drop-in replacement for `_fc_bmm` in `cases/origin_gemm.py`."""
    return dense_pffn_gemm(
        a,
        w,
        out_dtype=out_dtype,
        use_bf16=use_bf16,
        core_count=core_count,
        backend=backend,
    )
