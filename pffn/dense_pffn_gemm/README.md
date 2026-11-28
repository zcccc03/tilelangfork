# DensePFFN GEMM for Ascend 950DT

This package imports the supplied `dense_pffn_gemm` implementation without
changing its five source/design files. It provides the six DensePFFN batched
GEMMs with FP32/BF16 inputs, BF16 operand rounding and FP32 accumulation.

- `v4` (default): L2 ping-pong super-batches with overlapping AIV conversion
  and Cube computation. The supplied design document reports six-callsite
  on-board validation.
- `v5_ring` (experimental): persistent cross-batch ring scheduling. The
  supplied package marks full fused CV on-board verification as pending;
  the reported approximately 90% MFU belongs to pure BF16 Cube experiments.

## Environment

Use a Linux Ascend environment with compatible CANN, PyTorch, `torch_npu`,
and a built TileLang Ascend backend exposing `tilelang.ascend.language`.
Load that environment's CANN `set_env.sh` before running. The test entry
limits the device to 32 Cube cores and 64 Vector cores.

The commands below assume the repository root is the current directory and
the intended built TileLang backend is importable in the Python environment.

## Run V4 correctness checks

```bash
python pffn/dense_pffn_gemm/test_dense_pffn_gemm.py --device npu --backend v4
```

This checks the six original GEMM call sites, edge cases, and a reduced
DensePFFN forward/backward chain. For smaller six-callsite inputs:

```bash
python pffn/dense_pffn_gemm/test_dense_pffn_gemm.py --device npu --backend v4 --small
```

`--small` affects the six-callsite correctness inputs. It does not change
the original shapes used by `--bench`.

## Run MFU benchmark

```bash
python pffn/dense_pffn_gemm/test_dense_pffn_gemm.py --device npu --backend v4 --bench --iters 15
```

The benchmark runs correctness checks first, then reports per-callsite and
combined cold/warm latency and MFU. The original script performs five warm-up
calls per case. Cold samples zero a separate 1 GiB buffer before timing;
outputs and workspace allocations are reused. Device conversions are inside
the timed operator call. The nominal BF16 Cube denominator is 432 TFLOPS at
1650 MHz for 32 AICs; check actual hardware and frequency before comparing.

## Optional V5 experiment

```bash
python pffn/dense_pffn_gemm/test_dense_pffn_gemm.py --device npu --backend v5_ring --small
```

This is an experimental path, not a declaration of full V5 on-board success.

See `design.md` for the supplied performance history. Those measurements
were not rerun as part of this source import. The earlier standalone
`pffn/gemm_tilelang_ascend.py` implementation is separate from this package.
