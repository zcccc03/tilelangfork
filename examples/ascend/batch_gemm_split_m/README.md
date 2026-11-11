# Split-M FP16 batch GEMM on Ascend 950PR

This experiment uses 28 batches of `512 × 512 × 512` GEMM on 28 Cube cores. Inputs are FP16 and the accumulators and output are FP32. The kernel splits each `256 × 256` output tile along M into two `128 × 256` accumulators. Each `K=64` step loads two A slices and one B slice from L1. Two Mmad instructions share the B slice in L0B. L1 stages use `K=256`.

The instruction count follows from the tiling. There are `28 × (512/256) × (512/256) = 112` output tiles. Each tile has `512/64 = 8` K steps and issues two Mmad instructions per step, giving `112 × 8 × 2 = 1,792` Mmad instructions. Each of the 28 cores handles four output tiles and issues 64 Mmad instructions.

The `Ascend950PR_9589` simulator capture produced the following results with seed 0.

| Metric | Result |
| --- | ---: |
| CPU reference comparison | Exact, max absolute error 0 |
| Mmad instructions | 1,792 |
| L0B loads | 896 |
| All-core instruction span | 28.354547 µs |
| Throughput | 265.0789 TFLOP/s |
| Simulated MFU | 70.13% |

MFU uses `2 × 28 × 512³` FLOP and the 28-core FP16 Cube peak of 378 TFLOP/s in table 3-1. The elapsed time comes from the simulator's first instruction start and last instruction end across all cores. These figures describe one simulator capture. [Huawei Ascend 950 NPU Architecture White Paper](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)

Run the following from the repository root with the Ascend 950 simulator environment and Ascend TileLang build already active. Set the output directory to a fresh path for each capture.

```bash
example_dir="$PWD/examples/ascend/batch_gemm_split_m"
output_dir=/workspace/code/simulator/batch-split-m-b28-512
mkdir -p "$output_dir"
cd "$output_dir"
TILELANG_DISABLE_CACHE=1 msopprof simulator --soc-version=Ascend950PR_9589 \
  --kernel-name=main_kernel --launch-count=1 --timeout=60 --output="$PWD" \
  python -P -u "$example_dir/run.py" > run.log 2>&1
grep -F 'CORRECTNESS_PASS' run.log
python "$example_dir/report.py" OPPROF_*/simulator/trace.json
```
