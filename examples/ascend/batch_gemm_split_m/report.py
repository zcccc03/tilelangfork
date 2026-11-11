"""Calculate the simulator MFU from the all-core instruction trace."""

import argparse
import json
from collections import Counter
from pathlib import Path

PEAK_CUBE_TFLOPS = 378
BATCH = 28
DIM = 512
CORES = 28
TILE_MN = 256
TILE_K = 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="OPPROF_*/simulator/trace.json")
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    instructions = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    mmads = [event for event in instructions if event["name"] == "MMAD"]
    output_tiles = BATCH * (DIM // TILE_MN) ** 2
    expected_mmads = output_tiles * (DIM // TILE_K) * 2
    mmads_per_core = Counter(event["pid"] for event in mmads)
    if len(mmads) != expected_mmads or Counter(mmads_per_core.values()) != {expected_mmads // CORES: CORES}:
        raise ValueError("The trace does not contain the expected 28-core split-M workload")

    flops = 0
    for event in mmads:
        detail = json.loads(event["args"]["detail"])
        shape = (detail["m"], detail["n"], detail["k"])
        dtypes = tuple(detail[f"matrix_{operand}_dtype"] for operand in "abc")
        if shape != (128, TILE_MN, TILE_K) or dtypes != ("f16", "f16", "f32"):
            raise ValueError(f"Unexpected Mmad shape or dtypes: {shape}, {dtypes}")
        flops += 2 * shape[0] * shape[1] * shape[2]

    l0b_copies = sum(
        json.loads(event["args"]["detail"]).get("dst_mem") == "L0B"
        for event in instructions
        if event["name"] == "LOAD_2Dv2" and event["tid"] == "MTE1"
    )
    if l0b_copies * 2 != expected_mmads:
        raise ValueError("Expected one L0B load per two Mmad instructions")
    duration_us = max(event["ts"] + event["dur"] for event in instructions) - min(event["ts"] for event in instructions)
    tflops = flops / (duration_us * 1e6)
    print(f"{len(mmads)} Mmad, {l0b_copies} L0B loads, {flops:,} FLOP")
    print(f"All-core instruction span: {duration_us:.6f} us")
    print(f"{tflops:.4f} TFLOP/s, simulated MFU {tflops / PEAK_CUBE_TFLOPS:.2%} at {PEAK_CUBE_TFLOPS} TFLOP/s")


if __name__ == "__main__":
    main()
