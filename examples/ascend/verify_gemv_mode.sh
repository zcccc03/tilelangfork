#!/usr/bin/env bash
# Run inside a CANN simulation environment with this checkout's TileLang built.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=$(realpath -m "${1:?Usage: bash verify_gemv_mode.sh NEW_OUTPUT_DIR}")
mkdir "$out"
cd "$out"

cann=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}
set +u
source "$cann/set_env.sh"
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$repo:$cann/python/site-packages:${PYTHONPATH:-}"
export TVM_LIBRARY_PATH="$repo/build/lib"
export TL_TEMPLATE_PATH="$repo/src"

python -P - "$repo" <<'PY'
import sys
from pathlib import Path
import tilelang
from tilelang.libinfo import find_lib_path

repo = Path(sys.argv[1]).resolve()
assert Path(tilelang.__file__).resolve().parent == repo / "tilelang"
assert Path(find_lib_path("tilelang")).resolve().parent == (repo / "build/lib").resolve()
PY
python -P "$here/example_gemv.py" --n 32 --k 256 --out-dir "$out"
python - "$here/gemv_l1_sim_launcher.inc" <<'PY'
import re
import sys
from pathlib import Path

source = Path("03_kernel.asc").read_text()
assert "ascend_gemm_l1<1, 256, 32, 256," in source
signature = re.search(r"void main_kernel\(([^)]*)\)", source)
assert signature
order = [arg.split()[-1].lstrip("*") for arg in signature[1].split(",")]
assert sorted(order) == ["W", "X", "Y"], order
ptr = {"X": "(half*)dx", "W": "(half*)dw", "Y": "(float*)dy"}
args = ", ".join(ptr[name] for name in order)
Path("verify.asc").write_text(
    source + f"\n#define VERIFY_K 256\n#define VERIFY_ARGS {args}\n" + Path(sys.argv[1]).read_text()
)
PY
bisheng verify.asc -o verify --npu-arch=dav-3510 --run-mode=sim -O2 -std=c++20 \
    -mllvm -cce-aicore-dcpreload-args=false "-I$TL_TEMPLATE_PATH" \
    -lruntime_camodel -lnpu_drv_camodel -lascendcl -lprofapi -lunified_dlog \
    -lmmpa -lascend_dump -lc_sec -lerror_manager -lnpu_drv -lm 2>&1 | tee build.log
patchelf --set-rpath "$cann/tools/simulator/dav_3510/camodel" verify
npusim record -s Ascend950 -o "$out/simulation" -g -n all "$out/verify" 2>&1 | tee simulation.log
grep -qF 'GEMV NUMERICAL PASS: N=32 K=256 max_error=0' simulation.log
python - <<'PY'
import json
from pathlib import Path

matches = []
for trace in Path("simulation").rglob("trace_core*.json"):
    for event in json.loads(trace.read_text())["traceEvents"]:
        args = event.get("args", {})
        if (event["name"] == "MMAD" and event.get("tid", "").startswith("CUBE")
            and tuple(args.get(f"matrix_dim_{dim}") for dim in "mnk") == (1, 32, 256)
            and args.get("gemv_mode_enable") == 1):
            matches.append({"trace": str(trace), "event": event})
assert len(matches) == 1, f"Expected one Cube MMAD with GEMV enabled; found {len(matches)}"
Path("gemv-mode-evidence.json").write_text(json.dumps(matches[0], indent=2) + "\n")
print("GEMV MODE PASS: Cube MMAD, M=1 N=32 K=256, gemv_mode_enable=1")
PY
