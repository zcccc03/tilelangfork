"""Optional CAModel numerical execution; temporary files are always removed."""

import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile

import torch


def simulate(case, program, source, inputs, expected, initial, evidence):
    from tilelang.contrib.bisheng import get_npu_arch
    from tilelang import __file__ as tilelang_file
    from testing.ascend.gemm_features._support import compare

    output = initial if initial is not None else torch.full_like(expected, 0x12345678 if expected.dtype == torch.int32 else -12345)
    data = {program.buffer_map[param].name: value for param, value in zip(program.params, [*inputs, output], strict=True)}
    output_name = program.buffer_map[program.params[-1]].name
    signature = re.search(r'extern "C" __global__ __cube__ void (\w+)\(([^)]*)\)', source)
    assert signature, "Cube entry not found"
    args = signature[2].split(",")
    names = [arg.split()[-1].lstrip("*") for arg in args]
    aliases = {name: name if name in data else name.removesuffix("_data") for name in names}
    assert set(aliases.values()) == set(data)
    output_name = next(name for name, original in aliases.items() if original == output_name)
    data = {name: data[original] for name, original in aliases.items()}
    types = {name: arg.split("*")[0].strip().split()[-1] for name, arg in zip(names, args)}
    host = Path(__file__).with_name("camodel_launcher.inc").read_text(encoding="utf-8")
    allocs = "\n".join(
        f"void* d{name}; const size_t size{name} = {data[name].numel() * data[name].element_size()};"
        f"CK(aclrtMalloc(&d{name}, size{name}, ACL_MEM_MALLOC_HUGE_FIRST));"
        f'if (!load("{name}.bin", d{name}, size{name})) return 4;'
        for name in names
    )
    host = host.replace("@ALLOCS@", allocs).replace("@KERNEL@", signature[1]).replace("@OUTPUT@", output_name)
    host = host.replace("@REPEATS@", str(case.repeats)).replace("@ARGS@", ", ".join(f"({types[name]}*)d{name}" for name in names))
    host = host.replace("@FREE@", "\n".join(f"CK(aclrtFree(d{name}));" for name in names))
    assert not re.search(r"@[A-Z]+@", host)
    with tempfile.TemporaryDirectory(prefix="gemm_camodel_") as temp:
        folder = Path(temp)
        for name, value in data.items():
            (folder / (name + ".bin")).write_bytes(value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes())
        (folder / "verify.asc").write_text(source + "\n" + host, encoding="utf-8")
        evidence["camodel_launcher.asc"] = source + "\n" + host

        def run(command, log):
            with subprocess.Popen(
                command, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=os.name != "nt"
            ) as process:
                try:
                    stdout, stderr = process.communicate(timeout=180)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        stdout, stderr = process.communicate(timeout=5)
                    except subprocess.TimeoutExpired as partial:
                        # An uninterruptible model descendant can keep a pipe open.
                        stdout = partial.stdout or ""
                        stderr = partial.stderr or ""
                        if isinstance(stdout, bytes):
                            stdout = stdout.decode(errors="replace")
                        if isinstance(stderr, bytes):
                            stderr = stderr.decode(errors="replace")
                        process.stdout.close()
                        process.stderr.close()
                    evidence[log] = stdout + stderr
                    raise TimeoutError(f"{log}: 超过180秒，已终止本次仿真/编译进程组") from None
                evidence[log] = stdout + stderr
                if process.returncode:
                    raise RuntimeError(f"{log}: exit {process.returncode}\n" + evidence[log])

        template = str(Path(tilelang_file).resolve().parents[1] / "src")
        run(
            [
                "bisheng",
                "verify.asc",
                "-o",
                "verify",
                "--npu-arch=" + get_npu_arch(),
                "--run-mode=sim",
                "-O2",
                "-std=c++20",
                "-mllvm",
                "-cce-aicore-dcpreload-args=false",
                "-I" + template,
                "-lruntime_camodel",
                "-lnpu_drv_camodel",
                "-lascendcl",
                "-lprofapi",
                "-lunified_dlog",
                "-lmmpa",
                "-lascend_dump",
                "-lc_sec",
                "-lerror_manager",
                "-lnpu_drv",
                "-lm",
            ],
            "camodel_build.log",
        )
        cann = Path(os.environ["ASCEND_HOME_PATH"])
        run(["patchelf", "--set-rpath", str(cann / "tools/simulator/dav_3510/camodel"), "verify"], "camodel_elf.log")
        try:
            run(["npusim", "record", "-s", "Ascend950", "-n", "0", "-o", str(folder / "trace"), str(folder / "verify")], "simulation.log")
        except Exception:
            # A later launch can hang after earlier launches produced correct data.
            for launch in range(1, case.repeats + 1):
                output_file = folder / f"actual_{launch}.bin"
                if output_file.is_file() and output_file.stat().st_size == expected.numel() * expected.element_size():
                    actual = torch.frombuffer(bytearray(output_file.read_bytes()), dtype=expected.dtype).reshape_as(expected)
                    evidence[f"output_{launch}.pt"] = {"actual": actual, "expected": expected}
            raise
        assert "Simulation SUCCESS" in evidence["simulation.log"]
        max_error = 0
        for launch in range(1, case.repeats + 1):
            actual = torch.frombuffer(bytearray((folder / f"actual_{launch}.bin").read_bytes()), dtype=expected.dtype).reshape_as(expected)
            max_error = max(max_error, compare(case, actual, expected, evidence, launch))
    return max_error
