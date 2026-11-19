"""Case protocol, independent execution, and explicitly requested failure output.

Successful cases never create report/log directories. Source/IR goes to stdout.
The optional failure directory is used only for FAIL, never for PASS or SKIP.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import traceback

import torch
import tilelang


@dataclass(frozen=True)
class Case:
    id: str
    group: str
    feature: str
    scenario: str
    constraint: str
    responsibility: str
    inputs: str
    expected_inputs: str
    outputs: str
    expected: str
    kind: str = "numerical"
    stage: str = "device_compile"
    diagnostic: str = ""
    source_pattern: str = ""
    atol: float = 0
    rtol: float = 0
    repeats: int = 1
    note: str = ""
    kernel_origin: str = ""
    manual_schedule: bool = False


@dataclass(frozen=True)
class Options:
    mode: str = "hardware"
    device: int = 0
    failure_dir: str | None = None


def options(parser):
    parser.add_argument(
        "--mode",
        choices=["hardware", "compile", "simulator"],
        default="hardware",
        help="compile: source/IR + device compilation; never reports numerical PASS",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--failure-dir", help="Only create <path>/<case-id> on a test FAIL")
    return parser


def make_options(args):
    return Options(args.mode, args.device, args.failure_dir)


def compile_device(source, evidence):
    from tilelang.contrib.bisheng import find_bisheng_path, get_npu_arch

    template = Path(tilelang.__file__).resolve().parents[1] / "src"
    with tempfile.TemporaryDirectory(prefix="gemm_compile_") as temp:
        path = Path(temp)
        (path / "kernel.asc").write_text(source, encoding="utf-8")
        command = [
            find_bisheng_path(),
            "-c",
            str(path / "kernel.asc"),
            "-o",
            str(path / "kernel.o"),
            "--npu-arch=" + get_npu_arch(),
            "--run-mode=sim",
            "-std=c++20",
            "-I" + str(template),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        evidence["bisheng.log"] = completed.stdout + completed.stderr
        if completed.returncode:
            raise RuntimeError(evidence["bisheng.log"])


def emit_ir(program, evidence, *, lower_ir=None, manual_schedule=False):
    evidence["input.tir"] = program.script()
    print("\n[第一部分：输入 IR]\n" + evidence["input.tir"], flush=True)
    if isinstance(program, tilelang.tvm.IRModule):
        if lower_ir:
            module = lower_ir(program)
        else:
            from tilelang.ascend import transform

            with tilelang.tvm.target.Target("ascend"):
                module = tilelang.transform.LowerAccessPtr()(transform.AscendLowerTileOp()(program))
        evidence["lowered.tir"] = module.script()
        print("[下沉 IR]\n" + evidence["lowered.tir"], flush=True)
        print("[源码] 本单元仅验收局部 IR；没有独立 GM 入口，不生成可运行 AscendC 源码。", flush=True)
        return module, None
    context = (
        tilelang.tvm.transform.PassContext(opt_level=3, config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False})
        if manual_schedule
        else nullcontext()
    )
    with context:
        artifact = tilelang.lower(program, target="ascend", enable_host_codegen=False, enable_device_compile=False)
    evidence["kernel.asc"] = artifact.kernel_source
    evidence["lowered.tir"] = artifact.device_mod.script()
    print("[AscendC 源码]\n" + evidence["kernel.asc"] + "\n[下沉 IR]\n" + evidence["lowered.tir"], flush=True)
    return artifact.device_mod, artifact.kernel_source


def check_signature(program, inputs, expected, initial):
    values = [*inputs, expected if initial is None else initial]
    assert len(values) == len(program.params)
    for param, value in zip(program.params, values, strict=True):
        buffer = program.buffer_map[param]
        logical_shape = tuple(int(dim) for dim in buffer.shape)
        packed_fp4 = str(buffer.dtype) == "float4_e2m1fn" and str(value.dtype) == "torch.float4_e2m1fn_x2"
        storage_shape = (*logical_shape[:-1], logical_shape[-1] // 2) if packed_fp4 else logical_shape
        assert storage_shape == tuple(value.shape), (buffer.name, value.shape, storage_shape)
        assert packed_fp4 or str(buffer.dtype) == str(value.dtype).removeprefix("torch."), (buffer.name, value.dtype)


def compare(case, actual, expected, evidence, launch):
    evidence[f"output_{launch}.pt"] = {"actual": actual, "expected": expected}
    torch.testing.assert_close(actual, expected, atol=case.atol, rtol=case.rtol)
    return (actual.double() - expected.double()).abs().max().item()


def hardware(case, program, inputs, expected, initial, config, evidence):
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return None
    if not torch.npu.is_available():
        return None
    torch.npu.set_device(config.device)
    # Always pass an explicitly initialized GM output. Repeats test the same kernel.
    configs = {tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False} if case.manual_schedule else None
    kernel = tilelang.compile(program, target="ascend", out_idx=None, pass_configs=configs)
    device_inputs = [value.npu() for value in inputs]
    max_error = 0
    for launch in range(1, case.repeats + 1):
        output = (
            initial if initial is not None else torch.full_like(expected, 0x12345678 if expected.dtype == torch.int32 else -12345)
        ).npu()
        kernel(*device_inputs, output)
        torch.npu.synchronize()
        max_error = max(max_error, compare(case, output.cpu(), expected, evidence, launch))
    return max_error


def save_failure(case, make_kernel, evidence, config, error):
    if not config.failure_dir:
        return None
    path = Path(config.failure_dir).expanduser().resolve() / case.id
    path.mkdir(parents=True, exist_ok=True)
    (path / "case.json").write_text(json.dumps(asdict(case), ensure_ascii=False, indent=2), encoding="utf-8")
    source_file = inspect.getsourcefile(make_kernel)
    if source_file:
        shutil.copyfile(source_file, path / "test_case.py")
        repo = Path(__file__).resolve().parents[3]
        relative = Path(source_file).resolve().relative_to(repo).as_posix()
        command = f"python {relative} --mode {config.mode} --device {config.device}"
        (path / "run.json").write_text(
            json.dumps({"options": asdict(config), "command": command}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (path / "REPRODUCE.md").write_text(
            "# 单用例复现\n\n在原仓库根目录执行：\n\n```bash\n" + command + "\n```\n\n"
            "种子写死在测试文件中。随附实际生成的源码/IR、相关构造源码及可获得的CPU输入和实际输出；非法输入不启动设备数值执行。\n",
            encoding="utf-8",
        )
    for helper in ("_support.py", "_simulator.py", "camodel_launcher.inc"):
        shutil.copyfile(Path(__file__).with_name(helper), path / helper)
    for name, value in evidence.items():
        if name.endswith(".pt"):
            torch.save(value, path / name)
        else:
            (path / name).write_text(str(value), encoding="utf-8")
    (path / "failure.log").write_text(error, encoding="utf-8")
    return str(path)


def execute(case, make_kernel, make_reference, config=None, check_ir=None, check_source=None, lower_ir=None):
    config = config or Options()
    print("[用例说明] " + json.dumps(asdict(case), ensure_ascii=False), flush=True)
    evidence, phase = {}, "frontend"
    row = {
        "id": case.id,
        "feature": case.feature,
        "scenario": case.scenario,
        "responsibility": case.responsibility,
        "mode": config.mode,
        "expected": case.expected,
    }
    try:
        if case.kind == "numerical":
            phase = "reference"
            inputs, expected, initial = make_reference()
            evidence["cpu_reference.pt"] = {"inputs": inputs, "expected": expected, "initial_output": initial}
        phase = "frontend"
        program = make_kernel()
        phase = "lowering"
        lowered, source = emit_ir(program, evidence, lower_ir=lower_ir, manual_schedule=case.manual_schedule)
        if case.source_pattern:
            assert re.search(case.source_pattern, source or lowered.script()), "源码/IR 未满足本用例的目标断言"
        if check_source:
            check_source(source)
        if case.kind == "reject" and source is None:
            raise AssertionError("局部IR接受了目标非法条件，预期的下沉诊断没有发生")
        if case.kind == "ir":
            assert check_ir is not None, "IR case requires a behavioral verifier"
            check_ir(lowered)
            row.update(status="PASS", level="IR", actual="目标 IR 断言通过；数值运行不适用")
        else:
            phase = "device_compile"
            if not shutil.which("bisheng"):
                row.update(status="SKIP", level=phase, actual="缺少 Bisheng，设备编译尚未验收")
            else:
                compile_device(source, evidence)
                if case.kind == "reject":
                    raise AssertionError("目标非法输入被接受：期望在 " + case.stage + " 或更早阶段拒绝")
                if case.kind == "caller":
                    row.update(
                        status="OBSERVED",
                        level="DEVICE_COMPILE",
                        actual="编译器接受了静态越界调用；范围合法性由调用方保证，禁止启动此非法指令；本项只记录防护现状",
                    )
                elif config.mode == "compile":
                    check_signature(program, inputs, expected, initial)
                    row.update(status="COMPILE_ONLY", level="DEVICE_COMPILE", actual="源码/IR/设备编译通过；数值尚未执行")
                else:
                    check_signature(program, inputs, expected, initial)
                    phase = "execution"
                    if config.mode == "simulator":
                        from testing.ascend.gemm_features._simulator import simulate

                        error = simulate(case, program, source, inputs, expected, initial, evidence)
                    else:
                        error = hardware(case, program, inputs, expected, initial, config, evidence)
                    if error is None:
                        row.update(status="SKIP", level="HARDWARE", actual="NPU 不可用；真机数值尚未验收")
                    else:
                        row.update(
                            status="PASS",
                            level="HARDWARE" if config.mode == "hardware" else "SIMULATOR",
                            actual="运行结果正确：已与 CPU 参考逐次比较",
                            max_abs_error=error,
                            launches=case.repeats,
                        )
    except Exception as error:
        text = str(error)
        if (
            case.kind in {"reject", "caller"}
            and phase in {"frontend", "lowering", "device_compile"}
            and not isinstance(error, (FileNotFoundError, subprocess.TimeoutExpired))
            and case.diagnostic
            and re.search(case.diagnostic, text, re.IGNORECASE)
        ):
            row.update(
                status="OBSERVED" if case.kind == "caller" else "PASS",
                level=phase.upper(),
                actual="编译器提供了额外静态防护；调用方仍须保证运行时范围" if case.kind == "caller" else "目标约束已拦截",
                diagnostic=text,
            )
            print("[源码/IR] 在目标约束处已拒绝，后续源码或 IR 不产生。", flush=True)
        else:
            details = traceback.format_exc()
            row.update(status="FAIL", level=phase.upper(), actual=text)
            try:
                failure_path = save_failure(case, make_kernel, evidence, config, details)
                if failure_path:
                    row["failure_path"] = failure_path
            except OSError as logging_error:
                row["logging_error"] = str(logging_error)
    print("[第二部分：验收结果] " + json.dumps(row, ensure_ascii=False), flush=True)
    return row


def pytest_case(case, make_kernel, make_reference, config, check_ir=None, check_source=None, lower_ir=None):
    import pytest

    row = execute(case, make_kernel, make_reference, config, check_ir, check_source, lower_ir)
    if row["status"] in {"SKIP", "COMPILE_ONLY"}:
        pytest.skip(row["actual"])
    assert row["status"] in {"PASS", "OBSERVED"}, json.dumps(row, ensure_ascii=False)


def standalone(case, make_kernel, make_reference, check_ir=None, check_source=None, lower_ir=None):
    args = options(argparse.ArgumentParser(description=case.scenario)).parse_args()
    row = execute(case, make_kernel, make_reference, make_options(args), check_ir, check_source, lower_ir)
    return 0 if row["status"] in {"PASS", "OBSERVED"} else 1 if row["status"] == "FAIL" else 2
