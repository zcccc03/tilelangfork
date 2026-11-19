"""One-command runner. Numeric cases in hardware/simulator modes use independent processes."""

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.ascend.gemm_features._support import make_options, options  # noqa: E402

GROUPS = ["scaled_gemm", "data_types", "tails", "semantics"]
HERE = Path(__file__).resolve().parent


def modules():
    return [
        importlib.import_module(f"testing.ascend.gemm_features.{group}.{file.stem}")
        for group in GROUPS
        for file in sorted((HERE / group).glob("test_*.py"))
    ]


def report(path, results, selected):
    rows = [
        "# GEMM 逐用例执行报告",
        "",
        "本文件只记录本次实际执行结果。固定随机种子写在各测试文件；不锁定环境版本。",
        "",
        "COMPILE_ONLY 表示设备编译通过但数值未验收；SKIP 表示必需条件缺失；均不计作真机数值通过。",
        "SIMULATOR 表示仿真辅助验数，不能作为真机实测结果。",
        "OBSERVED 表示调用方约束的编译防护观察，不是数值PASS，也不要求编译器承担调用方责任。",
        "",
        "| 用例 | 三级特性 / 场景 | 文件及直接运行命令 | 责任 / 验收标准 | 结果 / 实际观察 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result, module in zip(results, selected, strict=True):
        case = module.CASE
        file = Path(module.__file__).relative_to(ROOT).as_posix()
        actual = result.get("actual", "")
        if result.get("diagnostic"):
            lines = result["diagnostic"].splitlines()
            diagnostic = next((line for line in lines if re.search(case.diagnostic, line, re.IGNORECASE)), lines[-1])
            delimiter = "`" * (max((len(run) for run in re.findall(r"`+", diagnostic)), default=0) + 1)
            actual += f"<br>实际诊断：{delimiter}{diagnostic}{delimiter}"
        if "max_abs_error" in result:
            actual += f"<br>max_abs_error={result['max_abs_error']}, launches={result['launches']}"
        if result.get("failure_path"):
            relative = os.path.relpath(result["failure_path"], Path(path).resolve().parent)
            actual += f"<br>[失败材料](<{relative}>)"
        level = (
            case.stage
            if case.kind == "reject"
            else "编译防护观察（调用方负责范围）"
            if case.kind == "caller"
            else "IR"
            if case.kind == "ir"
            else "真机数值"
        )
        cells = [
            case.id,
            f"{case.feature}<br>{case.scenario}",
            f"`{file}`<br>`python {file} --mode {result.get('mode', 'hardware')}`<br>内核来源：{case.kernel_origin}",
            f"{case.responsibility}<br>{case.expected}<br>必要层级：{level}",
            f"**{result['status']}** / {result.get('level', '')}<br>{actual}",
        ]
        rows.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", "<br>") for value in cells) + " |")
    counts = dict(Counter(result["status"] for result in results))
    guide_link = os.path.relpath(HERE / "EXECUTION_GUIDE.md", Path(path).expanduser().resolve().parent)
    rows += [
        "",
        "统计：`" + json.dumps(counts, ensure_ascii=False) + "`",
        "",
        f"完整输入输出契约、约束和执行方式见 [执行指南](<{guide_link}>)。",
    ]
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")


def child(module, config, timeout):
    command = [sys.executable, str(Path(module.__file__)), "--mode", config.mode, "--device", str(config.device)]
    if config.failure_dir:
        command += ["--failure-dir", config.failure_dir]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        print(process.stdout, end="", flush=True)
        if process.stderr:
            print(process.stderr, end="", file=sys.stderr, flush=True)
        records = [line.split("] ", 1)[1] for line in process.stdout.splitlines() if line.startswith("[第二部分：验收结果] ")]
        if records:
            row = json.loads(records[-1])
            expected_code = 0 if row["status"] in {"PASS", "OBSERVED"} else 1 if row["status"] == "FAIL" else 2
            if process.returncode == expected_code:
                return row
        error = f"子进程退出码 {process.returncode}，没有有效的最终验收结果\n{process.stdout}\n{process.stderr}"
    except subprocess.TimeoutExpired:
        error = f"子进程超过 {timeout} 秒；需要单独复现该用例"
    if config.failure_dir:
        try:
            path = Path(config.failure_dir).resolve() / module.CASE.id
            path.mkdir(parents=True, exist_ok=True)
            (path / "process_failure.log").write_text(error, encoding="utf-8")
        except OSError as logging_error:
            error += "\n失败材料保存错误：" + str(logging_error)
    return {"id": module.CASE.id, "status": "FAIL", "level": "PROCESS", "mode": config.mode, "actual": error}


def main():
    parser = options(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--case", action="append", help="Independent case ID; repeat to select several")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", help="Explicit optional Markdown report path; independent of failure logs")
    parser.add_argument("--timeout", type=int, default=240, help="Per numerical child process seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    available = modules()
    ids = {module.CASE.id for module in available}
    if args.case and set(args.case) - ids:
        parser.error("Unknown case IDs: " + ", ".join(sorted(set(args.case) - ids)))
    selected = [
        module
        for module in available
        if (not args.group or module.CASE.group == args.group) and (not args.case or module.CASE.id in args.case)
    ]
    if not selected:
        parser.error("No cases selected")
    if args.list:
        for module in selected:
            print(module.CASE.id, module.CASE.group, module.CASE.feature, module.CASE.scenario)
        return 0
    config, results = make_options(args), []
    for module in selected:
        print("RUN " + module.CASE.id, flush=True)
        if module.CASE.kind == "numerical" and config.mode != "compile":
            result = child(module, config, args.timeout)
        else:
            result = module.main(config)
        results.append(result)
        if args.report:
            report(args.report, results, selected[: len(results)])
    if args.report:
        report(args.report, results, selected)
    print("SUMMARY " + json.dumps(dict(Counter(result["status"] for result in results))), flush=True)
    return (
        1
        if any(result["status"] == "FAIL" for result in results)
        else 2
        if any(result["status"] in {"SKIP", "COMPILE_ONLY"} for result in results)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
