"""运行 Edge-LLM C++ ``llm_bench`` 并归档可复用的低层 benchmark 证据。

该入口绕过 HTTP/视觉预处理链路，适合比较同一 engine 的 prefill/decode
行为。``llm_bench`` 当前输出的是每次进程的一行聚合 CSV，本脚本不把它
扩展成不存在的逐 iteration 分布，而是把原始 CSV、完整日志、命令和解析
后的 JSONL 绑定在同一个 run metadata 中。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


CSV_PATH_PATTERN = re.compile(r"E2E timing CSV saved to:\s*(\S+)")
RUNTIME_WARNING_MARKERS = (
    "IRuntime::~IRuntime",
    "API Usage Error",
)


class DirectBenchmarkError(ValueError):
    """表示 llm_bench 输出不能被可靠归档。"""


def build_command(
    *,
    llm_bench: Path,
    engine_dir: Path,
    mode: str,
    iterations: int,
    warmup: int,
    seed: int,
    output_dir: Path,
    engine_dir_flag: str = "--engineDir",
    output_dir_flag: str | None = "--outputDir",
    past_kv_len: int = 768,
    osl: int = 1,
    input_len: int = 768,
    reuse_kv_len: int = 0,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """生成一次板端 ``llm_bench`` 命令，不执行外部进程。"""
    if mode not in {"prefill", "decode"}:
        raise ValueError(f"unsupported mode: {mode}")
    if not engine_dir_flag:
        raise ValueError("engine_dir_flag must not be empty")
    command = [
        str(llm_bench),
        engine_dir_flag,
        str(engine_dir),
        "--mode",
        mode,
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--seed",
        str(seed),
    ]
    if output_dir_flag is not None:
        if not output_dir_flag:
            raise ValueError("output_dir_flag must be None or non-empty")
        command.extend([output_dir_flag, str(output_dir)])
    if mode == "decode":
        command.extend(["--pastKVLen", str(past_kv_len), "--osl", str(osl)])
    else:
        command.extend(
            ["--inputLen", str(input_len), "--reuseKVLen", str(reuse_kv_len)]
        )
    command.extend(str(argument) for argument in extra_args)
    return command


def parse_e2e_csv(csv_path: Path, *, expected_mode: str) -> dict[str, Any]:
    """解析 Edge-LLM 的一行 E2E CSV，并保留原始字段。"""
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise DirectBenchmarkError(f"cannot read llm_bench CSV: {csv_path}") from error
    if len(rows) != 1:
        raise DirectBenchmarkError(
            f"expected exactly one llm_bench CSV row, got {len(rows)}: {csv_path}"
        )
    row = {str(key): value for key, value in rows[0].items() if key is not None}
    if row.get("mode") != expected_mode:
        raise DirectBenchmarkError(
            f"llm_bench CSV mode mismatch: expected {expected_mode}, got {row.get('mode')}"
        )
    required = {"e2e_time_ms", "per_token_ms", "throughput_tps"}
    missing = sorted(field for field in required if not row.get(field))
    if missing:
        raise DirectBenchmarkError(
            f"llm_bench CSV missing numeric fields {missing}: {csv_path}"
        )
    parsed: dict[str, Any] = dict(row)
    for field in required:
        try:
            parsed[field] = float(row[field])
        except (TypeError, ValueError) as error:
            raise DirectBenchmarkError(
                f"llm_bench CSV field {field} is not numeric: {csv_path}"
            ) from error
    for field in ("batch_size", "osl", "past_kv_len", "input_len"):
        if field in row and row[field] not in (None, ""):
            try:
                parsed[field] = int(row[field])
            except ValueError as error:
                raise DirectBenchmarkError(
                    f"llm_bench CSV field {field} is not an integer: {csv_path}"
                ) from error
    return parsed


def find_e2e_csv(*, output_dir: Path, combined_output: str, mode: str) -> Path:
    """优先从输出目录定位 CSV，必要时回退到日志中的绝对路径。"""
    candidates = sorted(
        path
        for path in output_dir.glob("**/*.csv")
        if f"e2e_{mode}_" in path.name
    )
    if not candidates:
        logged_paths = [Path(match) for match in CSV_PATH_PATTERN.findall(combined_output)]
        candidates = [
            path for path in logged_paths if f"e2e_{mode}_" in path.name and path.is_file()
        ]
    if not candidates:
        raise DirectBenchmarkError(
            f"cannot locate e2e_{mode} CSV under {output_dir} or llm_bench output"
        )
    if len(candidates) > 1:
        raise DirectBenchmarkError(
            f"multiple e2e_{mode} CSV files found; pass a clean output directory: {candidates}"
        )
    return candidates[0]


def run_direct_benchmark(
    *,
    llm_bench: Path,
    engine_dir: Path,
    mode: str,
    output_jsonl: Path,
    repetitions: int = 3,
    iterations: int = 20,
    warmup: int = 10,
    seed: int = 0,
    past_kv_len: int = 768,
    osl: int = 1,
    input_len: int = 768,
    reuse_kv_len: int = 0,
    output_root: Path | None = None,
    log_dir: Path | None = None,
    engine_dir_flag: str = "--engineDir",
    output_dir_flag: str | None = "--outputDir",
    extra_args: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """执行重复 direct benchmark，返回并写入 JSONL 和 run metadata。"""
    _validate_positive("repetitions", repetitions)
    _validate_positive("iterations", iterations)
    _validate_non_negative("warmup", warmup)
    if mode not in {"prefill", "decode"}:
        raise ValueError(f"unsupported mode: {mode}")
    if output_root is None:
        output_root = output_jsonl.parent / "direct_csv"
    if log_dir is None:
        log_dir = output_jsonl.parent / "direct_logs"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        run_output_dir = output_root / f"{mode}_rep{repetition}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            llm_bench=llm_bench,
            engine_dir=engine_dir,
            mode=mode,
            iterations=iterations,
            warmup=warmup,
            seed=seed,
            output_dir=run_output_dir,
            engine_dir_flag=engine_dir_flag,
            output_dir_flag=output_dir_flag,
            past_kv_len=past_kv_len,
            osl=osl,
            input_len=input_len,
            reuse_kv_len=reuse_kv_len,
            extra_args=extra_args,
        )
        log_path = log_dir / f"{mode}_rep{repetition}.log"
        started = time.perf_counter()
        completed_process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        process_runtime_ms = (time.perf_counter() - started) * 1000.0
        combined_output = "\n".join(
            part for part in (completed_process.stdout, completed_process.stderr) if part
        )
        log_path.write_text(combined_output, encoding="utf-8")
        warnings = [
            line.strip()
            for line in combined_output.splitlines()
            if any(marker in line for marker in RUNTIME_WARNING_MARKERS)
        ]
        run_record: dict[str, Any] = {
            "repetition": repetition,
            "command": command,
            "returncode": completed_process.returncode,
            "log": str(log_path),
            "csv_output_dir": str(run_output_dir),
            "process_runtime_ms": process_runtime_ms,
            "warnings": warnings,
        }
        if completed_process.returncode != 0:
            row = _failed_row(
                mode=mode,
                repetition=repetition,
                category="llm_bench_exit_code",
                message=f"llm_bench exited with return code {completed_process.returncode}",
            )
            run_record["status"] = "failed"
            run_record["failure"] = row["failure"]
            rows.append(row)
            run_records.append(run_record)
            continue
        try:
            csv_path = find_e2e_csv(
                output_dir=run_output_dir,
                combined_output=combined_output,
                mode=mode,
            )
            parsed = parse_e2e_csv(csv_path, expected_mode=mode)
        except DirectBenchmarkError as error:
            row = _failed_row(
                mode=mode,
                repetition=repetition,
                category="benchmark_output_invalid",
                message=str(error),
            )
            run_record["status"] = "failed"
            run_record["failure"] = row["failure"]
            rows.append(row)
            run_records.append(run_record)
            continue

        e2e_ms = float(parsed["e2e_time_ms"])
        timing_name = "decode_ms" if mode == "decode" else "prefill_ms"
        output_tokens = int(parsed.get("osl", osl)) if mode == "decode" else 0
        row = {
            "sample_id": f"direct-{mode}-rep{repetition}",
            "repetition": repetition,
            "status": "completed",
            "output_tokens": output_tokens,
            "timings_ms": {timing_name: e2e_ms, "end_to_end_ms": e2e_ms},
        }
        run_record.update(
            {
                "status": "completed",
                "csv": str(csv_path),
                "csv_row": parsed,
            }
        )
        rows.append(row)
        run_records.append(run_record)

    output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "schema_version": "parksight_tensorrt_direct_benchmark_v1",
        "backend": "tensorrt_edge_llm_cpp_direct",
        "llm_bench": str(llm_bench),
        "engine_dir": str(engine_dir),
        "mode": mode,
        "repetitions": repetitions,
        "iterations": iterations,
        "warmup": warmup,
        "seed": seed,
        "past_kv_len": past_kv_len if mode == "decode" else None,
        "osl": osl if mode == "decode" else None,
        "input_len": input_len if mode == "prefill" else None,
        "reuse_kv_len": reuse_kv_len if mode == "prefill" else None,
        "engine_dir_flag": engine_dir_flag,
        "output_dir_flag": output_dir_flag,
        "extra_args": list(extra_args),
        "output_jsonl": str(output_jsonl),
        "runs": run_records,
        "completed_runs": sum(record["status"] == "completed" for record in run_records),
        "failed_runs": sum(record["status"] == "failed" for record in run_records),
    }
    if provenance is not None:
        metadata["engine_provenance"] = dict(provenance)
    metadata_path = output_jsonl.with_suffix(output_jsonl.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-bench", required=True, type=Path)
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("prefill", "decode"))
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--past-kv-len", type=int, default=768)
    parser.add_argument("--osl", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=768)
    parser.add_argument("--reuse-kv-len", type=int, default=0)
    parser.add_argument("--engine-dir-flag", default="--engineDir")
    parser.add_argument(
        "--output-dir-flag",
        default="--outputDir",
        help="llm_bench 输出目录参数；传空字符串可关闭并依赖日志中的 CSV 路径",
    )
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--provenance-json", type=Path)
    args = parser.parse_args(argv)
    if args.repetitions <= 0 or args.iterations <= 0 or args.warmup < 0:
        parser.error("repetitions/iterations must be positive and warmup non-negative")
    if args.mode == "decode" and (args.past_kv_len < 0 or args.osl <= 0):
        parser.error("decode past-kv-len must be non-negative and osl positive")
    if args.mode == "prefill" and (args.input_len <= 0 or args.reuse_kv_len < 0):
        parser.error("prefill input-len must be positive and reuse-kv-len non-negative")
    provenance = None
    if args.provenance_json is not None:
        provenance = _read_object(args.provenance_json)
    output_dir_flag = args.output_dir_flag or None
    metadata = run_direct_benchmark(
        llm_bench=args.llm_bench,
        engine_dir=args.engine_dir,
        mode=args.mode,
        output_jsonl=args.output_jsonl,
        repetitions=args.repetitions,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        past_kv_len=args.past_kv_len,
        osl=args.osl,
        input_len=args.input_len,
        reuse_kv_len=args.reuse_kv_len,
        output_root=args.output_root,
        log_dir=args.log_dir,
        engine_dir_flag=args.engine_dir_flag,
        output_dir_flag=output_dir_flag,
        extra_args=args.extra_arg,
        provenance=provenance,
    )
    print(args.output_jsonl)
    print(
        f"mode={args.mode} repetitions={metadata['repetitions']} "
        f"completed={metadata['completed_runs']} failed={metadata['failed_runs']}"
    )
    return 0 if metadata["failed_runs"] == 0 else 2


def _failed_row(*, mode: str, repetition: int, category: str, message: str) -> dict[str, Any]:
    return {
        "sample_id": f"direct-{mode}-rep{repetition}",
        "repetition": repetition,
        "status": "failed",
        "output_tokens": None,
        "timings_ms": {"end_to_end_ms": 0.0},
        "failure": {
            "category": category,
            "message": message,
        },
    }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DirectBenchmarkError(f"cannot read provenance JSON: {path}") from error
    if not isinstance(payload, dict):
        raise DirectBenchmarkError("provenance JSON must be an object")
    return payload


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


if __name__ == "__main__":
    raise SystemExit(main())
