"""按并发度运行 Edge-LLM HTTP benchmark，输出 dynamic/continuous batching 对比。

该脚本只负责实验编排。``concurrency`` 是客户端并发请求数，只有服务端
engine 的 max batch、调度器和 KV cache 同时允许合并时，结果才可解释为
dynamic batching；它不把多个独立 HTTP worker 的吞吐自动宣称为 continuous
batching。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def build_runner_command(
    *,
    runner_script: Path,
    manifest: Path,
    workload: Path,
    data_root: Path,
    output_jsonl: Path,
    warmup_output_jsonl: Path | None,
    run_metadata_json: Path,
    concurrency: int,
    split: str,
    limit: int | None,
    repetitions: int,
    warmup: int,
    endpoint: str,
    model_name: str,
    timeout_seconds: float,
    no_stream_responses: bool,
    reuse_http_connection: bool,
    engine_max_batch_size: int | None,
    engine_max_kv_pool_pages: int | None,
    python_executable: str = sys.executable,
    extra_runner_args: Sequence[str] = (),
) -> list[str]:
    """生成一次 concurrency sweep 的子进程命令。"""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    command = [
        python_executable,
        str(runner_script),
        "--manifest",
        str(manifest),
        "--workload",
        str(workload),
        "--data-root",
        str(data_root),
        "--output-jsonl",
        str(output_jsonl),
        "--run-metadata-json",
        str(run_metadata_json),
        "--split",
        split,
        "--repetitions",
        str(repetitions),
        "--warmup",
        str(warmup),
        "--concurrency",
        str(concurrency),
        "--endpoint",
        endpoint,
        "--model-name",
        model_name,
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if warmup_output_jsonl is not None:
        command.extend(["--warmup-output-jsonl", str(warmup_output_jsonl)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if engine_max_batch_size is not None:
        command.extend(["--engine-max-batch-size", str(engine_max_batch_size)])
    if engine_max_kv_pool_pages is not None:
        command.extend(["--engine-max-kv-pool-pages", str(engine_max_kv_pool_pages)])
    if no_stream_responses:
        command.append("--no-stream-responses")
    if reuse_http_connection:
        command.append("--reuse-http-connection")
    command.extend(str(argument) for argument in extra_runner_args)
    return command


def run_sweep(
    *,
    runner_script: Path,
    manifest: Path,
    workload: Path,
    data_root: Path,
    output_dir: Path,
    concurrencies: Sequence[int],
    split: str = "test",
    limit: int | None = None,
    repetitions: int = 3,
    warmup: int = 1,
    endpoint: str = "http://127.0.0.1:8000",
    model_name: str = "local",
    timeout_seconds: float = 120.0,
    no_stream_responses: bool = False,
    reuse_http_connection: bool = False,
    engine_max_batch_size: int | None = None,
    engine_max_kv_pool_pages: int | None = None,
    extra_runner_args: Sequence[str] = (),
    fail_fast: bool = False,
) -> dict[str, Any]:
    """执行每个并发度的 HTTP runner，并汇总 run metadata。"""
    normalized = [int(value) for value in concurrencies]
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("concurrencies must contain positive integers")
    if engine_max_batch_size is not None and max(normalized) > engine_max_batch_size:
        raise ValueError(
            "concurrency exceeds engine_max_batch_size; do not interpret a rejected batch as throughput"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for concurrency in normalized:
        stem = f"concurrency_{concurrency}"
        output_jsonl = output_dir / f"{stem}.jsonl"
        warmup_jsonl = output_dir / f"{stem}.warmup.jsonl" if warmup > 0 else None
        metadata_json = output_dir / f"{stem}.run.json"
        command = build_runner_command(
            runner_script=runner_script,
            manifest=manifest,
            workload=workload,
            data_root=data_root,
            output_jsonl=output_jsonl,
            warmup_output_jsonl=warmup_jsonl,
            run_metadata_json=metadata_json,
            concurrency=concurrency,
            split=split,
            limit=limit,
            repetitions=repetitions,
            warmup=warmup,
            endpoint=endpoint,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            no_stream_responses=no_stream_responses,
            reuse_http_connection=reuse_http_connection,
            engine_max_batch_size=engine_max_batch_size,
            engine_max_kv_pool_pages=engine_max_kv_pool_pages,
            extra_runner_args=extra_runner_args,
        )
        completed_process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log_path = output_dir / f"{stem}.log"
        log_path.write_text(
            "\n".join(
                part
                for part in (completed_process.stdout, completed_process.stderr)
                if part
            ),
            encoding="utf-8",
        )
        record: dict[str, Any] = {
            "concurrency": concurrency,
            "command": command,
            "returncode": completed_process.returncode,
            "log": str(log_path),
            "output_jsonl": str(output_jsonl),
            "warmup_output_jsonl": str(warmup_jsonl) if warmup_jsonl else None,
            "run_metadata_json": str(metadata_json),
        }
        if metadata_json.is_file():
            try:
                record["run_metadata"] = json.loads(
                    metadata_json.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as error:
                record["metadata_error"] = str(error)
        record["status"] = "completed" if completed_process.returncode == 0 else "failed"
        records.append(record)
        if fail_fast and completed_process.returncode != 0:
            break
    summary = {
        "schema_version": "parksight_tensorrt_concurrency_sweep_v1",
        "backend": "tensorrt_edge_llm_http",
        "runner_script": str(runner_script),
        "manifest": str(manifest),
        "workload": str(workload),
        "data_root": str(data_root),
        "endpoint": endpoint,
        "engine_max_batch_size": engine_max_batch_size,
        "engine_max_kv_pool_pages": engine_max_kv_pool_pages,
        "concurrencies": normalized,
        "records": records,
        "completed_runs": sum(record["status"] == "completed" for record in records),
        "failed_runs": sum(record["status"] == "failed" for record in records),
    }
    summary_path = output_dir / "concurrency_sweep.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runner-script", type=Path, default=Path(__file__).with_name("run_edgellm_benchmark.py"))
    parser.add_argument("--concurrency", action="append", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model-name", default="local")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--no-stream-responses", action="store_true")
    parser.add_argument("--reuse-http-connection", action="store_true")
    parser.add_argument("--engine-max-batch-size", type=int)
    parser.add_argument("--engine-max-kv-pool-pages", type=int)
    parser.add_argument("--extra-runner-arg", action="append", default=[])
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    concurrencies = args.concurrency or [1, 2, 4]
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.repetitions <= 0 or args.warmup < 0:
        parser.error("repetitions must be positive and warmup non-negative")
    if args.engine_max_batch_size is not None and args.engine_max_batch_size <= 0:
        parser.error("--engine-max-batch-size must be positive")
    if args.engine_max_kv_pool_pages is not None and args.engine_max_kv_pool_pages <= 0:
        parser.error("--engine-max-kv-pool-pages must be positive")
    summary = run_sweep(
        runner_script=args.runner_script,
        manifest=args.manifest,
        workload=args.workload,
        data_root=args.data_root,
        output_dir=args.output_dir,
        concurrencies=concurrencies,
        split=args.split,
        limit=args.limit,
        repetitions=args.repetitions,
        warmup=args.warmup,
        endpoint=args.endpoint,
        model_name=args.model_name,
        timeout_seconds=args.timeout_seconds,
        no_stream_responses=args.no_stream_responses,
        reuse_http_connection=args.reuse_http_connection,
        engine_max_batch_size=args.engine_max_batch_size,
        engine_max_kv_pool_pages=args.engine_max_kv_pool_pages,
        extra_runner_args=args.extra_runner_arg,
        fail_fast=args.fail_fast,
    )
    print(summary["summary_path"])
    print(
        f"runs={len(summary['records'])} completed={summary['completed_runs']} "
        f"failed={summary['failed_runs']}"
    )
    return 0 if summary["failed_runs"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
