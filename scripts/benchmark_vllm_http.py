"""Run a vLLM HTTP study with separate warm-up and optional local GPU telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from parksight_vlm.app.config import AppStudyConfig
from parksight_vlm.app.run_study import run_configured_study
from parksight_vlm.studies.server_metrics import summarize_window


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New report path for a repeated run")
    parser.add_argument("--gpu-telemetry", action="store_true", help="Sample this host's GPU; only use when the vLLM server runs here")
    parser.add_argument("--metrics-url", help="vLLM Prometheus endpoint, e.g. http://127.0.0.1:8000/metrics; requires exclusive server traffic")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of independent closed-loop HTTP workers")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    config = AppStudyConfig.load(args.config)
    if config.runtime.backend != "vllm_http":
        parser.error("benchmark requires a vllm_http runtime")
    if args.output is not None:
        config = replace(config, output_path=args.output.resolve())
    output = config.output_path
    warmup_output = output.with_name(output.stem + ".warmup.json")
    execution_output = output.with_name(output.stem + ".execution.json")
    telemetry_output = output.with_suffix(".gpu.csv")
    metrics_before = output.with_suffix(".metrics.before.prom")
    metrics_after = output.with_suffix(".metrics.after.prom")
    metrics_summary = output.with_suffix(".server_metrics.json")
    destinations = [output, warmup_output, execution_output]
    if args.gpu_telemetry:
        destinations.append(telemetry_output)
    if args.metrics_url:
        destinations.extend([metrics_before, metrics_after, metrics_summary])
    for path in destinations:
        if path.exists():
            parser.error(f"evidence already exists: {path}; use --output for another run")
    output.parent.mkdir(parents=True, exist_ok=True)
    execution = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "study_id": config.study.study_id,
        "workload_identity": config.study.workload.identity,
        "runtime_options_sha256": hashlib.sha256(
            json.dumps(dict(config.runtime.options), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "image_preprocessing": config.runtime.options.get("image_preprocessing", "source"),
        "client_source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((Path(__file__).resolve().parents[1] / "src" / "parksight_vlm").rglob("*.py"))
        },
        "dataset_metadata_sha256": {
            "manifest": hashlib.sha256(config.manifest_path.read_bytes()).hexdigest(),
            "annotations": hashlib.sha256(config.annotations_path.read_bytes()).hexdigest(),
        },
        "gpu_telemetry_scope": "local_device_1s_samples" if args.gpu_telemetry else None,
        "status": "running",
        "client_concurrency": args.concurrency,
        "server_metrics_enabled": bool(args.metrics_url),
    }

    def save_execution() -> None:
        execution_output.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")

    # Reserve the execution record atomically so concurrent runs cannot share
    # the same evidence paths after both have passed the existence checks.
    try:
        with execution_output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(execution, indent=2) + "\n")
    except FileExistsError:
        parser.error(f"evidence already exists: {execution_output}; use --output")
    try:
        warmup_config = replace(
            config,
            study=replace(config.study, study_id=config.study.study_id + "_warmup", repetitions=1),
            output_path=warmup_output,
        )
        warmup = run_configured_study(warmup_config)
        if warmup.failure_summary:
            execution["status"] = "warmup_failed"
            return 2
        print(f"Warm-up completed: {len(warmup.records)} requests", flush=True)
        monitor = None
        telemetry = None
        try:
            if args.gpu_telemetry:
                telemetry = telemetry_output.open("w", encoding="utf-8")
                monitor = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu", "--format=csv", "--loop-ms=1000"],
                    stdout=telemetry,
                )
            if args.metrics_url:
                with urlopen(args.metrics_url, timeout=10) as response:
                    before = response.read().decode("utf-8")
                metrics_before.write_text(before, encoding="utf-8")
            window_start = time.perf_counter()
            report = run_configured_study(config, concurrency=args.concurrency)
            if args.metrics_url:
                with urlopen(args.metrics_url, timeout=10) as response:
                    after = response.read().decode("utf-8")
                window_seconds = time.perf_counter() - window_start
                metrics_after.write_text(after, encoding="utf-8")
                server = summarize_window(before, after, window_seconds)
                server["client_backend_completed_requests"] = report.performance_metrics.backend_completed_sample_count
                server["request_count_matches_client"] = server.get("request_success_count") == report.performance_metrics.backend_completed_sample_count
                server["output_token_count_matches_client"] = server.get("generated_tokens") == report.performance_metrics.token_counts["output_tokens"]
                metrics_summary.write_text(json.dumps(server, indent=2) + "\n", encoding="utf-8")
        finally:
            if monitor is not None:
                monitor.terminate()
                monitor.wait(timeout=10)
            if telemetry is not None:
                telemetry.close()
        execution["status"] = "failed" if report.failure_summary else "succeeded"
        print(output)
        print(json.dumps(report.performance_metrics.to_mapping(), indent=2))
        return 2 if report.failure_summary else 0
    except BaseException:
        execution["status"] = "interrupted_or_failed"
        raise
    finally:
        execution["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_execution()


if __name__ == "__main__":
    raise SystemExit(main())
