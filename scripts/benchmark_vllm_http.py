"""Run a vLLM HTTP study with separate warm-up and optional local GPU telemetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from parksight_vlm.app.config import AppStudyConfig
from parksight_vlm.app.run_study import run_configured_study
from parksight_vlm.studies.server_metrics import summarize_window
from parksight_vlm.studies.server_config import compare_server_identity, extract_server_config


GPU_TELEMETRY_BASIC_FIELDS = (
    "timestamp", "memory.used", "utilization.gpu", "power.draw", "temperature.gpu",
)
GPU_TELEMETRY_EXTENDED_FIELDS = GPU_TELEMETRY_BASIC_FIELDS + (
    "clocks.current.sm", "clocks.current.memory",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)


def _gpu_columns(header: list[str]) -> dict[str, int] | None:
    names = [name.strip().split(" [", 1)[0] for name in header]
    if not set(GPU_TELEMETRY_BASIC_FIELDS).issubset(names):
        return None
    return {name: names.index(name) for name in GPU_TELEMETRY_BASIC_FIELDS}


def _valid_gpu_sample_time(
    row: list[str], *, width: int, columns: dict[str, int],
) -> datetime | None:
    if len(row) != width:
        return None
    try:
        for name in GPU_TELEMETRY_BASIC_FIELDS[1:]:
            value = float(row[columns[name]].strip().split(" ", 1)[0])
            if not math.isfinite(value) or value < 0:
                return None
        if float(row[columns["utilization.gpu"]].strip().split(" ", 1)[0]) > 100:
            return None
        timestamp = row[columns["timestamp"]].strip()
        for format_string in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(timestamp, format_string)
            except ValueError:
                continue
    except (ValueError, IndexError):
        pass
    return None


def select_gpu_telemetry_fields() -> tuple[str, ...]:
    """Probe supported nvidia-smi fields before the formal measurement starts."""
    for fields in (GPU_TELEMETRY_EXTENDED_FIELDS, GPU_TELEMETRY_BASIC_FIELDS):
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv"],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        rows = list(csv.reader(probe.stdout.splitlines()))
        if (probe.returncode == 0 and len(rows) >= 2
                and len(rows[0]) == len(fields)
                and (columns := _gpu_columns(rows[0])) is not None
                and _valid_gpu_sample_time(rows[1], width=len(rows[0]), columns=columns) is not None):
            return fields
    raise RuntimeError("nvidia-smi cannot provide the required GPU telemetry fields")


def summarize_gpu_telemetry(
    path: Path, *, window_start: datetime | None, window_end: datetime | None,
) -> tuple[int, int]:
    """Count valid CSV rows and rows sampled during the formal study window."""
    total, in_window, _, _ = inspect_gpu_telemetry_window(
        path, window_start=window_start, window_end=window_end,
    )
    return total, in_window


def inspect_gpu_telemetry_window(
    path: Path, *, window_start: datetime | None, window_end: datetime | None,
) -> tuple[int, int, datetime | None, datetime | None]:
    """Return valid sample coverage as well as counts for a timed run."""
    total = in_window = 0
    first_window_sample = last_window_sample = None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.reader(stream)
        header = next(rows, [])
        columns = _gpu_columns(header)
        if columns is None:
            return 0, 0, None, None
        for row in rows:
            sampled_local = _valid_gpu_sample_time(row, width=len(header), columns=columns)
            if sampled_local is None:
                continue
            total += 1
            if window_start is not None and window_end is not None:
                sampled = sampled_local.replace(tzinfo=window_start.tzinfo)
                if window_start <= sampled <= window_end:
                    in_window += 1
                    if first_window_sample is None or sampled < first_window_sample:
                        first_window_sample = sampled
                    if last_window_sample is None or sampled > last_window_sample:
                        last_window_sample = sampled
    return total, in_window, first_window_sample, last_window_sample


def assess_gpu_telemetry_coverage(
    *, window_start: datetime | None, window_end: datetime | None,
    sample_count: int, first_sample: datetime | None, last_sample: datetime | None,
) -> dict[str, int | float | bool | None]:
    """Require samples throughout the 1 Hz monitoring window, not one row."""
    duration = (
        (window_end - window_start).total_seconds()
        if window_start is not None and window_end is not None else None
    )
    minimum_samples = max(1, math.floor(duration / 3)) if duration is not None and duration >= 0 else None
    start_gap = (
        (first_sample - window_start).total_seconds()
        if first_sample is not None and window_start is not None else None
    )
    end_gap = (
        (window_end - last_sample).total_seconds()
        if last_sample is not None and window_end is not None else None
    )
    return {
        "minimum_window_samples": minimum_samples,
        "start_gap_seconds": start_gap,
        "end_gap_seconds": end_gap,
        "window_covered": (
            minimum_samples is not None
            and sample_count >= minimum_samples
            and start_gap is not None and 0 <= start_gap <= 3
            and end_gap is not None and 0 <= end_gap <= 3
        ),
    }


def stop_gpu_monitor(monitor: subprocess.Popen) -> tuple[int | None, bool]:
    """Stop the sampler with a bounded wait, preserving its prior exit status."""
    exit_code_before_stop = monitor.poll()
    if exit_code_before_stop is not None:
        return exit_code_before_stop, False
    monitor.terminate()
    try:
        monitor.wait(timeout=10)
        return None, False
    except subprocess.TimeoutExpired:
        monitor.kill()
        monitor.wait(timeout=10)
        return None, True


def _http_origin(url: str) -> tuple[str, str, int]:
    """Identify the HTTP service used for inference and evidence collection."""
    if not isinstance(url, str):
        raise ValueError("URL must be an absolute HTTP(S) URL")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be an absolute HTTP(S) URL")
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New report path for a repeated run")
    parser.add_argument("--gpu-telemetry", action="store_true", help="Sample this host's GPU; only use when the vLLM server runs here")
    parser.add_argument("--metrics-url", help="vLLM Prometheus endpoint, e.g. http://127.0.0.1:8000/metrics; requires exclusive server traffic")
    parser.add_argument("--server-info-url", help="Optional vLLM /server_info?config_format=json endpoint; verifies model revision and dtype, then saves effective settings before/after measurement")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of independent closed-loop HTTP workers")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    config = AppStudyConfig.load(args.config)
    if config.runtime.backend != "vllm_http":
        parser.error("benchmark requires a vllm_http runtime")
    try:
        inference_origin = _http_origin(
            config.runtime.options.get("base_url", "http://127.0.0.1:8000")
        )
        for name, url in (("--metrics-url", args.metrics_url),
                          ("--server-info-url", args.server_info_url)):
            if url is not None and _http_origin(url) != inference_origin:
                parser.error(f"{name} must use the inference service's HTTP origin")
    except ValueError as error:
        parser.error(str(error))
    if args.output is not None:
        config = replace(config, output_path=args.output.resolve())
    output = config.output_path
    warmup_output = output.with_name(output.stem + ".warmup.json")
    execution_output = output.with_name(output.stem + ".execution.json")
    telemetry_output = output.with_suffix(".gpu.csv")
    metrics_before = output.with_suffix(".metrics.before.prom")
    metrics_after = output.with_suffix(".metrics.after.prom")
    metrics_summary = output.with_suffix(".server_metrics.json")
    server_config_before = output.with_suffix(".server_config.before.json")
    server_config_after = output.with_suffix(".server_config.after.json")
    destinations = [output, warmup_output, execution_output]
    if args.gpu_telemetry:
        destinations.append(telemetry_output)
    if args.metrics_url:
        destinations.extend([metrics_before, metrics_after, metrics_summary])
    if args.server_info_url:
        destinations.extend([server_config_before, server_config_after])
    for path in destinations:
        if path.exists():
            parser.error(f"evidence already exists: {path}; use --output for another run")
    output.parent.mkdir(parents=True, exist_ok=True)
    execution = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        "server_config_snapshot_enabled": bool(args.server_info_url),
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
        if args.server_info_url:
            with urlopen(args.server_info_url, timeout=10) as response:
                effective_before = extract_server_config(json.load(response))
            server_config_before.write_text(
                json.dumps(effective_before, indent=2) + "\n", encoding="utf-8"
            )
            identity_before = compare_server_identity(
                effective_before,
                model_revision=config.runtime.model_revision,
                precision=config.runtime.precision,
            )
            execution["server_identity_before"] = identity_before
            if not all(value is True for value in identity_before.values()):
                execution["server_identity_verified"] = False
                execution["status"] = "server_identity_unverified"
                return 2
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
        measurement_start_local = None
        measurement_end_local = None
        try:
            if args.gpu_telemetry:
                try:
                    telemetry_fields = select_gpu_telemetry_fields()
                except RuntimeError as error:
                    execution["status"] = "gpu_telemetry_unavailable"
                    execution["gpu_telemetry_error"] = str(error)
                    return 2
                execution["gpu_telemetry_fields"] = list(telemetry_fields)
                telemetry = telemetry_output.open("w", encoding="utf-8")
                monitor = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=" + ",".join(telemetry_fields),
                     "--format=csv", "--loop-ms=1000"],
                    stdout=telemetry,
                )
            if args.metrics_url:
                with urlopen(args.metrics_url, timeout=10) as response:
                    before = response.read().decode("utf-8")
                metrics_before.write_text(before, encoding="utf-8")
            measurement_start_local = datetime.now().astimezone()
            window_start = time.perf_counter()
            report = run_configured_study(config, concurrency=args.concurrency)
            measurement_end_local = datetime.now().astimezone()
            if args.metrics_url:
                with urlopen(args.metrics_url, timeout=10) as response:
                    after = response.read().decode("utf-8")
                window_seconds = time.perf_counter() - window_start
                metrics_after.write_text(after, encoding="utf-8")
                server = summarize_window(before, after, window_seconds)
                server["client_backend_completed_requests"] = report.performance_metrics.backend_completed_sample_count
                server["request_count_matches_client"] = server.get("request_success_count") == report.performance_metrics.backend_completed_sample_count
                token_counts = report.performance_metrics.token_counts
                server["client_output_usage_record_count"] = token_counts.get("output_usage_record_count")
                server["client_output_tokens"] = token_counts.get("output_tokens")
                server["client_output_usage_complete"] = (
                    token_counts.get("output_usage_record_count")
                    == report.performance_metrics.backend_completed_sample_count
                    and token_counts.get("output_tokens") is not None
                )
                server["output_token_count_matches_client"] = (
                    server.get("generated_tokens") == token_counts["output_tokens"]
                    if server["client_output_usage_complete"]
                    and server.get("generated_tokens") is not None
                    else None
                )
                execution["server_metrics_consistent"] = bool(
                    server.get("valid") and server["request_count_matches_client"]
                    and server["output_token_count_matches_client"]
                    and server["finish_reason_coverage_complete"] is not False
                    and server["non_stop_request_count"] in (None, 0)
                )
                metrics_summary.write_text(json.dumps(server, indent=2) + "\n", encoding="utf-8")
            if args.server_info_url:
                with urlopen(args.server_info_url, timeout=10) as response:
                    effective_after = extract_server_config(json.load(response))
                server_config_after.write_text(json.dumps(effective_after, indent=2) + "\n", encoding="utf-8")
                execution["server_config_unchanged"] = effective_before == effective_after
                identity_after = compare_server_identity(
                    effective_after,
                    model_revision=config.runtime.model_revision,
                    precision=config.runtime.precision,
                )
                execution["server_identity_after"] = identity_after
                execution["server_identity_verified"] = (
                    execution["server_config_unchanged"]
                    and all(value is True for value in identity_after.values())
                )
        finally:
            try:
                if monitor is not None:
                    telemetry_exit_code_before_stop, forced_stop = stop_gpu_monitor(monitor)
                    execution["gpu_telemetry_forced_stop"] = forced_stop
            finally:
                if telemetry is not None:
                    telemetry.close()
            if telemetry is not None:
                sample_count, window_sample_count, first_sample, last_sample = inspect_gpu_telemetry_window(
                    telemetry_output,
                    window_start=measurement_start_local,
                    window_end=measurement_end_local,
                )
                execution["gpu_telemetry_sample_count"] = sample_count
                execution["gpu_telemetry_window_sample_count"] = window_sample_count
                coverage = assess_gpu_telemetry_coverage(
                    window_start=measurement_start_local,
                    window_end=measurement_end_local,
                    sample_count=window_sample_count,
                    first_sample=first_sample,
                    last_sample=last_sample,
                )
                execution.update({f"gpu_telemetry_{key}": value for key, value in coverage.items()})
                execution["gpu_measurement_window_start_local"] = (
                    measurement_start_local.isoformat() if measurement_start_local is not None else None
                )
                execution["gpu_measurement_window_end_local"] = (
                    measurement_end_local.isoformat() if measurement_end_local is not None else None
                )
                execution["gpu_telemetry_valid"] = (
                    monitor is not None
                    and telemetry_exit_code_before_stop is None
                    and execution["gpu_telemetry_window_covered"]
                )
        failed = (
            bool(report.failure_summary)
            or execution.get("server_config_unchanged") is False
            or execution.get("server_identity_verified") is False
            or execution.get("server_metrics_consistent") is False
            or execution.get("gpu_telemetry_valid") is False
        )
        execution["status"] = "failed" if failed else "succeeded"
        print(output)
        print(json.dumps(report.performance_metrics.to_mapping(), indent=2))
        return 2 if failed else 0
    except BaseException:
        execution["status"] = "interrupted_or_failed"
        raise
    finally:
        execution["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_execution()


if __name__ == "__main__":
    raise SystemExit(main())
