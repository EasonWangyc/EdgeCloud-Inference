"""vLLM Prometheus 窗口差分；保留直方图桶，不伪造逐请求计时。"""

from __future__ import annotations

import math
import re
from typing import Any


_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{.*\})?)\s+([^\s]+)(?:\s+[^\s]+)?$')
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*("(?:\\.|[^"\\])*")\s*')
_TIMINGS = (
    "time_to_first_token_seconds", "inter_token_latency_seconds",
    "request_time_per_output_token_seconds", "e2e_request_latency_seconds",
    "request_queue_time_seconds", "request_prefill_time_seconds", "request_decode_time_seconds",
)


def parse_samples(text: str) -> dict[str, float]:
    samples = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.fullmatch(line)
        if match is None:
            continue
        try:
            value = float(match[2])
        except ValueError:
            continue
        if math.isfinite(value):
            series = _canonical_series(match[1])
            if series in samples:
                raise ValueError("duplicate Prometheus metric series")
            samples[series] = value
    return samples


def _canonical_series(series: str) -> str:
    """Match the same Prometheus series even when its label order changes."""
    labels = _series_labels(series)
    if labels is None:
        return series
    name = series.split("{", 1)[0]
    normalized_labels = ",".join(
        f"{key}={value}" for key, value in sorted(labels)
    )
    return f"{name}{{{normalized_labels}}}"


def _series_labels(series: str) -> tuple[tuple[str, str], ...] | None:
    _, separator, labels = series.partition("{")
    if not separator or not labels.endswith("}"):
        return None
    body = labels[:-1]
    parsed: list[tuple[str, str]] = []
    position = 0
    while position < len(body):
        match = _LABEL.match(body, position)
        if match is None:
            return None
        parsed.append((match[1], match[2]))
        position = match.end()
        if position < len(body):
            if body[position] != ",":
                return None
            position += 1
    if len({key for key, _ in parsed}) != len(parsed):
        return None
    return tuple(parsed)


def summarize_window(before: str, after: str, elapsed_seconds: float) -> dict[str, Any]:
    """只对累积 count/sum/bucket/total 做差；窗口必须没有其他请求。"""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    try:
        previous, current = parse_samples(before), parse_samples(after)
    except ValueError:
        return {"valid": False, "reason": "duplicate_metric_series"}
    process_start_before = {
        key: value for key, value in previous.items()
        if key.split("{", 1)[0] == "process_start_time_seconds"
    }
    process_start_after = {
        key: value for key, value in current.items()
        if key.split("{", 1)[0] == "process_start_time_seconds"
    }
    if (process_start_before or process_start_after) and process_start_before != process_start_after:
        return {
            "valid": False,
            "reason": "server_process_changed_or_identity_missing",
            "process_start_time_seconds_before": process_start_before,
            "process_start_time_seconds_after": process_start_after,
        }
    selected = {}
    reset = []

    def is_cumulative(key: str) -> bool:
        name = key.split("{", 1)[0]
        return name.startswith("vllm:") and name.endswith(("_sum", "_count", "_bucket", "_total"))

    reset.extend(key for key in previous if is_cumulative(key) and key not in current)
    for key, value in current.items():
        if not is_cumulative(key):
            continue
        delta = value - previous.get(key, 0.0)
        if delta < 0:
            reset.append(key)
        selected[key] = delta
    if reset:
        return {"valid": False, "reason": "counter_reset_or_server_restart", "reset_samples": reset}

    def total(name: str) -> float | None:
        values = [v for k, v in selected.items() if k.split("{", 1)[0] == name]
        return sum(values) if values else None

    timings = {}
    for family in _TIMINGS:
        count, seconds = total("vllm:" + family + "_count"), total("vllm:" + family + "_sum")
        timings[family] = {
            "observation_count": count,
            "mean_ms": seconds * 1000 / count if seconds is not None and count is not None and count > 0 else None,
        }
    generated = total("vllm:generation_tokens_total")
    completed = total("vllm:request_success_total")
    finish_reason_counts: dict[str, float] = {}
    for series, delta in selected.items():
        if series.split("{", 1)[0] != "vllm:request_success_total":
            continue
        labels = _series_labels(series)
        reason = dict(labels).get("finished_reason") if labels is not None else None
        if reason is not None:
            decoded_reason = reason[1:-1]
            finish_reason_counts[decoded_reason] = (
                finish_reason_counts.get(decoded_reason, 0.0) + delta
            )
    reason_coverage = sum(finish_reason_counts.values()) if finish_reason_counts else None
    reason_coverage_complete = (
        reason_coverage == completed
        if reason_coverage is not None and completed is not None else None
    )
    return {
        "valid": bool(selected),
        "scope": "server_global_window_requires_exclusive_traffic",
        "elapsed_seconds": elapsed_seconds,
        "request_success_count": completed,
        "finish_reason_counts": finish_reason_counts or None,
        "finish_reason_coverage_complete": reason_coverage_complete,
        "non_stop_request_count": (
            sum(count for reason, count in finish_reason_counts.items() if reason != "stop")
            if finish_reason_counts else None
        ),
        "generated_tokens": generated,
        "requests_per_second": completed / elapsed_seconds if completed is not None else None,
        "output_tokens_per_second": generated / elapsed_seconds if generated is not None else None,
        "timings": timings,
        "counter_and_histogram_deltas": selected,
    }
