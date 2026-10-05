"""vLLM Prometheus 窗口差分；保留直方图桶，不伪造逐请求计时。"""

from __future__ import annotations

import math
import re
from typing import Any


_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{.*\})?)\s+([^\s]+)(?:\s+[^\s]+)?$')
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
            samples[match[1]] = value
    return samples


def summarize_window(before: str, after: str, elapsed_seconds: float) -> dict[str, Any]:
    """只对累积 count/sum/bucket/total 做差；窗口必须没有其他请求。"""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    previous, current = parse_samples(before), parse_samples(after)
    selected = {}
    reset = []
    for key, value in current.items():
        name = key.split("{", 1)[0]
        if not name.startswith("vllm:") or not name.endswith(("_sum", "_count", "_bucket", "_total")):
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
    return {
        "valid": bool(selected),
        "scope": "server_global_window_requires_exclusive_traffic",
        "elapsed_seconds": elapsed_seconds,
        "request_success_count": completed,
        "generated_tokens": generated,
        "requests_per_second": completed / elapsed_seconds if completed is not None else None,
        "output_tokens_per_second": generated / elapsed_seconds if generated is not None else None,
        "timings": timings,
        "counter_and_histogram_deltas": selected,
    }
