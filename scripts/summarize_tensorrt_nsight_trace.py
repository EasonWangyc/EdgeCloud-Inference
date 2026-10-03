"""汇总 Nsight Systems 导出的 TensorRT GPU trace，生成 kernel 归因摘要。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import sha256_file


_DURATION_FIELD = "Duration (ns)"
_NAME_FIELD = "Name"
_TRANSFER_CATEGORIES = {"memcpy", "memset"}
_CATEGORY_ORDER = (
    "int4_w4a16",
    "attention_fmha",
    "attention_xqa",
    "sampling",
    "embedding",
    "memcpy",
    "memset",
    "other_kernel",
)


def summarize_trace(path: Path | str, *, label: str | None = None) -> dict[str, Any]:
    """从 Nsight Systems CUDA GPU trace CSV 提取可审计的 kernel 统计。"""
    trace_path = Path(path)
    try:
        with trace_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = {_DURATION_FIELD, _NAME_FIELD} - fields
            if missing:
                raise ValueError(
                    f"Nsight trace is missing required columns: {sorted(missing)}"
                )
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"cannot read Nsight trace: {trace_path}") from error

    events: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        name = (row.get(_NAME_FIELD) or "").strip()
        raw_duration = (row.get(_DURATION_FIELD) or "").strip()
        if not name or not raw_duration:
            continue
        try:
            duration_ns = float(raw_duration)
        except ValueError as error:
            raise ValueError(
                f"invalid Nsight duration at row {row_number}: {raw_duration!r}"
            ) from error
        if duration_ns < 0:
            raise ValueError(f"negative Nsight duration at row {row_number}")
        events.append(
            {
                "name": name,
                "duration_ms": duration_ns / 1_000_000.0,
                "category": classify_kernel(name),
            }
        )

    categories = {
        category: _summary(
            [event["duration_ms"] for event in events if event["category"] == category]
        )
        for category in _CATEGORY_ORDER
    }
    kernel_events = [
        event for event in events if event["category"] not in _TRANSFER_CATEGORIES
    ]
    by_name: dict[tuple[str, str], list[float]] = {}
    for event in kernel_events:
        key = (event["category"], event["name"])
        by_name.setdefault(key, []).append(event["duration_ms"])
    top_kernels = []
    for (category, name), durations in by_name.items():
        item = _summary(durations)
        top_kernels.append(
            {
                "name": name,
                "category": category,
                **item,
            }
        )
    top_kernels.sort(key=lambda item: (-item["total_ms"], item["name"]))

    return {
        "schema_version": "parksight_tensorrt_nsight_trace_summary_v1",
        "input": {
            "path": str(trace_path),
            "sha256": sha256_file(trace_path),
            "label": label,
        },
        "trace": {
            "row_count": len(rows),
            "event_count": len(events),
            "trace_duration_ms": round(sum(event["duration_ms"] for event in events), 6),
            "kernel_duration_ms": round(sum(event["duration_ms"] for event in kernel_events), 6),
            "transfer_duration_ms": round(
                sum(
                    event["duration_ms"]
                    for event in events
                    if event["category"] in _TRANSFER_CATEGORIES
                ),
                6,
            ),
        },
        "categories": categories,
        "top_kernels": top_kernels[:50],
        "evidence_boundary": (
            "This summary aggregates GPU trace event durations. It does not map a kernel to a model "
            "layer, prove operator fusion, or replace matched E2E/decode benchmarks. Compare traces "
            "only when workload, engine, power mode and capture scope are matched."
        ),
    }


def classify_kernel(name: str) -> str:
    """按稳定的公开 kernel-name 片段分类；未知 kernel 保留为 other_kernel。"""
    lowered = name.lower()
    if "cuda memcpy" in lowered or "memcpy" in lowered:
        return "memcpy"
    if "memset" in lowered:
        return "memset"
    if "w4a16" in lowered or "w4a16" in lowered.replace(" ", ""):
        return "int4_w4a16"
    if "fmha" in lowered or "flash_attention" in lowered:
        return "attention_fmha"
    if "xqa" in lowered:
        return "attention_xqa"
    if "greedytop1" in lowered or "topkstage" in lowered or re.search(r"\btopk\b", lowered):
        return "sampling"
    if "embeddinglookup" in lowered:
        return "embedding"
    return "other_kernel"


def _summary(durations: list[float]) -> dict[str, Any]:
    if not durations:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "max_ms": None,
        }
    ordered = sorted(durations)
    return {
        "count": len(ordered),
        "total_ms": round(sum(ordered), 6),
        "mean_ms": round(sum(ordered) / len(ordered), 6),
        "p50_ms": round(_percentile(ordered, 0.50), 6),
        "p90_ms": round(_percentile(ordered, 0.90), 6),
        "max_ms": round(ordered[-1], 6),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label")
    args = parser.parse_args(argv)
    result = summarize_trace(args.trace, label=args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
