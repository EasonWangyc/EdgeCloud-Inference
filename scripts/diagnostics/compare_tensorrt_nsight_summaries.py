"""比较两份 TensorRT Nsight GPU trace 汇总，输出可审计的耗时差异。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = "parksight_tensorrt_nsight_trace_summary_v1"
_TRACE_METRICS = (
    "trace_duration_ms",
    "kernel_duration_ms",
    "transfer_duration_ms",
)
_CATEGORY_METRICS = (
    "count",
    "total_ms",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "max_ms",
)


def compare_summaries(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """计算 candidate 相对 baseline 的 trace/category/kernel 差异。"""
    _validate_summary(baseline, "baseline")
    _validate_summary(candidate, "candidate")
    if baseline.get("classification_version") != candidate.get("classification_version"):
        raise ValueError("kernel classification versions differ; re-summarize both raw traces")

    baseline_label = baseline.get("input", {}).get("label") or "baseline"
    candidate_label = candidate.get("input", {}).get("label") or "candidate"
    trace_delta = _metric_delta(
        baseline.get("trace", {}), candidate.get("trace", {}), _TRACE_METRICS
    )

    categories: dict[str, Any] = {}
    category_names = list(
        dict.fromkeys(
            [*baseline.get("categories", {}).keys(), *candidate.get("categories", {}).keys()]
        )
    )
    for category in category_names:
        categories[category] = _metric_delta(
            baseline.get("categories", {}).get(category, {}),
            candidate.get("categories", {}).get(category, {}),
            _CATEGORY_METRICS,
        )

    baseline_kernels = {
        (item.get("category"), item.get("name")): item
        for item in baseline.get("top_kernels", [])
    }
    candidate_kernels = {
        (item.get("category"), item.get("name")): item
        for item in candidate.get("top_kernels", [])
    }
    kernel_changes = []
    for key in sorted(set(baseline_kernels) | set(candidate_kernels), key=str):
        before = baseline_kernels.get(key, {})
        after = candidate_kernels.get(key, {})
        kernel_changes.append(
            {
                "category": key[0],
                "name": key[1],
                "baseline_total_ms": before.get("total_ms", 0.0),
                "candidate_total_ms": after.get("total_ms", 0.0),
                "delta": _delta(
                    before.get("total_ms", 0.0), after.get("total_ms", 0.0)
                ),
            }
        )
    kernel_changes.sort(
        key=lambda item: (
            -max(item["baseline_total_ms"], item["candidate_total_ms"]),
            item["category"],
            item["name"],
        )
    )

    return {
        "schema_version": "parksight_tensorrt_nsight_trace_comparison_v1",
        "comparison": {
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "baseline_path": baseline.get("input", {}).get("path"),
            "candidate_path": candidate.get("input", {}).get("path"),
        },
        "trace": trace_delta,
        "categories": categories,
        "top_kernel_changes": kernel_changes[:50],
        "evidence_boundary": (
            "Deltas are candidate minus baseline for matched Nsight GPU trace summaries. "
            "Negative duration is faster. This does not replace matched TTFT/decode/E2E, "
            "quality, memory, power or soak evidence, and does not prove operator fusion."
        ),
    }


def _validate_summary(summary: dict[str, Any], name: str) -> None:
    if summary.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"{name} is not a {_SCHEMA_VERSION} summary"
        )
    for field in ("trace", "categories", "top_kernels"):
        if field not in summary:
            raise ValueError(f"{name} summary is missing {field!r}")


def _metric_delta(
    baseline: dict[str, Any], candidate: dict[str, Any], metrics: tuple[str, ...]
) -> dict[str, Any]:
    return {metric: _delta(baseline.get(metric), candidate.get(metric)) for metric in metrics}


def _delta(baseline: Any, candidate: Any) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {"baseline": baseline, "candidate": candidate, "absolute": None, "relative_percent": None}
    absolute = candidate - baseline
    relative = None if baseline == 0 else (absolute / baseline) * 100.0
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute": round(absolute, 6),
        "relative_percent": None if relative is None else round(relative, 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_summaries(baseline, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
