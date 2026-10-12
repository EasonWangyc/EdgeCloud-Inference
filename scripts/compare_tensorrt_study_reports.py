"""比较两份 TensorRT StudyReport 的阶段性能、资源和质量指标。"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_IDENTITY_FIELDS = (
    "workload_identity",
    "split",
    "power_mode",
    "client_concurrency",
    "load_pattern",
    "repetitions",
)
_RUNTIME_FIELDS = (
    "backend",
    "backend_revision",
    "model_id",
    "model_revision",
    "adapter_revision",
    "precision",
)
_STAGES = (
    "vision_encode_ms",
    "model_generate_ms",
    "prefill_ms",
    "time_to_first_token_ms",
    "decode_ms",
    "end_to_end_ms",
)
_QUALITY_FIELDS = (
    "json_validity_rate",
    "risk_level_accuracy",
    "event_micro_f1",
)
_PERFORMANCE_FIELDS = (
    "tokens_per_second",
    "wall_output_tokens_per_second",
    "decode_tokens_per_second",
    "aggregate_output_tokens_per_end_to_end_second",
    "peak_memory_mb",
    "average_power_w",
    "peak_temperature_c",
)


def compare_study_reports(
    baseline_path: Path | str,
    candidate_path: Path | str,
) -> dict[str, Any]:
    """Return an auditable comparison for two StudyReport JSON files."""
    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    baseline_identity = _mapping(baseline.get("study_identity"), "baseline.study_identity")
    candidate_identity = _mapping(candidate.get("study_identity"), "candidate.study_identity")
    baseline_performance = _mapping(
        baseline.get("performance_metrics"), "baseline.performance_metrics"
    )
    candidate_performance = _mapping(
        candidate.get("performance_metrics"), "candidate.performance_metrics"
    )
    baseline_quality = _mapping(baseline.get("quality_metrics"), "baseline.quality_metrics")
    candidate_quality = _mapping(candidate.get("quality_metrics"), "candidate.quality_metrics")

    identity = _compare_fields(baseline_identity, candidate_identity, _IDENTITY_FIELDS)
    baseline_runtime = _mapping(
        baseline_identity.get("runtime_identity"),
        "baseline.study_identity.runtime_identity",
    )
    candidate_runtime = _mapping(
        candidate_identity.get("runtime_identity"),
        "candidate.study_identity.runtime_identity",
    )
    runtime = _compare_fields(baseline_runtime, candidate_runtime, _RUNTIME_FIELDS)
    stages = {
        stage: _compare_summary(
            baseline_performance.get("stage_latency_ms", {}).get(stage)
            if isinstance(baseline_performance.get("stage_latency_ms"), Mapping)
            else None,
            candidate_performance.get("stage_latency_ms", {}).get(stage)
            if isinstance(candidate_performance.get("stage_latency_ms"), Mapping)
            else None,
        )
        for stage in _STAGES
    }
    quality = {
        field: _compare_scalar(baseline_quality.get(field), candidate_quality.get(field), higher_is_better=True)
        for field in _QUALITY_FIELDS
    }
    performance = {
        field: _compare_scalar(
            baseline_performance.get(field), candidate_performance.get(field),
            higher_is_better="tokens_per" in field,
        )
        for field in _PERFORMANCE_FIELDS
    }
    baseline_count = baseline_performance.get("backend_completed_sample_count")
    candidate_count = candidate_performance.get("backend_completed_sample_count")
    records = compare_records(baseline.get("records"), candidate.get("records"))
    environment = _compare_fields(
        baseline.get("environment_snapshot") or {}, candidate.get("environment_snapshot") or {},
        ("hostname", "machine", "actual_power_mode", "measurement_mode", "input_image_sha256"),
    )
    return {
        "schema_version": "parksight_tensorrt_study_comparison_v2",
        "inputs": {
            "baseline": str(Path(baseline_path)),
            "candidate": str(Path(candidate_path)),
            "baseline_sha256": hashlib.sha256(Path(baseline_path).read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest(),
        },
        "match": {
            "identity": identity,
            "runtime": runtime,
            "records": records,
            "environment": environment,
            "comparable": (identity["comparable"] and runtime["comparable"]
                           and records["workload_comparable"] and environment["comparable"]),
        },
        "execution": {
            "baseline_repetitions": baseline_identity.get("repetitions"),
            "candidate_repetitions": candidate_identity.get("repetitions"),
            "baseline_backend_completed_sample_count": baseline_count,
            "candidate_backend_completed_sample_count": candidate_count,
            "stage_latency_ms": stages,
            "performance": performance,
        },
        "quality": quality,
        "evidence_boundary": (
            "Comparable means serialized workload, runtime, environment and case/input coverage match; "
            "exact_output_match is a separate observation. Neither proves engine weight provenance "
            "or authorizes promotion. This does not attribute a gain to "
            "an operator, tactic, kernel, KV-cache layout, or CUDA Graph; use Engine Inspector "
            "and Nsight evidence for attribution. Missing fields remain unverified."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; use a fresh comparison path")
    result = compare_study_reports(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)
    return 0


def _load_report(path: Path | str) -> Mapping[str, Any]:
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid TensorRT StudyReport: {report_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"TensorRT StudyReport must be an object: {report_path}")
    return payload


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _compare_fields(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    compared = {
        field: {"baseline": baseline.get(field), "candidate": candidate.get(field)}
        for field in fields
    }
    mismatches = {
        field: values
        for field, values in compared.items()
        if values["baseline"] != values["candidate"]
    }
    missing = [
        field
        for field, values in compared.items()
        if values["baseline"] is None or values["candidate"] is None
    ]
    return {
        "status": "mismatch" if mismatches else ("unverified" if missing else "matched"),
        "comparable": not mismatches and not missing,
        "compared_fields": compared,
        "mismatches": mismatches,
        "missing_fields": missing,
    }


def _compare_summary(baseline: Any, candidate: Any) -> dict[str, Any]:
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return {
            "status": "unverified",
            "p50": _compare_scalar(None, None),
            "p90": _compare_scalar(None, None),
            "p99": _compare_scalar(None, None),
            "baseline_count": baseline.get("count") if isinstance(baseline, Mapping) else None,
            "candidate_count": candidate.get("count") if isinstance(candidate, Mapping) else None,
        }
    result = {
        percentile: _compare_scalar(baseline.get(percentile), candidate.get(percentile))
        for percentile in ("p50", "p90", "p99")
    }
    result["status"] = (
        "matched"
        if all(item["improvement"] is not None for item in result.values())
        else "unverified"
    )
    result["baseline_count"] = baseline.get("count")
    result["candidate_count"] = candidate.get("count")
    return result


def _compare_scalar(baseline: Any, candidate: Any, *, higher_is_better: bool = False) -> dict[str, Any]:
    if not _is_number(baseline) or not _is_number(candidate):
        return {
            "baseline": baseline,
            "candidate": candidate,
            "improvement": None,
            "speedup": None,
        }
    baseline_float = float(baseline)
    candidate_float = float(candidate)
    improvement = (
        (candidate_float / baseline_float - 1.0 if higher_is_better
         else 1.0 - candidate_float / baseline_float)
        if baseline_float > 0 else None
    )
    speedup = (
        candidate_float / baseline_float if baseline_float > 0 else None
    ) if higher_is_better else (
        baseline_float / candidate_float if candidate_float > 0 else None
    )
    return {
        "baseline": baseline_float,
        "candidate": candidate_float,
        "higher_is_better": higher_is_better,
        "improvement": improvement,
        "speedup": speedup,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_records(baseline: Any, candidate: Any) -> dict[str, Any]:
    """Compare repeated case coverage and inputs independently of generated outputs."""
    fields = ("workload_identity", "input_tokens", "raw_output", "assessment",
              "output_tokens", "decode_tokens", "failure")
    result: dict[str, Any] = {"status": "unverified", "workload_comparable": False,
                             "exact_output_match": False, "mismatches": {}, "missing_fields": []}
    if not isinstance(baseline, list) or not isinstance(candidate, list) or not baseline or not candidate:
        result["missing_fields"] = ["nonempty records"]
        return result
    if any(not isinstance(row, Mapping) or not isinstance(row.get("case_id"), str)
           or not row["case_id"] for row in baseline + candidate):
        raise ValueError("records must contain nonempty string case_id values")
    coverage = [Counter(row["case_id"] for row in rows) for rows in (baseline, candidate)]
    result["case_counts"] = {"baseline": dict(coverage[0]), "candidate": dict(coverage[1])}
    if coverage[0] != coverage[1]:
        result["status"] = "mismatch"
        result["mismatches"] = {"case_coverage": result["case_counts"]}
        return result
    # Multisets retain repetitions while allowing harmless record reordering.
    for field in fields:
        missing = any(field not in row or (row[field] is None and field != "failure")
                      for row in baseline + candidate)
        if missing:
            result["missing_fields"].append(field)
            continue
        counts = [Counter((row["case_id"], json.dumps(row[field], sort_keys=True, ensure_ascii=False))
                          for row in rows) for rows in (baseline, candidate)]
        if counts[0] != counts[1]:
            result["mismatches"][field] = sorted({case for case, _ in (counts[0] - counts[1]) | (counts[1] - counts[0])})
    result["workload_comparable"] = all(
        field not in result["missing_fields"] and field not in result["mismatches"]
        for field in ("workload_identity", "input_tokens")
    )
    result["exact_output_match"] = not result["missing_fields"] and not result["mismatches"]
    if result["exact_output_match"]:
        complete = [Counter(json.dumps({key: row[key] for key in ("case_id", *fields)},
                                       sort_keys=True, ensure_ascii=False) for row in rows)
                    for rows in (baseline, candidate)]
        if complete[0] != complete[1]:
            result["mismatches"]["record_content"] = "field association differs across repetitions"
            result["exact_output_match"] = False
    result["status"] = "mismatch" if result["mismatches"] else ("unverified" if result["missing_fields"] else "matched")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
