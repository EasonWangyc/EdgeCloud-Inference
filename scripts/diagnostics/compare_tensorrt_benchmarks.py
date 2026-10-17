"""比较两份 TensorRT 低层 benchmark 汇总的阶段时延和吞吐。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


_FIXED_METADATA_FIELDS = (
    "model_revision",
    "edge_llm_revision",
    "platform",
    "power_mode",
    "precision",
    "max_batch_size",
    "max_input_len",
    "max_kv_cache_capacity",
    "cuda_graph",
    "weight_streaming",
)
# These fields are absent from older LLM-only summaries, so they remain
# optional. When a visual-engine A/B supplies them, both sides must provide
# the same image-token profile; otherwise the comparison is not comparable.
_OPTIONAL_FIXED_METADATA_FIELDS = (
    "max_kv_pool_pages",
    "min_image_tokens",
    "max_image_tokens",
    "max_image_tokens_per_image",
)
_STAGES = (
    "vision_encode_ms",
    "model_generate_ms",
    "prefill_ms",
    "time_to_first_token_ms",
    "decode_ms",
    "end_to_end_ms",
    "http_round_trip_ms",
)
_RUN_METADATA_FIELDS = (
    "concurrency",
    "stream_responses",
    "reuse_http_connection",
)


def compare_benchmarks(
    baseline_path: Path | str,
    candidate_path: Path | str,
    *,
    min_repetitions: int = 3,
    min_p50_improvement: float = 0.10,
    min_p90_improvement: float = 0.05,
    changed_engine_component: str | None = None,
) -> dict[str, Any]:
    """返回 candidate 相对 baseline 的阶段级 improvement 和 speedup。"""
    baseline = _load_summary(baseline_path)
    candidate = _load_summary(candidate_path)
    baseline_metadata = _mapping(baseline.get("metadata"), "baseline.metadata")
    candidate_metadata = _mapping(candidate.get("metadata"), "candidate.metadata")
    metadata_comparison = _compare_metadata(baseline_metadata, candidate_metadata)
    runtime_comparison = _compare_runtime_metadata(
        baseline_metadata.get("runtime_metadata"),
        candidate_metadata.get("runtime_metadata"),
    )
    component_swap = _compare_engine_component_swap(
        runtime_comparison,
        changed_engine_component,
    )
    if changed_engine_component is not None and not component_swap["comparable"]:
        metadata_comparison = {
            **metadata_comparison,
            "status": "mismatch" if component_swap["status"] == "mismatch" else "unverified",
            "comparable": False,
        }
    run_comparison = _compare_run_metadata(baseline_metadata, candidate_metadata)

    baseline_execution = _mapping(baseline.get("execution"), "baseline.execution")
    candidate_execution = _mapping(candidate.get("execution"), "candidate.execution")
    baseline_stages = _mapping(
        baseline_execution.get("stage_latency_ms"),
        "baseline.execution.stage_latency_ms",
    )
    candidate_stages = _mapping(
        candidate_execution.get("stage_latency_ms"),
        "candidate.execution.stage_latency_ms",
    )
    stage_comparison: dict[str, Any] = {}
    for stage in _STAGES:
        left = _optional_summary(baseline_stages.get(stage))
        right = _optional_summary(candidate_stages.get(stage))
        stage_comparison[stage] = _compare_summary(left, right)
    repetition_comparison = _compare_repetitions(
        baseline_execution.get("repetitions"),
        candidate_execution.get("repetitions"),
    )
    repetition_gate = _repetition_gate(
        repetition_comparison,
        min_repetitions=min_repetitions,
        min_p50_improvement=min_p50_improvement,
        min_p90_improvement=min_p90_improvement,
    )

    return {
        "schema_version": "parksight_tensorrt_benchmark_comparison_v1",
        "inputs": {"baseline": str(Path(baseline_path)), "candidate": str(Path(candidate_path))},
        "match": metadata_comparison,
        "runtime": runtime_comparison,
        "component_swap": component_swap,
        "execution": {
            "baseline_sample_count": baseline_execution.get("sample_count"),
            "candidate_sample_count": candidate_execution.get("sample_count"),
            "baseline_completed_sample_count": baseline_execution.get("completed_sample_count"),
            "candidate_completed_sample_count": candidate_execution.get("completed_sample_count"),
            "stage_latency_ms": stage_comparison,
            "repetitions": repetition_comparison,
            "repetition_gate": repetition_gate,
            "decode_tokens_per_second": _compare_rate(
                baseline_execution.get("decode_tokens_per_second"),
                candidate_execution.get("decode_tokens_per_second"),
            ),
            "end_to_end_tokens_per_second": _compare_rate(
                baseline_execution.get("end_to_end_tokens_per_second"),
                candidate_execution.get("end_to_end_tokens_per_second"),
            ),
            "aggregate_output_tokens_per_second": _compare_rate(
                _run_metric(baseline_execution, baseline_metadata, "aggregate_output_tokens_per_second"),
                _run_metric(candidate_execution, candidate_metadata, "aggregate_output_tokens_per_second"),
            ),
        },
        "run": run_comparison,
        "evidence_boundary": (
            "Improvement is computed from matched benchmark summary fields. It does not identify "
            "the responsible operator or tactic; use Inspector/Nsight evidence for attribution. "
            "Missing metadata means the A/B match is unverified."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-repetitions", type=int, default=3)
    parser.add_argument("--min-p50-improvement", type=float, default=0.10)
    parser.add_argument("--min-p90-improvement", type=float, default=0.05)
    parser.add_argument(
        "--changed-engine-component",
        choices=("llm", "visual"),
        help="严格校验仅该 engine 发生替换，适用于 component-level A/B",
    )
    args = parser.parse_args(argv)
    if args.min_repetitions <= 0:
        parser.error("--min-repetitions must be positive")
    if not 0 <= args.min_p50_improvement <= 1:
        parser.error("--min-p50-improvement must be between 0 and 1")
    if not 0 <= args.min_p90_improvement <= 1:
        parser.error("--min-p90-improvement must be between 0 and 1")
    result = compare_benchmarks(
        args.baseline,
        args.candidate,
        min_repetitions=args.min_repetitions,
        min_p50_improvement=args.min_p50_improvement,
        min_p90_improvement=args.min_p90_improvement,
        changed_engine_component=args.changed_engine_component,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


def _load_summary(path: Path | str) -> Mapping[str, Any]:
    summary_path = Path(path)
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid TensorRT benchmark summary: {summary_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark summary must be an object: {summary_path}")
    if payload.get("schema_version") != "parksight_tensorrt_benchmark_v1":
        raise ValueError(f"unsupported benchmark summary schema: {summary_path}")
    return payload


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _lookup(payload: Mapping[str, Any], field: str) -> Any:
    if field in payload:
        return payload[field]
    for value in payload.values():
        if isinstance(value, Mapping):
            found = _lookup(value, field)
            if found is not None:
                return found
    return None


def _compare_metadata(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    compared: dict[str, Any] = {}
    mismatches: dict[str, Any] = {}
    unverified_fields: list[str] = []
    for field in _FIXED_METADATA_FIELDS:
        left = _lookup(baseline, field)
        right = _lookup(candidate, field)
        compared[field] = {"baseline": left, "candidate": right}
        if left != right:
            mismatches[field] = {"baseline": left, "candidate": right}
        elif left is None:
            unverified_fields.append(field)
    for field in _OPTIONAL_FIXED_METADATA_FIELDS:
        left = _lookup(baseline, field)
        right = _lookup(candidate, field)
        if left is None and right is None:
            continue
        compared[field] = {"baseline": left, "candidate": right}
        if left != right:
            mismatches[field] = {"baseline": left, "candidate": right}
            if left is None or right is None:
                unverified_fields.append(field)
    if mismatches:
        status = "mismatch"
    elif unverified_fields:
        status = "unverified"
    else:
        status = "matched"
    return {
        "status": status,
        "comparable": status == "matched",
        "compared_fields": compared,
        "mismatches": mismatches,
        "unverified_fields": unverified_fields,
    }


def _compare_runtime_metadata(baseline: Any, candidate: Any) -> dict[str, Any]:
    """单独报告 runtime A/B 的 engine 身份和开关差异。"""
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return {
            "status": "unverified",
            "same_engine": None,
            "changed_options": {},
            "evidence_boundary": "runtime metadata is missing from one or both summaries",
        }
    baseline_engines = baseline.get("engines")
    candidate_engines = candidate.get("engines")
    baseline_hashes = _engine_hashes(baseline_engines)
    candidate_hashes = _engine_hashes(candidate_engines)
    complete_engine_identity = _complete_engine_hashes(baseline_hashes) and _complete_engine_hashes(
        candidate_hashes
    )
    same_engine = (
        baseline_hashes == candidate_hashes if complete_engine_identity else None
    )
    baseline_options = baseline.get("options")
    candidate_options = candidate.get("options")
    if not isinstance(baseline_options, Mapping) or not isinstance(candidate_options, Mapping):
        return {
            "status": "unverified",
            "same_engine": same_engine,
            "changed_options": {},
            "evidence_boundary": "runtime metadata options are missing or malformed",
        }
    changed_options = {
        key: {"baseline": baseline_options.get(key), "candidate": candidate_options.get(key)}
        for key in sorted(set(baseline_options) | set(candidate_options))
        if baseline_options.get(key) != candidate_options.get(key)
    }
    status = (
        "matched"
        if same_engine is True and not changed_options
        else "changed"
        if same_engine is True
        else "unverified"
    )
    return {
        "status": status,
        "same_engine": same_engine,
        "changed_options": changed_options,
        "baseline_engine_sha256": baseline_hashes,
        "candidate_engine_sha256": candidate_hashes,
        "evidence_boundary": (
            "unverified means one or both runtime metadata objects lack complete llm/visual "
            "engine SHA-256 identity; changed means the runtime A/B intentionally differs "
            "in engine identity or options; neither status by itself proves a performance cause"
        ),
    }


def _compare_engine_component_swap(
    runtime_comparison: Mapping[str, Any],
    changed_component: str | None,
) -> dict[str, Any]:
    """验证 component-level A/B 是否只替换了指定 engine。"""
    if changed_component is None:
        return {
            "status": "not_requested",
            "comparable": True,
            "changed_component": None,
            "changed_components": [],
        }
    baseline_hashes = runtime_comparison.get("baseline_engine_sha256")
    candidate_hashes = runtime_comparison.get("candidate_engine_sha256")
    if not isinstance(baseline_hashes, Mapping) or not isinstance(candidate_hashes, Mapping):
        return {
            "status": "unverified",
            "comparable": False,
            "changed_component": changed_component,
            "changed_components": [],
            "reason": "runtime metadata must contain both llm and visual engine SHA-256 values",
        }
    missing = [
        component
        for component in ("llm", "visual")
        if not baseline_hashes.get(component) or not candidate_hashes.get(component)
    ]
    if missing:
        return {
            "status": "unverified",
            "comparable": False,
            "changed_component": changed_component,
            "changed_components": [],
            "reason": f"missing engine SHA-256 for: {', '.join(missing)}",
        }
    changed_components = [
        component
        for component in ("llm", "visual")
        if baseline_hashes[component] != candidate_hashes[component]
    ]
    correct_change = changed_component in changed_components
    only_requested_changed = changed_components == [changed_component]
    no_runtime_option_change = not runtime_comparison.get("changed_options")
    comparable = correct_change and only_requested_changed and no_runtime_option_change
    status = "matched" if comparable else "mismatch"
    return {
        "status": status,
        "comparable": comparable,
        "changed_component": changed_component,
        "changed_components": changed_components,
        "llm_engine_unchanged": "llm" not in changed_components,
        "visual_engine_unchanged": "visual" not in changed_components,
        "runtime_options_unchanged": no_runtime_option_change,
        "reason": None if comparable else "engine swap is not limited to the requested component",
    }


def _compare_run_metadata(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """单独报告并发度等运行变量，不把它们误判为 engine fixed mismatch。"""
    compared: dict[str, Any] = {}
    changed: dict[str, Any] = {}
    for field in _RUN_METADATA_FIELDS:
        left = _lookup(baseline, field)
        right = _lookup(candidate, field)
        if left is None and right is None:
            continue
        compared[field] = {"baseline": left, "candidate": right}
        if left != right:
            changed[field] = {"baseline": left, "candidate": right}
    if not compared:
        status = "unverified"
    elif changed:
        status = "changed"
    elif len(compared) == len(_RUN_METADATA_FIELDS):
        status = "matched"
    else:
        status = "unverified"
    return {
        "status": status,
        "compared_fields": compared,
        "changed_fields": changed,
        "evidence_boundary": (
            "run metadata describes concurrency and transport variables; changed values are "
            "intentional A/B dimensions, not proof of a TensorRT kernel cause"
        ),
    }


def _run_metric(
    execution: Mapping[str, Any], metadata: Mapping[str, Any], field: str
) -> Any:
    """兼容新 summary execution 字段及旧 summary metadata 字段。"""
    value = execution.get(field)
    return value if value is not None else _lookup(metadata, field)


def _engine_hashes(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for name in ("llm", "visual"):
        engine = value.get(name)
        if isinstance(engine, Mapping):
            result[name] = engine.get("sha256")
    return result if result else None


def _complete_engine_hashes(value: Mapping[str, Any] | None) -> bool:
    """只有同时具备 llm/visual SHA-256 才能证明 runtime engine 身份。"""
    if value is None or set(value) != {"llm", "visual"}:
        return False
    return all(
        isinstance(value[name], str) and bool(value[name].strip())
        for name in ("llm", "visual")
    )


def _optional_summary(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _compare_summary(
    baseline: Mapping[str, Any] | None, candidate: Mapping[str, Any] | None
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for percentile in ("p50", "p90", "p99"):
        result[percentile] = _compare_scalar(
            baseline.get(percentile) if baseline else None,
            candidate.get(percentile) if candidate else None,
        )
    result["baseline_count"] = baseline.get("count") if baseline else 0
    result["candidate_count"] = candidate.get("count") if candidate else 0
    return result


def _compare_repetitions(baseline: Any, candidate: Any) -> dict[str, Any]:
    """逐 repetition 比较阶段分位数，保留缺失 repetition 的证据边界。"""
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        return {"status": "unverified", "comparisons": {}}
    comparisons: dict[str, Any] = {}
    all_ids = sorted(set(baseline) | set(candidate), key=str)
    for repetition in all_ids:
        left = baseline.get(repetition)
        right = candidate.get(repetition)
        left_stages = left.get("stage_latency_ms", {}) if isinstance(left, Mapping) else {}
        right_stages = right.get("stage_latency_ms", {}) if isinstance(right, Mapping) else {}
        comparisons[str(repetition)] = {
            "baseline_sample_count": left.get("sample_count") if isinstance(left, Mapping) else None,
            "candidate_sample_count": right.get("sample_count") if isinstance(right, Mapping) else None,
            "stage_latency_ms": {
                stage: _compare_summary(
                    _optional_summary(left_stages.get(stage)),
                    _optional_summary(right_stages.get(stage)),
                )
                for stage in _STAGES
            },
        }
    return {
        "status": "matched" if set(baseline) == set(candidate) else "unverified",
        "comparisons": comparisons,
    }


def _repetition_gate(
    comparison: Mapping[str, Any],
    *,
    min_repetitions: int,
    min_p50_improvement: float,
    min_p90_improvement: float,
) -> dict[str, Any]:
    """检查每次重复的 E2E p50/p90 是否分别达到门槛。"""
    if comparison.get("status") != "matched":
        return {
            "status": "unverified",
            "eligible": False,
            "required_repetitions": min_repetitions,
            "min_p50_improvement": min_p50_improvement,
            "min_p90_improvement": min_p90_improvement,
            "reason": "matching repetition summaries are required",
        }
    repetition_items = comparison.get("comparisons", {})
    if not isinstance(repetition_items, Mapping) or len(repetition_items) < min_repetitions:
        return {
            "status": "unverified",
            "eligible": False,
            "required_repetitions": min_repetitions,
            "observed_repetitions": len(repetition_items) if isinstance(repetition_items, Mapping) else 0,
            "min_p50_improvement": min_p50_improvement,
            "min_p90_improvement": min_p90_improvement,
            "reason": "not enough repetition summaries",
        }
    checks: dict[str, Any] = {}
    eligible = True
    for repetition, item in repetition_items.items():
        stages = item.get("stage_latency_ms", {}) if isinstance(item, Mapping) else {}
        e2e = stages.get("end_to_end_ms") if isinstance(stages, Mapping) else None
        p50 = e2e.get("p50", {}) if isinstance(e2e, Mapping) else {}
        p90 = e2e.get("p90", {}) if isinstance(e2e, Mapping) else {}
        p50_improvement = p50.get("improvement") if isinstance(p50, Mapping) else None
        p90_improvement = p90.get("improvement") if isinstance(p90, Mapping) else None
        p50_pass = _is_number(p50_improvement) and p50_improvement >= min_p50_improvement
        p90_pass = _is_number(p90_improvement) and p90_improvement >= min_p90_improvement
        checks[str(repetition)] = {
            "p50_improvement": p50_improvement,
            "p90_improvement": p90_improvement,
            "p50_pass": p50_pass,
            "p90_pass": p90_pass,
            "eligible": p50_pass and p90_pass,
        }
        eligible = eligible and p50_pass and p90_pass
    return {
        "status": "pass" if eligible else "fail",
        "eligible": eligible,
        "required_repetitions": min_repetitions,
        "observed_repetitions": len(repetition_items),
        "min_p50_improvement": min_p50_improvement,
        "min_p90_improvement": min_p90_improvement,
        "per_repetition": checks,
    }


def _compare_scalar(baseline: Any, candidate: Any) -> dict[str, Any]:
    if not _is_number(baseline) or not _is_number(candidate):
        return {
            "baseline": baseline,
            "candidate": candidate,
            "speedup": None,
            "improvement": None,
        }
    speedup = baseline / candidate if candidate > 0 else None
    improvement = 1.0 - candidate / baseline if baseline > 0 else None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "speedup": speedup,
        "improvement": improvement,
    }


def _compare_rate(baseline: Any, candidate: Any) -> dict[str, Any]:
    """比较越大越好的速率，speedup 为 candidate / baseline。"""
    if not _is_number(baseline) or not _is_number(candidate):
        return {
            "baseline": baseline,
            "candidate": candidate,
            "speedup": None,
            "improvement": None,
        }
    speedup = candidate / baseline if baseline > 0 else None
    improvement = candidate / baseline - 1.0 if baseline > 0 else None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "speedup": speedup,
        "improvement": improvement,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
