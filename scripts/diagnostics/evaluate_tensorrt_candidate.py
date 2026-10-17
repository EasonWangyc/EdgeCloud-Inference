"""判断 TensorRT 候选 StudyReport 是否满足晋级门禁。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import (
    TensorRTTuningConfig,
    TensorRTValidationError,
    write_json,
)


_PERFORMANCE_STAGES = (
    "prefill_ms",
    "time_to_first_token_ms",
    "decode_ms",
    "end_to_end_ms",
)


def evaluate_candidate(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    soak: Mapping[str, Any] | None = None,
    min_repetitions: int = 3,
    min_p50_improvement: float = 0.10,
    min_p90_improvement: float = 0.05,
    max_memory_regression: float = 0.05,
    require_backend_completion: bool = True,
    require_json_validity: bool = True,
    require_quality_non_regression: bool = True,
    min_soak_samples: int = 100,
    baseline_jetson_summary: Mapping[str, Any] | None = None,
    candidate_jetson_summary: Mapping[str, Any] | None = None,
    candidate_runtime_log_summary: Mapping[str, Any] | None = None,
    candidate_runtime_metadata: Mapping[str, Any] | None = None,
    benchmark_comparison: Mapping[str, Any] | None = None,
    require_paged_kv_active: bool = False,
    require_fmha_tiled_active: bool = False,
) -> dict[str, Any]:
    """比较候选报告，并严格按配置启用相应的晋级门禁。"""
    reasons: list[str] = []
    baseline_identity = _mapping(baseline.get("study_identity"), "baseline.study_identity")
    candidate_identity = _mapping(candidate.get("study_identity"), "candidate.study_identity")
    baseline_quality = _mapping(baseline.get("quality_metrics"), "baseline.quality_metrics")
    candidate_quality = _mapping(candidate.get("quality_metrics"), "candidate.quality_metrics")
    baseline_performance = _mapping(baseline.get("performance_metrics"), "baseline.performance_metrics")
    candidate_performance = _mapping(candidate.get("performance_metrics"), "candidate.performance_metrics")

    fixed_fields = ("workload_identity", "split", "power_mode")
    mismatches = {
        field: (baseline_identity.get(field), candidate_identity.get(field))
        for field in fixed_fields
        if baseline_identity.get(field) != candidate_identity.get(field)
    }
    if mismatches:
        reasons.append(f"fixed experiment identity mismatch: {mismatches}")

    benchmark_gate_ok = None
    if benchmark_comparison is not None:
        benchmark_execution = _mapping(
            benchmark_comparison.get("execution"),
            "benchmark_comparison.execution",
        )
        benchmark_gate = _mapping(
            benchmark_execution.get("repetition_gate"),
            "benchmark_comparison.execution.repetition_gate",
        )
        benchmark_gate_ok = benchmark_gate.get("eligible") is True
        if not benchmark_gate_ok:
            reasons.append(
                "low-level benchmark repetition gate is not eligible: "
                f"{benchmark_gate.get('status', 'missing')}"
            )

    baseline_runtime = baseline_identity.get("runtime_identity")
    candidate_runtime = candidate_identity.get("runtime_identity")
    if baseline_runtime is None or candidate_runtime is None:
        reasons.append("runtime identity evidence is required for promotion")
    else:
        baseline_runtime_mapping = _mapping(
            baseline_runtime, "baseline.study_identity.runtime_identity"
        )
        candidate_runtime_mapping = _mapping(
            candidate_runtime, "candidate.study_identity.runtime_identity"
        )
        runtime_fields = (
            "backend",
            "backend_revision",
            "model_id",
            "model_revision",
            "adapter_revision",
            "precision",
        )
        runtime_mismatches = {
            field: (
                baseline_runtime_mapping.get(field),
                candidate_runtime_mapping.get(field),
            )
            for field in runtime_fields
            if baseline_runtime_mapping.get(field)
            != candidate_runtime_mapping.get(field)
        }
        if runtime_mismatches:
            reasons.append(f"runtime identity mismatch: {runtime_mismatches}")

    repetitions = candidate_identity.get("repetitions")
    baseline_repetitions = baseline_identity.get("repetitions")
    if not isinstance(baseline_repetitions, int) or baseline_repetitions < min_repetitions:
        reasons.append(f"baseline repetitions must be at least {min_repetitions}")
    if not isinstance(repetitions, int) or repetitions < min_repetitions:
        reasons.append(f"candidate repetitions must be at least {min_repetitions}")

    completion = _completion_rate(candidate, candidate_performance)
    if require_backend_completion and completion != 1.0:
        reasons.append(f"candidate backend completion rate is {completion}")
    json_validity = _number(candidate_quality.get("json_validity_rate"), "candidate.quality_metrics.json_validity_rate")
    if require_json_validity and json_validity != 1.0:
        reasons.append(f"candidate JSON validity rate is {json_validity}")

    quality_comparison: dict[str, dict[str, float]] = {}
    for field in ("risk_level_accuracy", "event_micro_f1"):
        baseline_value = _number(baseline_quality.get(field), f"baseline.quality_metrics.{field}")
        candidate_value = _number(candidate_quality.get(field), f"candidate.quality_metrics.{field}")
        quality_comparison[field] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
        }
        if require_quality_non_regression and candidate_value < baseline_value:
            reasons.append(f"quality regression in {field}: {baseline_value} -> {candidate_value}")

    baseline_e2e = _stage_p50_p90(baseline_performance, "baseline")
    candidate_e2e = _stage_p50_p90(candidate_performance, "candidate")
    stage_comparison = _compare_performance_stages(
        baseline_performance,
        candidate_performance,
    )
    p50_improvement = 1.0 - candidate_e2e[0] / baseline_e2e[0]
    p90_improvement = 1.0 - candidate_e2e[1] / baseline_e2e[1]
    if p50_improvement < min_p50_improvement:
        reasons.append(
            f"p50 improvement {p50_improvement:.4f} is below {min_p50_improvement:.4f}"
        )
    if p90_improvement < min_p90_improvement:
        reasons.append(
            f"p90 improvement {p90_improvement:.4f} is below {min_p90_improvement:.4f}"
        )

    external_resources = _validate_external_resources(
        baseline_jetson_summary=baseline_jetson_summary,
        candidate_jetson_summary=candidate_jetson_summary,
        baseline_identity=baseline_identity,
        candidate_identity=candidate_identity,
        baseline_record_count=len(baseline.get("records", []))
        if isinstance(baseline.get("records"), list)
        else None,
        candidate_record_count=len(candidate.get("records", []))
        if isinstance(candidate.get("records"), list)
        else None,
    )
    memory_regression = None
    baseline_memory = baseline_performance.get("peak_memory_mb")
    candidate_memory = candidate_performance.get("peak_memory_mb")
    if baseline_memory is None and external_resources is not None:
        baseline_memory = external_resources["baseline"]["peak_memory_mb"]
    if candidate_memory is None and external_resources is not None:
        candidate_memory = external_resources["candidate"]["peak_memory_mb"]
    if baseline_memory is None or candidate_memory is None:
        reasons.append("peak memory metrics are unavailable for the comparison")
    else:
        baseline_memory = _number(baseline_memory, "baseline.performance_metrics.peak_memory_mb")
        candidate_memory = _number(candidate_memory, "candidate.performance_metrics.peak_memory_mb")
        memory_regression = candidate_memory / baseline_memory - 1.0
        if memory_regression > max_memory_regression:
            reasons.append(
                f"peak memory regression {memory_regression:.4f} exceeds {max_memory_regression:.4f}"
            )

    soak_ok = None
    if soak is not None:
        soak_failure_summary = soak.get("failure_summary", {})
        soak_ok = isinstance(soak_failure_summary, Mapping) and not soak_failure_summary
        if not soak_ok:
            reasons.append("soak report contains failures")
        soak_identity = soak.get("study_identity")
        if not isinstance(soak_identity, Mapping):
            reasons.append("soak report identity is required for promotion")
        else:
            soak_identity_mismatches = _identity_mismatches(
                candidate_identity,
                soak_identity,
            )
            if soak_identity_mismatches:
                reasons.append(
                    f"soak runtime/workload identity mismatch: {soak_identity_mismatches}"
                )
        soak_records = soak.get("records")
        soak_sample_count = len(soak_records) if isinstance(soak_records, list) else 0
        if soak_sample_count < min_soak_samples:
            reasons.append(
                f"soak report must contain at least {min_soak_samples} records; "
                f"observed {soak_sample_count}"
            )
    else:
        soak_ok = False
        reasons.append("soak report is required for promotion")

    graph_capture_ok = None
    graph_replay_ok = None
    paged_kv_active = None
    paged_kv_pool_pages = None
    fmha_tiled_active = None
    if require_fmha_tiled_active:
        if candidate_runtime_metadata is None:
            reasons.append("candidate runtime metadata is required to confirm FMHA variant")
        else:
            if candidate_runtime_metadata.get("schema_version") != "parksight_tensorrt_runtime_options_v1":
                reasons.append("candidate runtime metadata schema is not confirmed")
            metadata_options = _mapping(
                candidate_runtime_metadata.get("options"),
                "candidate_runtime_metadata.options",
            )
            fmha_option = metadata_options.get("fmha_force_granular_tiling")
            metadata_enabled = fmha_option in ("1", 1, True, "enabled")
            if not metadata_enabled:
                reasons.append("candidate FMHA tiled runtime option is not confirmed enabled")

        if candidate_runtime_log_summary is None:
            reasons.append("candidate runtime log summary is required to confirm FMHA tiled selection")
        else:
            fmha_summary = _mapping(
                candidate_runtime_log_summary.get("fmha"),
                "candidate_runtime_log_summary.fmha",
            )
            fmha_tiled_active = fmha_summary.get("forced_tiled_head128_observed") is True
            if not fmha_tiled_active:
                reasons.append("candidate FMHA tiled selection marker is not observed")

    if candidate_runtime_log_summary is not None:
        if require_paged_kv_active:
            attention = _mapping(
                candidate_runtime_log_summary.get("attention"),
                "candidate_runtime_log_summary.attention",
            )
            paged_kv_active = attention.get("use_paged_kv_cache") is True
            if not paged_kv_active:
                reasons.append("candidate paged KV cache is not confirmed active")
            raw_pages = attention.get("max_kv_pool_pages")
            if isinstance(raw_pages, bool) or not isinstance(raw_pages, int) or raw_pages <= 0:
                reasons.append("candidate paged KV pool page count is not confirmed")
            else:
                paged_kv_pool_pages = raw_pages
        graph = _mapping(
            candidate_runtime_log_summary.get("cuda_graph"),
            "candidate_runtime_log_summary.cuda_graph",
        )
        graph_capture_ok = graph.get("decoder_capture_success") is True
        if not graph_capture_ok:
            reasons.append("candidate decoder CUDA Graph capture is not confirmed")
        graph_replay_ok = graph.get("replay_observed") is True
        if not graph_replay_ok:
            reasons.append("candidate decoder CUDA Graph replay is not confirmed")
    elif require_paged_kv_active:
        reasons.append("candidate runtime log summary is required to confirm paged KV")

    return {
        "schema_version": "parksight_tensorrt_candidate_gate_v1",
        "promotion_eligible": not reasons,
        "reasons": reasons,
        "comparison": {
            "baseline_study_id": baseline_identity.get("study_id"),
            "candidate_study_id": candidate_identity.get("study_id"),
            "baseline_e2e_ms": {"p50": baseline_e2e[0], "p90": baseline_e2e[1]},
            "candidate_e2e_ms": {"p50": candidate_e2e[0], "p90": candidate_e2e[1]},
            "stage_latency_ms": stage_comparison,
            "p50_improvement": p50_improvement,
            "p90_improvement": p90_improvement,
            "memory_regression": memory_regression,
            "soak_ok": soak_ok,
            "decoder_graph_capture_ok": graph_capture_ok,
            "decoder_graph_replay_ok": graph_replay_ok,
            "paged_kv_active": paged_kv_active,
            "paged_kv_pool_pages": paged_kv_pool_pages,
            "fmha_tiled_active": fmha_tiled_active,
            "benchmark_repetition_gate_ok": benchmark_gate_ok,
            "backend_completion_rate": completion,
            "json_validity_rate": json_validity,
            "quality": quality_comparison,
            "resource_evidence": external_resources,
        },
        "policy": {
            "min_repetitions": min_repetitions,
            "min_p50_improvement": min_p50_improvement,
            "min_p90_improvement": min_p90_improvement,
            "max_memory_regression": max_memory_regression,
            "require_backend_completion": require_backend_completion,
            "require_json_validity": require_json_validity,
            "require_quality_non_regression": require_quality_non_regression,
            "min_soak_samples": min_soak_samples,
            "require_paged_kv_active": require_paged_kv_active,
            "require_fmha_tiled_active": require_fmha_tiled_active,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--soak", type=Path)
    parser.add_argument(
        "--candidate-runtime-log-summary",
        type=Path,
        help="可选 runtime log summary；提供后必须同时确认 decoder capture 与 replay",
    )
    parser.add_argument(
        "--candidate-runtime-metadata",
        type=Path,
        help="可选服务启动 runtime metadata；FMHA candidate 门禁需要它确认开关",
    )
    parser.add_argument(
        "--require-paged-kv-active",
        action="store_true",
        help="paged-KV candidate 必须由 runtime summary 明确确认已启用",
    )
    parser.add_argument(
        "--require-fmha-tiled-active",
        action="store_true",
        help="FMHA candidate 必须由 runtime metadata 和 log marker 明确确认命中 tiled 路径",
    )
    parser.add_argument(
        "--benchmark-comparison",
        type=Path,
        help="可选低层 benchmark compare；提供后必须通过 repetition gate",
    )
    parser.add_argument("--tuning-config", type=Path)
    parser.add_argument(
        "--baseline-jetson-summary",
        type=Path,
        help="可选 parksight_jetson_runtime_summary_v1；需与 baseline records 数量一致",
    )
    parser.add_argument(
        "--candidate-jetson-summary",
        type=Path,
        help="可选 parksight_jetson_runtime_summary_v1；需与 candidate records 数量一致",
    )
    args = parser.parse_args(argv)
    baseline = _read_object(args.baseline)
    candidate = _read_object(args.candidate)
    soak = _read_object(args.soak) if args.soak is not None else None
    candidate_runtime_log_summary = (
        _read_object(args.candidate_runtime_log_summary)
        if args.candidate_runtime_log_summary is not None
        else None
    )
    candidate_runtime_metadata = (
        _read_object(args.candidate_runtime_metadata)
        if args.candidate_runtime_metadata is not None
        else None
    )
    baseline_jetson_summary = (
        _read_object(args.baseline_jetson_summary)
        if args.baseline_jetson_summary is not None
        else None
    )
    candidate_jetson_summary = (
        _read_object(args.candidate_jetson_summary)
        if args.candidate_jetson_summary is not None
        else None
    )
    benchmark_comparison = (
        _read_object(args.benchmark_comparison)
        if args.benchmark_comparison is not None
        else None
    )
    policy = {
        "min_repetitions": 3,
        "min_p50_improvement": 0.10,
        "min_p90_improvement": 0.05,
        "max_memory_regression": 0.05,
    }
    if args.tuning_config is not None:
        tuning_config = TensorRTTuningConfig.load(args.tuning_config)
        acceptance = tuning_config.acceptance
        policy = {
            "min_repetitions": acceptance.min_repetitions,
            "min_p50_improvement": acceptance.min_p50_improvement,
            "min_p90_improvement": acceptance.min_p90_improvement,
            "max_memory_regression": acceptance.max_memory_regression,
            "require_backend_completion": acceptance.require_backend_completion,
            "require_json_validity": acceptance.require_json_validity,
            "require_quality_non_regression": acceptance.require_quality_non_regression,
            "min_soak_samples": tuning_config.soak_samples,
        }
    result = evaluate_candidate(
        baseline=baseline,
        candidate=candidate,
        soak=soak,
        candidate_runtime_log_summary=candidate_runtime_log_summary,
        candidate_runtime_metadata=candidate_runtime_metadata,
        baseline_jetson_summary=baseline_jetson_summary,
        candidate_jetson_summary=candidate_jetson_summary,
        benchmark_comparison=benchmark_comparison,
        require_paged_kv_active=args.require_paged_kv_active,
        require_fmha_tiled_active=args.require_fmha_tiled_active,
        **policy,
    )
    write_json(result, args.output)
    print(args.output)
    return 0 if result["promotion_eligible"] else 2


def _read_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise TensorRTValidationError("report path must be provided")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TensorRTValidationError(f"cannot read report: {path}") from error
    if not isinstance(payload, dict):
        raise TensorRTValidationError(f"report must be an object: {path}")
    return payload


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TensorRTValidationError(f"{context} must be an object")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TensorRTValidationError(f"{context} must be numeric")
    return float(value)


def _identity_mismatches(
    candidate_identity: Mapping[str, Any], soak_identity: Mapping[str, Any]
) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    for field in ("workload_identity", "split", "power_mode"):
        if candidate_identity.get(field) != soak_identity.get(field):
            mismatches[field] = (
                candidate_identity.get(field),
                soak_identity.get(field),
            )
    candidate_runtime = candidate_identity.get("runtime_identity")
    soak_runtime = soak_identity.get("runtime_identity")
    if not isinstance(candidate_runtime, Mapping) or not isinstance(soak_runtime, Mapping):
        mismatches["runtime_identity"] = (
            candidate_runtime,
            soak_runtime,
        )
    else:
        runtime_mismatches = {
            field: (candidate_runtime.get(field), soak_runtime.get(field))
            for field in (
                "backend",
                "backend_revision",
                "model_id",
                "model_revision",
                "adapter_revision",
                "precision",
            )
            if candidate_runtime.get(field) != soak_runtime.get(field)
        }
        if runtime_mismatches:
            mismatches["runtime_identity"] = runtime_mismatches
    return mismatches


def _validate_external_resources(
    *,
    baseline_jetson_summary: Mapping[str, Any] | None,
    candidate_jetson_summary: Mapping[str, Any] | None,
    baseline_identity: Mapping[str, Any],
    candidate_identity: Mapping[str, Any],
    baseline_record_count: int | None,
    candidate_record_count: int | None,
) -> dict[str, Any] | None:
    if baseline_jetson_summary is None and candidate_jetson_summary is None:
        return None
    if baseline_jetson_summary is None or candidate_jetson_summary is None:
        raise TensorRTValidationError(
            "baseline and candidate Jetson summaries must be provided together"
        )
    baseline_resources = _read_jetson_resources(
        baseline_jetson_summary,
        expected_identity=baseline_identity,
        expected_record_count=baseline_record_count,
        context="baseline Jetson summary",
    )
    candidate_resources = _read_jetson_resources(
        candidate_jetson_summary,
        expected_identity=candidate_identity,
        expected_record_count=candidate_record_count,
        context="candidate Jetson summary",
    )
    return {
        "schema_version": "parksight_external_jetson_resources_v1",
        "baseline": baseline_resources,
        "candidate": candidate_resources,
    }


def _read_jetson_resources(
    summary: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    expected_record_count: int | None,
    context: str,
) -> dict[str, Any]:
    if summary.get("schema_version") != "parksight_jetson_runtime_summary_v1":
        raise TensorRTValidationError(
            f"{context} has unsupported or missing schema_version"
        )
    summary_identity = summary.get("study_identity")
    if not isinstance(summary_identity, Mapping):
        raise TensorRTValidationError(f"{context}.study_identity must be an object")
    identity_mismatches = _identity_mismatches(expected_identity, summary_identity)
    if identity_mismatches:
        raise TensorRTValidationError(
            f"{context} identity mismatch: {identity_mismatches}"
        )
    runtime_execution = summary.get("runtime_execution")
    telemetry = summary.get("jetson_telemetry")
    if not isinstance(runtime_execution, Mapping) or not isinstance(telemetry, Mapping):
        raise TensorRTValidationError(
            f"{context} must contain runtime_execution and jetson_telemetry"
        )
    summary_record_count = runtime_execution.get("record_count")
    if (
        expected_record_count is None
        or not isinstance(summary_record_count, int)
        or summary_record_count != expected_record_count
    ):
        raise TensorRTValidationError(
            f"{context} record_count does not match StudyReport: "
            f"summary={summary_record_count}, report={expected_record_count}"
        )
    resources = {
        "record_count": summary_record_count,
        "peak_memory_mb": _telemetry_number(
            telemetry, "ram_used_mb", "maximum", context
        ),
        "average_power_w": _telemetry_number(
            telemetry, "vdd_in_w", "mean", context
        ),
        "peak_temperature_c": _telemetry_number(
            telemetry, "gpu_temperature_c", "maximum", context
        ),
        "swap_used_mb_maximum": _telemetry_number(
            telemetry, "swap_used_mb", "maximum", context
        ),
        "gpu_utilization_percent_mean": _telemetry_number(
            telemetry, "gpu_utilization_percent", "mean", context
        ),
    }
    sources = summary.get("evidence_sources")
    if isinstance(sources, Mapping):
        resources["evidence_sources"] = dict(sources)
    return resources


def _telemetry_number(
    telemetry: Mapping[str, Any],
    group_name: str,
    field_name: str,
    context: str,
) -> float:
    group = telemetry.get(group_name)
    if not isinstance(group, Mapping):
        raise TensorRTValidationError(f"{context}.jetson_telemetry.{group_name} must be an object")
    value = group.get(field_name)
    return _number(
        value,
        f"{context}.jetson_telemetry.{group_name}.{field_name}",
    )


def _completion_rate(report: Mapping[str, Any], performance: Mapping[str, Any]) -> float:
    records = report.get("records")
    total = len(records) if isinstance(records, list) else 0
    completed_value = performance.get("backend_completed_sample_count")
    completed_context = "backend_completed_sample_count"
    if completed_value is None:
        # StudyReport currently exposes this count as successful_sample_count;
        # accept the newer backend-specific name when available.
        completed_value = performance.get("successful_sample_count")
        completed_context = "successful_sample_count"
    completed = _number(
        completed_value,
        completed_context,
    )
    if total == 0:
        total = int(completed)
    if total <= 0:
        return 0.0
    return completed / total


def _stage_p50_p90(performance: Mapping[str, Any], context: str) -> tuple[float, float]:
    stages = _mapping(performance.get("stage_latency_ms"), f"{context}.performance_metrics.stage_latency_ms")
    e2e = _mapping(stages.get("end_to_end_ms"), f"{context}.end_to_end_ms")
    return (
        _number(e2e.get("p50"), f"{context}.end_to_end_ms.p50"),
        _number(e2e.get("p90"), f"{context}.end_to_end_ms.p90"),
    )


def _compare_performance_stages(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """保留所有已采集阶段的 p50/p90 变化；不把缺失值伪造成 0。"""
    baseline_stages = baseline.get("stage_latency_ms")
    candidate_stages = candidate.get("stage_latency_ms")
    if not isinstance(baseline_stages, Mapping) or not isinstance(candidate_stages, Mapping):
        return {stage: _missing_stage_comparison() for stage in _PERFORMANCE_STAGES}

    result: dict[str, Any] = {}
    for stage in _PERFORMANCE_STAGES:
        baseline_summary = baseline_stages.get(stage)
        candidate_summary = candidate_stages.get(stage)
        if not isinstance(baseline_summary, Mapping) or not isinstance(candidate_summary, Mapping):
            result[stage] = _missing_stage_comparison()
            continue
        stage_result: dict[str, Any] = {}
        for percentile in ("p50", "p90", "p99"):
            baseline_value = baseline_summary.get(percentile)
            candidate_value = candidate_summary.get(percentile)
            if not _is_number(baseline_value) or not _is_number(candidate_value):
                stage_result[percentile] = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "improvement": None,
                    "speedup": None,
                }
                continue
            baseline_float = float(baseline_value)
            candidate_float = float(candidate_value)
            stage_result[percentile] = {
                "baseline": baseline_float,
                "candidate": candidate_float,
                "improvement": (
                    1.0 - candidate_float / baseline_float
                    if baseline_float > 0
                    else None
                ),
                "speedup": (
                    baseline_float / candidate_float
                    if candidate_float > 0
                    else None
                ),
            }
        stage_result["baseline_count"] = baseline_summary.get("count")
        stage_result["candidate_count"] = candidate_summary.get("count")
        result[stage] = stage_result
    return result


def _missing_stage_comparison() -> dict[str, Any]:
    return {
        percentile: {
            "baseline": None,
            "candidate": None,
            "improvement": None,
            "speedup": None,
        }
        for percentile in ("p50", "p90", "p99")
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
