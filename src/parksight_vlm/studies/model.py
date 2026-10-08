"""不可变的研究配置、指标和报告对象。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parksight_vlm.casebook import DatasetSplit
from parksight_vlm.inference import InferenceRecord, RuntimeIdentity
from parksight_vlm.workload import FrozenWorkload


class StudyValidationError(ValueError):
    """当研究定义或选定 workload 无效时抛出。"""


@dataclass(frozen=True, slots=True)
class StudyDefinition:
    """基于一个 workload 和数据集划分的可重复研究。"""

    study_id: str
    workload: FrozenWorkload
    split: DatasetSplit
    repetitions: int
    power_mode: str

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, workload: FrozenWorkload
    ) -> "StudyDefinition":
        if not isinstance(payload, Mapping):
            raise StudyValidationError("study payload must be a mapping")
        expected_fields = {"study_id", "split", "repetitions", "power_mode"}
        actual_fields = set(payload)
        if actual_fields != expected_fields:
            missing_fields = expected_fields - actual_fields
            unexpected_fields = actual_fields - expected_fields
            raise StudyValidationError(
                "invalid study fields; "
                f"missing={sorted(missing_fields)}, unexpected={sorted(unexpected_fields)}"
            )
        study_id = _parse_text(payload["study_id"], "study_id")
        power_mode = _parse_text(payload["power_mode"], "power_mode")
        repetitions = payload["repetitions"]
        if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
            raise StudyValidationError("repetitions must be a positive integer")
        try:
            split = DatasetSplit(payload["split"])
        except (TypeError, ValueError) as error:
            raise StudyValidationError("split must be train, validation, or test") from error
        return cls(
            study_id=study_id,
            workload=workload,
            split=split,
            repetitions=repetitions,
            power_mode=power_mode,
        )

    @classmethod
    def load(cls, path: Path | str, *, workload: FrozenWorkload) -> "StudyDefinition":
        study_path = Path(path)
        try:
            payload = json.loads(study_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise StudyValidationError(f"cannot read study file: {study_path}") from error
        except json.JSONDecodeError as error:
            raise StudyValidationError(f"invalid study JSON: {study_path}") from error
        return cls.from_mapping(payload, workload=workload)

    def identity_mapping(self, runtime_identity: RuntimeIdentity) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "workload_identity": self.workload.identity,
            "runtime_identity": runtime_identity.to_mapping(),
            "split": self.split.value,
            "repetitions": self.repetitions,
            "power_mode": self.power_mode,
        }


@dataclass(frozen=True, slots=True)
class PercentileSummary:
    count: int
    p50: float
    p90: float
    p99: float

    def to_mapping(self) -> dict[str, int | float]:
        return {"count": self.count, "p50": self.p50, "p90": self.p90, "p99": self.p99}


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    sample_count: int
    json_validity_rate: float
    risk_level_accuracy: float
    event_micro_precision: float
    event_micro_recall: float
    event_micro_f1: float
    event_macro_f1: float
    unsafe_advice_rate: float
    semantic_consistency_rate: float
    event_errors: Mapping[str, Mapping[str, int]]
    event_metrics: Mapping[str, Mapping[str, int | float]]
    semantic_issue_counts: Mapping[str, int]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "json_validity_rate": self.json_validity_rate,
            "risk_level_accuracy": self.risk_level_accuracy,
            "event_micro_precision": self.event_micro_precision,
            "event_micro_recall": self.event_micro_recall,
            "event_micro_f1": self.event_micro_f1,
            "event_macro_f1": self.event_macro_f1,
            "unsafe_advice_rate": self.unsafe_advice_rate,
            "semantic_consistency_rate": self.semantic_consistency_rate,
            "event_errors": {event: dict(errors) for event, errors in self.event_errors.items()},
            "event_metrics": {
                event: dict(metrics) for event, metrics in self.event_metrics.items()
            },
            "semantic_issue_counts": dict(self.semantic_issue_counts),
        }


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    successful_sample_count: int
    backend_completed_sample_count: int
    cold_start_ms: float | None
    stage_latency_ms: Mapping[str, PercentileSummary]
    tokens_per_second: float | None
    aggregate_output_tokens_per_end_to_end_second: float | None
    peak_memory_mb: float | None
    average_power_w: float | None
    peak_temperature_c: float | None
    first_request_ms: float | None = None
    stream_chunk_interval_ms: PercentileSummary | None = None
    token_counts: Mapping[str, int] = field(default_factory=dict)
    measurement_wall_seconds: float | None = None
    completed_requests_per_second: float | None = None
    wall_output_tokens_per_second: float | None = None
    attempted_sample_count: int | None = None
    failed_sample_count: int | None = None
    all_request_stage_latency_ms: Mapping[str, PercentileSummary] = field(default_factory=dict)
    failed_request_stage_latency_ms: Mapping[str, PercentileSummary] = field(default_factory=dict)
    attempted_requests_per_second: float | None = None
    successful_requests_per_second: float | None = None
    decode_tokens_per_second: float | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "successful_sample_count": self.successful_sample_count,
            "backend_completed_sample_count": self.backend_completed_sample_count,
            "attempted_sample_count": self.attempted_sample_count,
            "failed_sample_count": self.failed_sample_count,
            "all_request_stage_latency_ms": {
                stage: summary.to_mapping() for stage, summary in self.all_request_stage_latency_ms.items()
            },
            "failed_request_stage_latency_ms": {
                stage: summary.to_mapping() for stage, summary in self.failed_request_stage_latency_ms.items()
            },
            "attempted_requests_per_second": self.attempted_requests_per_second,
            "successful_requests_per_second": self.successful_requests_per_second,
            "cold_start_ms": self.cold_start_ms,
            "first_request_ms": self.first_request_ms,
            "stream_chunk_interval_ms": self.stream_chunk_interval_ms.to_mapping() if self.stream_chunk_interval_ms is not None else None,
            "token_counts": dict(self.token_counts),
            "measurement_wall_seconds": self.measurement_wall_seconds,
            "completed_requests_per_second": self.completed_requests_per_second,
            "wall_output_tokens_per_second": self.wall_output_tokens_per_second,
            "stage_latency_ms": {
                stage: summary.to_mapping() for stage, summary in self.stage_latency_ms.items()
            },
            "tokens_per_second": self.tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "aggregate_output_tokens_per_end_to_end_second": (
                self.aggregate_output_tokens_per_end_to_end_second
            ),
            "peak_memory_mb": self.peak_memory_mb,
            "average_power_w": self.average_power_w,
            "peak_temperature_c": self.peak_temperature_c,
        }


@dataclass(frozen=True, slots=True)
class StudyReport:
    """完整研究证据，包括全部底层推理记录。"""

    study_identity: Mapping[str, Any]
    environment_snapshot: Mapping[str, Any]
    quality_metrics: QualityMetrics
    performance_metrics: PerformanceMetrics
    failure_summary: Mapping[str, int]
    records: tuple[InferenceRecord, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "study_identity": dict(self.study_identity),
            "environment_snapshot": dict(self.environment_snapshot),
            "quality_metrics": self.quality_metrics.to_mapping(),
            "performance_metrics": self.performance_metrics.to_mapping(),
            "failure_summary": dict(self.failure_summary),
            "records": [record.to_mapping() for record in self.records],
        }

    def write_json(self, path: Path | str) -> None:
        """将完整报告写为便于阅读的 UTF-8 JSON。"""
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(self.to_mapping(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _parse_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StudyValidationError(f"{field_name} must be a non-blank string")
    return value.strip()
