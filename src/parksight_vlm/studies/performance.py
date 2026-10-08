"""时延、吞吐、内存、功耗和温度汇总。"""

from __future__ import annotations

import math
from collections.abc import Sequence

from parksight_vlm.inference import InferenceRecord

from .model import PerformanceMetrics, PercentileSummary


def compute_performance_metrics(
    records: Sequence[InferenceRecord], *, measurement_wall_seconds: float | None = None
) -> PerformanceMetrics:
    """分别汇总完整模型输出、所有尝试和失败记录，不补估缺失测量。"""
    if measurement_wall_seconds is not None and (
        isinstance(measurement_wall_seconds, bool)
        or not math.isfinite(measurement_wall_seconds) or measurement_wall_seconds <= 0
    ):
        raise ValueError("measurement_wall_seconds must be finite and positive")
    successful_records = [record for record in records if record.succeeded]
    failed_records = [record for record in records if not record.succeeded]
    # JSON 校验失败仍然可能已经完整执行了模型；这些记录应参与运行时性能统计，
    # 但不参与质量指标。raw_output=None 才表示后端没有返回生成结果。
    backend_completed_records = [
        record for record in records if record.raw_output is not None
    ]
    stage_latency_ms = _stage_summaries(backend_completed_records)
    # A request may target an already-warm HTTP service. Without a separate
    # runtime initialization measurement, its latency cannot prove cold start.
    first_request_ms = records[0].stage_timings.end_to_end_ms if records else None

    total_tokens = 0
    total_decode_ms = 0.0
    for record in backend_completed_records:
        if record.output_tokens is None or record.stage_timings.decode_ms is None:
            continue
        total_tokens += record.output_tokens
        total_decode_ms += record.stage_timings.decode_ms
    tokens_per_second = None
    if total_decode_ms > 0:
        tokens_per_second = total_tokens / (total_decode_ms / 1000.0)

    decode_rate_record_count = 0
    measured_decode_tokens = 0
    measured_decode_ms = 0.0
    for record in backend_completed_records:
        tokens, duration = record.decode_tokens, record.stage_timings.decode_ms
        if tokens is None or duration is None:
            continue
        measured_decode_tokens += tokens
        measured_decode_ms += duration
        decode_rate_record_count += 1
    decode_tokens_per_second = (
        measured_decode_tokens / (measured_decode_ms / 1000)
        if measured_decode_ms > 0 else None
    )

    total_output_tokens = 0
    total_end_to_end_ms = 0.0
    for record in backend_completed_records:
        if record.output_tokens is None or record.stage_timings.end_to_end_ms is None:
            continue
        total_output_tokens += record.output_tokens
        total_end_to_end_ms += record.stage_timings.end_to_end_ms
    aggregate_output_tokens_per_end_to_end_second = None
    if total_end_to_end_ms > 0:
        aggregate_output_tokens_per_end_to_end_second = (
            total_output_tokens / (total_end_to_end_ms / 1000.0)
        )

    # A failure can still carry measured resource evidence; unknown values
    # stay excluded rather than being filled with zero.
    memory_values = _resource_values(records, "peak_memory_mb")
    power_values = _resource_values(records, "average_power_w")
    temperature_values = _resource_values(records, "peak_temperature_c")
    return PerformanceMetrics(
        successful_sample_count=len(successful_records),
        backend_completed_sample_count=len(backend_completed_records),
        attempted_sample_count=len(records),
        failed_sample_count=len(failed_records),
        all_request_stage_latency_ms=_stage_summaries(records),
        failed_request_stage_latency_ms=_stage_summaries(failed_records),
        attempted_requests_per_second=(len(records) / measurement_wall_seconds if measurement_wall_seconds is not None else None),
        successful_requests_per_second=(len(successful_records) / measurement_wall_seconds if measurement_wall_seconds is not None else None),
        cold_start_ms=None,
        first_request_ms=first_request_ms,
        stage_latency_ms=stage_latency_ms,
        tokens_per_second=tokens_per_second,
        decode_tokens_per_second=decode_tokens_per_second,
        aggregate_output_tokens_per_end_to_end_second=(
            aggregate_output_tokens_per_end_to_end_second
        ),
        peak_memory_mb=max(memory_values) if memory_values else None,
        average_power_w=sum(power_values) / len(power_values) if power_values else None,
        peak_temperature_c=max(temperature_values) if temperature_values else None,
        stream_chunk_interval_ms=(
            _summarize(chunk_intervals) if (chunk_intervals := [
                interval for record in backend_completed_records
                for interval in record.stream_timings.chunk_intervals_ms
            ]) else None
        ),
        token_counts={
            "input_tokens": sum(record.input_tokens or 0 for record in backend_completed_records),
            "output_tokens": sum(record.output_tokens or 0 for record in backend_completed_records),
            "input_usage_record_count": sum(record.input_tokens is not None for record in backend_completed_records),
            "output_usage_record_count": sum(record.output_tokens is not None for record in backend_completed_records),
            "decode_tokens": sum(record.decode_tokens or 0 for record in backend_completed_records),
            "decode_usage_record_count": sum(record.decode_tokens is not None for record in backend_completed_records),
            "decode_rate_record_count": decode_rate_record_count,
        },
        measurement_wall_seconds=measurement_wall_seconds,
        completed_requests_per_second=(len(backend_completed_records) / measurement_wall_seconds if measurement_wall_seconds is not None else None),
        wall_output_tokens_per_second=(
            sum(record.output_tokens or 0 for record in backend_completed_records) / measurement_wall_seconds
            if measurement_wall_seconds is not None and backend_completed_records
            and all(record.output_tokens is not None for record in backend_completed_records) else None
        ),
    )


def _stage_summaries(records: Sequence[InferenceRecord]) -> dict[str, PercentileSummary]:
    values: dict[str, list[float]] = {}
    for record in records:
        for stage, value in record.stage_timings.to_mapping().items():
            if value is not None:
                values.setdefault(stage, []).append(value)
    return {stage: _summarize(samples) for stage, samples in values.items()}


def _summarize(values: list[float]) -> PercentileSummary:
    ordered_values = sorted(values)
    return PercentileSummary(
        count=len(ordered_values),
        p50=_percentile(ordered_values, 0.50),
        p90=_percentile(ordered_values, 0.90),
        p99=_percentile(ordered_values, 0.99),
    )


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    weight = position - lower_index
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight


def _resource_values(records: Sequence[InferenceRecord], field_name: str) -> list[float]:
    values = []
    for record in records:
        value = getattr(record.resource_snapshot, field_name)
        if value is not None:
            values.append(value)
    return values
