"""TensorRT tuning 配置、engine provenance 和低层 benchmark 证据。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TensorRTValidationError(ValueError):
    """TensorRT 实验配置或运行证据不满足约束。"""


def validate_profile_contract(
    *,
    max_input_len: int,
    max_kv_cache_capacity: int,
    observed_input_tokens: int,
    max_generate_length: int,
) -> dict[str, Any]:
    """检查输入 profile 与 KV 容量是否覆盖实测 prompt 和输出预算。"""
    values = {
        "max_input_len": max_input_len,
        "max_kv_cache_capacity": max_kv_cache_capacity,
        "observed_input_tokens": observed_input_tokens,
        "max_generate_length": max_generate_length,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values.values()
    ):
        raise TensorRTValidationError(
            "profile contract values must be non-negative integers"
        )
    required_kv_capacity = observed_input_tokens + max_generate_length
    reasons: list[str] = []
    if observed_input_tokens > max_input_len:
        reasons.append(
            f"observed input {observed_input_tokens} exceeds max_input_len {max_input_len}"
        )
    if required_kv_capacity > max_kv_cache_capacity:
        reasons.append(
            f"input plus output budget {required_kv_capacity} exceeds "
            f"max_kv_cache_capacity {max_kv_cache_capacity}"
        )
    return {
        **values,
        "required_kv_capacity": required_kv_capacity,
        "input_headroom": max_input_len - observed_input_tokens,
        "kv_headroom": max_kv_cache_capacity - required_kv_capacity,
        "valid": not reasons,
        "reasons": reasons,
    }


def estimate_kv_cache_bytes(
    *,
    max_batch_size: int,
    max_kv_cache_capacity: int,
    num_kv_heads: int,
    head_dim: int,
    element_size_bytes: int = 2,
) -> int:
    """估算连续 K/V cache 的字节数，不含 allocator 对齐开销。"""
    values = {
        "max_batch_size": max_batch_size,
        "max_kv_cache_capacity": max_kv_cache_capacity,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "element_size_bytes": element_size_bytes,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values.values()
    ):
        raise TensorRTValidationError("KV cache dimensions must be positive integers")
    return (
        max_batch_size
        * 2
        * num_kv_heads
        * max_kv_cache_capacity
        * head_dim
        * element_size_bytes
    )


@dataclass(frozen=True, slots=True)
class TensorRTFixedConfig:
    """所有 A/B 实验必须保持不变的条件。"""

    model_revision: str
    edge_llm_revision: str
    platform: str
    power_mode: str
    precision: str
    max_batch_size: int
    max_input_len: int
    max_kv_cache_capacity: int


@dataclass(frozen=True, slots=True)
class TensorRTAcceptance:
    """候选 engine 晋级为默认版本的门禁。"""

    min_repetitions: int
    min_p50_improvement: float
    min_p90_improvement: float
    max_memory_regression: float
    require_backend_completion: bool
    require_json_validity: bool
    require_quality_non_regression: bool


@dataclass(frozen=True, slots=True)
class TensorRTTuningConfig:
    """可审计的 TensorRT 单变量优化实验矩阵。"""

    schema_version: str
    experiment_id: str
    fixed: TensorRTFixedConfig
    builder_optimization_levels: tuple[int, ...]
    workspace_limits_mib: tuple[int, ...]
    cuda_graph_modes: tuple[str, ...]
    weight_streaming_budgets: tuple[str, ...]
    profile_variants: tuple[tuple[int, int], ...]
    repetitions: int
    samples_per_repetition: int
    soak_samples: int
    acceptance: TensorRTAcceptance

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TensorRTTuningConfig":
        _require_exact_fields(
            payload,
            {
                "schema_version",
                "experiment_id",
                "fixed",
                "sweep",
                "validation",
                "acceptance",
            },
            "TensorRT tuning config",
        )
        fixed_payload = _require_mapping(payload["fixed"], "fixed")
        _require_exact_fields(
            fixed_payload,
            {
                "model_revision",
                "edge_llm_revision",
                "platform",
                "power_mode",
                "precision",
                "max_batch_size",
                "max_input_len",
                "max_kv_cache_capacity",
            },
            "fixed",
        )
        fixed = TensorRTFixedConfig(
            model_revision=_parse_text(fixed_payload["model_revision"], "fixed.model_revision"),
            edge_llm_revision=_parse_text(
                fixed_payload["edge_llm_revision"], "fixed.edge_llm_revision"
            ),
            platform=_parse_text(fixed_payload["platform"], "fixed.platform"),
            power_mode=_parse_text(fixed_payload["power_mode"], "fixed.power_mode"),
            precision=_parse_text(fixed_payload["precision"], "fixed.precision"),
            max_batch_size=_parse_positive_int(
                fixed_payload["max_batch_size"], "fixed.max_batch_size"
            ),
            max_input_len=_parse_positive_int(
                fixed_payload["max_input_len"], "fixed.max_input_len"
            ),
            max_kv_cache_capacity=_parse_positive_int(
                fixed_payload["max_kv_cache_capacity"], "fixed.max_kv_cache_capacity"
            ),
        )
        sweep = _require_mapping(payload["sweep"], "sweep")
        _require_exact_fields(
            sweep,
            {
                "builder_optimization_levels",
                "workspace_limits_mib",
                "cuda_graph_modes",
                "weight_streaming_budgets",
                "profile_variants",
            },
            "sweep",
        )
        profile_variants = tuple(
            _parse_profile_variant(item, index)
            for index, item in enumerate(_require_sequence(sweep["profile_variants"], "sweep.profile_variants"))
        )
        validation = _require_mapping(payload["validation"], "validation")
        _require_exact_fields(
            validation,
            {"repetitions", "samples_per_repetition", "soak_samples"},
            "validation",
        )
        acceptance_payload = _require_mapping(payload["acceptance"], "acceptance")
        _require_exact_fields(
            acceptance_payload,
            {
                "min_repetitions",
                "min_p50_improvement",
                "min_p90_improvement",
                "max_memory_regression",
                "require_backend_completion",
                "require_json_validity",
                "require_quality_non_regression",
            },
            "acceptance",
        )
        acceptance = TensorRTAcceptance(
            min_repetitions=_parse_positive_int(
                acceptance_payload["min_repetitions"], "acceptance.min_repetitions"
            ),
            min_p50_improvement=_parse_fraction(
                acceptance_payload["min_p50_improvement"], "acceptance.min_p50_improvement"
            ),
            min_p90_improvement=_parse_fraction(
                acceptance_payload["min_p90_improvement"], "acceptance.min_p90_improvement"
            ),
            max_memory_regression=_parse_fraction(
                acceptance_payload["max_memory_regression"], "acceptance.max_memory_regression"
            ),
            require_backend_completion=_parse_bool(
                acceptance_payload["require_backend_completion"],
                "acceptance.require_backend_completion",
            ),
            require_json_validity=_parse_bool(
                acceptance_payload["require_json_validity"],
                "acceptance.require_json_validity",
            ),
            require_quality_non_regression=_parse_bool(
                acceptance_payload["require_quality_non_regression"],
                "acceptance.require_quality_non_regression",
            ),
        )
        return cls(
            schema_version=_parse_text(payload["schema_version"], "schema_version"),
            experiment_id=_parse_text(payload["experiment_id"], "experiment_id"),
            fixed=fixed,
            builder_optimization_levels=_parse_positive_int_sequence(
                sweep["builder_optimization_levels"],
                "sweep.builder_optimization_levels",
                allow_zero=True,
                maximum=5,
            ),
            workspace_limits_mib=_parse_positive_int_sequence(
                sweep["workspace_limits_mib"],
                "sweep.workspace_limits_mib",
            ),
            cuda_graph_modes=_parse_choice_sequence(
                sweep["cuda_graph_modes"],
                "sweep.cuda_graph_modes",
                choices={"enabled", "disabled"},
            ),
            weight_streaming_budgets=_parse_text_sequence(
                sweep["weight_streaming_budgets"], "sweep.weight_streaming_budgets"
            ),
            profile_variants=profile_variants,
            repetitions=_parse_positive_int(validation["repetitions"], "validation.repetitions"),
            samples_per_repetition=_parse_positive_int(
                validation["samples_per_repetition"], "validation.samples_per_repetition"
            ),
            soak_samples=_parse_positive_int(validation["soak_samples"], "validation.soak_samples"),
            acceptance=acceptance,
        )

    @classmethod
    def load(cls, path: Path | str) -> "TensorRTTuningConfig":
        config_path = Path(path)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise TensorRTValidationError(f"cannot read TensorRT tuning config: {config_path}") from error
        except json.JSONDecodeError as error:
            raise TensorRTValidationError(f"invalid TensorRT tuning JSON: {config_path}") from error
        if not isinstance(payload, Mapping):
            raise TensorRTValidationError("TensorRT tuning config must be an object")
        return cls.from_mapping(payload)

    def fixed_mapping(self) -> dict[str, Any]:
        return {
            "model_revision": self.fixed.model_revision,
            "edge_llm_revision": self.fixed.edge_llm_revision,
            "platform": self.fixed.platform,
            "power_mode": self.fixed.power_mode,
            "precision": self.fixed.precision,
            "max_batch_size": self.fixed.max_batch_size,
            "max_input_len": self.fixed.max_input_len,
            "max_kv_cache_capacity": self.fixed.max_kv_cache_capacity,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "fixed": self.fixed_mapping(),
            "sweep": {
                "builder_optimization_levels": list(self.builder_optimization_levels),
                "workspace_limits_mib": list(self.workspace_limits_mib),
                "cuda_graph_modes": list(self.cuda_graph_modes),
                "weight_streaming_budgets": list(self.weight_streaming_budgets),
                "profile_variants": [
                    {"max_input_len": input_len, "max_kv_cache_capacity": kv_capacity}
                    for input_len, kv_capacity in self.profile_variants
                ],
            },
            "validation": {
                "repetitions": self.repetitions,
                "samples_per_repetition": self.samples_per_repetition,
                "soak_samples": self.soak_samples,
            },
            "acceptance": {
                "min_repetitions": self.acceptance.min_repetitions,
                "min_p50_improvement": self.acceptance.min_p50_improvement,
                "min_p90_improvement": self.acceptance.min_p90_improvement,
                "max_memory_regression": self.acceptance.max_memory_regression,
                "require_backend_completion": self.acceptance.require_backend_completion,
                "require_json_validity": self.acceptance.require_json_validity,
                "require_quality_non_regression": self.acceptance.require_quality_non_regression,
            },
        }

    def builder_variants(self) -> tuple[dict[str, Any], ...]:
        """生成第一阶段的 builder level 单变量矩阵。"""
        workspace = self.workspace_limits_mib[0]
        return tuple(
            {
                "variant_id": f"builder_opt_{level}_workspace_{workspace}",
                "builder_optimization_level": level,
                "workspace_limit_mib": workspace,
                "max_input_len": self.fixed.max_input_len,
                "max_kv_cache_capacity": self.fixed.max_kv_cache_capacity,
            }
            for level in self.builder_optimization_levels
        )


@dataclass(frozen=True, slots=True)
class TensorRTRuntimeVariant:
    """一个只改变 host-side runtime 开关的实验变体。"""

    variant_id: str
    pin_optimization_profiles: bool
    cache_binding_state: bool
    cache_registered_bindings: bool
    skip_redundant_profile_switch: bool
    skip_fallback_binding_scan: bool
    fmha_force_granular_tiling: bool | None = None
    int4_gemv_n_per_block: int | None = None
    int4_gemv_block_size: int | None = None
    runtime_patches: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variant_id": self.variant_id,
            "pin_optimization_profiles": self.pin_optimization_profiles,
            "cache_binding_state": self.cache_binding_state,
            "cache_registered_bindings": self.cache_registered_bindings,
            "skip_redundant_profile_switch": self.skip_redundant_profile_switch,
            "skip_fallback_binding_scan": self.skip_fallback_binding_scan,
        }
        if self.fmha_force_granular_tiling is not None:
            payload["fmha_force_granular_tiling"] = self.fmha_force_granular_tiling
        if self.int4_gemv_n_per_block is not None:
            payload["int4_gemv_n_per_block"] = self.int4_gemv_n_per_block
        if self.int4_gemv_block_size is not None:
            payload["int4_gemv_block_size"] = self.int4_gemv_block_size
        if self.runtime_patches:
            payload["runtime_patches"] = list(self.runtime_patches)
        return payload


@dataclass(frozen=True, slots=True)
class TensorRTRuntimeTuningConfig:
    """独立于 engine 构建参数的 host-side runtime A/B 矩阵。"""

    schema_version: str
    experiment_id: str
    fixed: Mapping[str, Any]
    variants: tuple[TensorRTRuntimeVariant, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TensorRTRuntimeTuningConfig":
        _require_exact_fields(
            payload,
            {"schema_version", "experiment_id", "fixed", "variants"},
            "TensorRT runtime tuning config",
        )
        fixed_payload = _require_mapping(payload["fixed"], "runtime fixed")
        fixed_fields = {
            "model_revision",
            "edge_llm_revision",
            "platform",
            "power_mode",
            "precision",
            "max_batch_size",
            "max_input_len",
            "max_kv_cache_capacity",
            "cuda_graph_mode",
        }
        _require_exact_fields(fixed_payload, fixed_fields, "runtime fixed")
        fixed = {
            "model_revision": _parse_text(fixed_payload["model_revision"], "runtime fixed.model_revision"),
            "edge_llm_revision": _parse_text(fixed_payload["edge_llm_revision"], "runtime fixed.edge_llm_revision"),
            "platform": _parse_text(fixed_payload["platform"], "runtime fixed.platform"),
            "power_mode": _parse_text(fixed_payload["power_mode"], "runtime fixed.power_mode"),
            "precision": _parse_text(fixed_payload["precision"], "runtime fixed.precision"),
            "max_batch_size": _parse_positive_int(fixed_payload["max_batch_size"], "runtime fixed.max_batch_size"),
            "max_input_len": _parse_positive_int(fixed_payload["max_input_len"], "runtime fixed.max_input_len"),
            "max_kv_cache_capacity": _parse_positive_int(
                fixed_payload["max_kv_cache_capacity"], "runtime fixed.max_kv_cache_capacity"
            ),
            "cuda_graph_mode": _parse_text(fixed_payload["cuda_graph_mode"], "runtime fixed.cuda_graph_mode"),
        }
        variants: list[TensorRTRuntimeVariant] = []
        seen_ids: set[str] = set()
        for index, raw_variant in enumerate(_require_sequence(payload["variants"], "runtime variants")):
            variant_payload = _require_mapping(raw_variant, f"runtime variants[{index}]")
            required_variant_fields = {
                "variant_id",
                "pin_optimization_profiles",
                "cache_binding_state",
                "cache_registered_bindings",
                "skip_redundant_profile_switch",
                "skip_fallback_binding_scan",
            }
            optional_variant_fields = {
                "fmha_force_granular_tiling",
                "int4_gemv_n_per_block",
                "int4_gemv_block_size",
                "runtime_patches",
            }
            _require_fields(
                variant_payload,
                required_variant_fields,
                required_variant_fields | optional_variant_fields,
                f"runtime variants[{index}]",
            )
            variant_id = _parse_text(variant_payload["variant_id"], f"runtime variants[{index}].variant_id")
            if variant_id in seen_ids:
                raise TensorRTValidationError(f"duplicate runtime variant_id: {variant_id}")
            seen_ids.add(variant_id)
            variants.append(
                TensorRTRuntimeVariant(
                    variant_id=variant_id,
                    pin_optimization_profiles=_parse_bool(
                        variant_payload["pin_optimization_profiles"],
                        f"runtime variants[{index}].pin_optimization_profiles",
                    ),
                    cache_binding_state=_parse_bool(
                        variant_payload["cache_binding_state"],
                        f"runtime variants[{index}].cache_binding_state",
                    ),
                    cache_registered_bindings=_parse_bool(
                        variant_payload["cache_registered_bindings"],
                        f"runtime variants[{index}].cache_registered_bindings",
                    ),
                    skip_redundant_profile_switch=_parse_bool(
                        variant_payload["skip_redundant_profile_switch"],
                        f"runtime variants[{index}].skip_redundant_profile_switch",
                    ),
                    skip_fallback_binding_scan=_parse_bool(
                        variant_payload["skip_fallback_binding_scan"],
                        f"runtime variants[{index}].skip_fallback_binding_scan",
                    ),
                    fmha_force_granular_tiling=(
                        _parse_bool(
                            variant_payload["fmha_force_granular_tiling"],
                            f"runtime variants[{index}].fmha_force_granular_tiling",
                        )
                        if "fmha_force_granular_tiling" in variant_payload
                        else None
                    ),
                    int4_gemv_n_per_block=_parse_optional_choice_int(
                        variant_payload.get("int4_gemv_n_per_block"),
                        f"runtime variants[{index}].int4_gemv_n_per_block",
                        choices={2, 4},
                    ),
                    int4_gemv_block_size=_parse_optional_choice_int(
                        variant_payload.get("int4_gemv_block_size"),
                        f"runtime variants[{index}].int4_gemv_block_size",
                        choices={128, 256, 512},
                    ),
                    runtime_patches=_parse_text_sequence_allow_empty(
                        variant_payload.get("runtime_patches"),
                        f"runtime variants[{index}].runtime_patches",
                    ),
                )
            )
        return cls(
            schema_version=_parse_text(payload["schema_version"], "schema_version"),
            experiment_id=_parse_text(payload["experiment_id"], "experiment_id"),
            fixed=fixed,
            variants=tuple(variants),
        )

    @classmethod
    def load(cls, path: Path | str) -> "TensorRTRuntimeTuningConfig":
        config_path = Path(path)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise TensorRTValidationError(f"cannot read TensorRT runtime tuning config: {config_path}") from error
        except json.JSONDecodeError as error:
            raise TensorRTValidationError(f"invalid TensorRT runtime tuning JSON: {config_path}") from error
        if not isinstance(payload, Mapping):
            raise TensorRTValidationError("TensorRT runtime tuning config must be an object")
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "fixed": dict(self.fixed),
            "variants": [variant.to_mapping() for variant in self.variants],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """Edge-LLM 低层 benchmark 输出的一条样本记录。"""

    sample_id: str
    repetition: int
    status: str
    output_tokens: int | None
    timings_ms: Mapping[str, float]
    failure: Mapping[str, Any] | None = None
    decode_tokens: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], line_number: int) -> "BenchmarkSample":
        required_fields = {"sample_id", "repetition", "status", "output_tokens", "timings_ms"}
        actual_fields = set(payload)
        missing_fields = required_fields - actual_fields
        unexpected_fields = actual_fields - required_fields - {"failure", "decode_tokens"}
        if missing_fields or unexpected_fields:
            raise TensorRTValidationError(
                f"invalid benchmark line {line_number} fields; "
                f"missing={sorted(missing_fields)}, unexpected={sorted(unexpected_fields)}"
            )
        sample_id = _parse_text(payload["sample_id"], f"benchmark line {line_number}.sample_id")
        repetition = _parse_positive_int(payload["repetition"], f"benchmark line {line_number}.repetition")
        status = _parse_text(payload["status"], f"benchmark line {line_number}.status")
        if status not in {"completed", "failed"}:
            raise TensorRTValidationError(f"benchmark line {line_number}.status must be completed or failed")
        output_tokens = payload["output_tokens"]
        if output_tokens is not None:
            output_tokens = _parse_non_negative_int(output_tokens, f"benchmark line {line_number}.output_tokens")
        decode_tokens = payload.get("decode_tokens")
        if decode_tokens is not None:
            decode_tokens = _parse_non_negative_int(
                decode_tokens, f"benchmark line {line_number}.decode_tokens"
            )
            if output_tokens is not None and decode_tokens > output_tokens:
                raise TensorRTValidationError(
                    f"benchmark line {line_number}.decode_tokens exceeds output_tokens"
                )
        timings_payload = _require_mapping(payload["timings_ms"], f"benchmark line {line_number}.timings_ms")
        timings: dict[str, float] = {}
        for name, value in timings_payload.items():
            if not isinstance(name, str) or not name.strip():
                raise TensorRTValidationError(f"benchmark line {line_number}.timings_ms contains blank name")
            normalized_name = {"ttft_ms": "time_to_first_token_ms", "e2e_ms": "end_to_end_ms"}.get(name, name)
            timings[normalized_name] = _parse_non_negative_float(
                value, f"benchmark line {line_number}.timings_ms.{name}"
            )
        failure = payload.get("failure")
        if failure is not None and not isinstance(failure, Mapping):
            raise TensorRTValidationError(
                f"benchmark line {line_number}.failure must be an object or null"
            )
        return cls(sample_id, repetition, status, output_tokens, timings, failure, decode_tokens)


def load_benchmark_samples(path: Path | str) -> tuple[BenchmarkSample, ...]:
    """读取 Edge-LLM 低层 runtime 输出的 JSONL。"""
    samples: list[BenchmarkSample] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TensorRTValidationError(f"cannot read benchmark JSONL: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise TensorRTValidationError(f"invalid benchmark JSON at line {line_number}") from error
        if not isinstance(payload, Mapping):
            raise TensorRTValidationError(f"benchmark line {line_number} must be an object")
        samples.append(BenchmarkSample.from_mapping(payload, line_number))
    if not samples:
        raise TensorRTValidationError("benchmark JSONL contains no samples")
    return tuple(samples)


def summarize_benchmark_samples(
    samples: Sequence[BenchmarkSample],
    *,
    metadata: Mapping[str, Any] | None = None,
    warmup_samples: Sequence[BenchmarkSample] | None = None,
) -> dict[str, Any]:
    """汇总低层阶段时延，同时区分正式样本和 warm-up 样本。"""
    completed = [sample for sample in samples if sample.status == "completed"]
    stage_values: dict[str, list[float]] = {}
    for sample in completed:
        for stage, value in sample.timings_ms.items():
            stage_values.setdefault(stage, []).append(value)
    output_values = [
        float(sample.output_tokens)
        for sample in completed
        if sample.output_tokens is not None
    ]
    repetition_summaries = _summarize_repetitions(samples)
    warmup = list(warmup_samples or ())
    warmup_completed = [sample for sample in warmup if sample.status == "completed"]
    first_warmup_end_to_end_ms = next(
        (
            sample.timings_ms["end_to_end_ms"]
            for sample in warmup_completed
            if "end_to_end_ms" in sample.timings_ms
        ),
        None,
    )
    return {
        "schema_version": "parksight_tensorrt_benchmark_v1",
        "metadata": dict(metadata or {}),
        "execution": {
            "sample_count": len(samples),
            "completed_sample_count": len(completed),
            "failed_sample_count": len(samples) - len(completed),
            "failed_sample_ids": [sample.sample_id for sample in samples if sample.status == "failed"],
            "failed_categories": dict(
                sorted(
                    Counter(
                        str(sample.failure.get("category", "unknown"))
                        for sample in samples
                        if sample.status == "failed" and sample.failure is not None
                    ).items()
                )
            ),
            # 不能把 warm-up 后的第一个正式样本伪装成 cold start；没有单独的
            # warm-up/启动证据时显式保留 null。
            "cold_start_ms": first_warmup_end_to_end_ms,
            "output_tokens": _numeric_summary(output_values),
            "stage_latency_ms": {
                stage: _numeric_summary(values) for stage, values in sorted(stage_values.items())
            },
            **_sample_token_rates(completed),
            # Run-level throughput is supplied by the low-level runner. It is
            # intentionally kept separate from request-level E2E tok/s: under
            # concurrency, summing per-request rates would overcount overlap.
            "concurrency": _metadata_integer(metadata, "concurrency"),
            "steady_state_wall_clock_ms": _metadata_number(
                metadata, "steady_state_wall_clock_ms"
            ),
            "steady_state_requests": _metadata_integer(
                metadata, "steady_state_requests"
            ),
            "steady_state_completed_requests": _metadata_integer(
                metadata, "steady_state_completed_requests"
            ),
            "steady_state_output_tokens": _metadata_integer(
                metadata, "steady_state_output_tokens"
            ),
            "aggregate_output_tokens_per_second": _metadata_number(
                metadata, "aggregate_output_tokens_per_second"
            ),
            "repetitions": repetition_summaries,
        },
        "warmup": {
            "sample_count": len(warmup),
            "completed_sample_count": len(warmup_completed),
            "failed_sample_count": len(warmup) - len(warmup_completed),
            "first_completed_end_to_end_ms": first_warmup_end_to_end_ms,
            "evidence_boundary": (
                "cold_start_ms is populated only from the separately supplied warm-up JSONL; "
                "it measures the first request after the server became ready, not process launch"
            ),
        },
    }


def _summarize_repetitions(
    samples: Sequence[BenchmarkSample],
) -> dict[str, dict[str, Any]]:
    """按输入 repetition 保留独立分位数，避免总体统计掩盖单次异常。"""
    repetition_ids = sorted({sample.repetition for sample in samples})
    summaries: dict[str, dict[str, Any]] = {}
    for repetition in repetition_ids:
        group = [sample for sample in samples if sample.repetition == repetition]
        completed = [sample for sample in group if sample.status == "completed"]
        stage_values: dict[str, list[float]] = {}
        for sample in completed:
            for stage, value in sample.timings_ms.items():
                stage_values.setdefault(stage, []).append(value)
        summaries[str(repetition)] = {
            "sample_count": len(group),
            "completed_sample_count": len(completed),
            "failed_sample_count": len(group) - len(completed),
            "stage_latency_ms": {
                stage: _numeric_summary(values)
                for stage, values in sorted(stage_values.items())
            },
            **_sample_token_rates(completed),
        }
    return summaries


def sha256_file(path: Path | str) -> str:
    """计算 engine 或其他产物的 SHA-256。"""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TensorRTValidationError(f"cannot read artifact for hashing: {path}") from error
    return digest.hexdigest()


def build_engine_provenance(
    *, engine_path: Path | str, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """生成不可变 engine 文件身份和构建/runtime 参数快照。"""
    path = Path(engine_path).resolve()
    if not path.is_file():
        raise TensorRTValidationError(f"engine file not found: {path}")
    return {
        "schema_version": "parksight_tensorrt_engine_provenance_v1",
        "engine": {
            "path": str(path),
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "configuration": dict(metadata),
    }


def write_json(payload: Mapping[str, Any], path: Path | str) -> None:
    """以 UTF-8 写入结构化 TensorRT 证据。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sample_token_rates(samples: Sequence[BenchmarkSample]) -> dict[str, Any]:
    """Pair counts and durations within each request; expose measurement coverage."""
    result: dict[str, Any] = {}
    for name, count_field, duration_field in (
        ("decode_tokens_per_second", "output_tokens", "decode_ms"),
        ("end_to_end_tokens_per_second", "output_tokens", "end_to_end_ms"),
        ("measured_decode_tokens_per_second", "decode_tokens", "decode_ms"),
    ):
        pairs = [
            (getattr(sample, count_field), sample.timings_ms[duration_field])
            for sample in samples
            if getattr(sample, count_field) is not None
            and duration_field in sample.timings_ms
        ]
        duration_ms = sum(duration for _, duration in pairs)
        result[name] = (
            sum(count for count, _ in pairs) / (duration_ms / 1000.0)
            if duration_ms > 0 else None
        )
        result[name + "_sample_count"] = len(pairs)
    result["decode_rate_evidence_boundary"] = (
        "decode_tokens_per_second is legacy output_tokens/decode_ms; "
        "measured_decode_tokens_per_second requires explicit decode_tokens. "
        "Rates use only count/duration pairs from the same completed request."
    )
    return result


def _metadata_number(metadata: Mapping[str, Any] | None, key: str) -> float | None:
    """读取 runner 元数据中的非负数；缺失或非法值保持为 null。"""
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _metadata_integer(metadata: Mapping[str, Any] | None, key: str) -> int | None:
    """读取 runner 元数据中的非负整数；避免把并发吞吐元数据当作样本事实。"""
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value

def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "p50": None, "p90": None, "p99": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "maximum": ordered[-1],
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _parse_profile_variant(value: Any, index: int) -> tuple[int, int]:
    payload = _require_mapping(value, f"sweep.profile_variants[{index}]")
    _require_exact_fields(payload, {"max_input_len", "max_kv_cache_capacity"}, f"sweep.profile_variants[{index}]")
    return (
        _parse_positive_int(payload["max_input_len"], f"sweep.profile_variants[{index}].max_input_len"),
        _parse_positive_int(payload["max_kv_cache_capacity"], f"sweep.profile_variants[{index}].max_kv_cache_capacity"),
    )


def _require_exact_fields(payload: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise TensorRTValidationError(
            f"invalid {context} fields; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TensorRTValidationError(f"{context} must be an object")
    return value


def _require_sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TensorRTValidationError(f"{context} must be a non-empty array")
    return value


def _parse_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TensorRTValidationError(f"{context} must be a non-blank string")
    return value.strip()


def _parse_text_sequence(value: Any, context: str) -> tuple[str, ...]:
    return tuple(_parse_text(item, f"{context}[{index}]") for index, item in enumerate(_require_sequence(value, context)))


def _parse_text_sequence_allow_empty(value: Any, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TensorRTValidationError(f"{context} must be an array")
    return tuple(_parse_text(item, f"{context}[{index}]") for index, item in enumerate(value))


def _parse_choice_sequence(value: Any, context: str, *, choices: set[str]) -> tuple[str, ...]:
    values = _parse_text_sequence(value, context)
    invalid = sorted(set(values) - choices)
    if invalid:
        raise TensorRTValidationError(f"{context} contains unsupported values: {invalid}")
    return values


def _parse_optional_choice_int(
    value: Any,
    context: str,
    *,
    choices: set[int],
) -> int | None:
    if value is None:
        return None
    parsed = _parse_positive_int(value, context)
    if parsed not in choices:
        raise TensorRTValidationError(
            f"{context} must be one of {sorted(choices)}"
        )
    return parsed


def _require_fields(
    payload: Mapping[str, Any],
    required: set[str],
    allowed: set[str],
    context: str,
) -> None:
    actual = set(payload)
    missing = required - actual
    unexpected = actual - allowed
    if missing or unexpected:
        raise TensorRTValidationError(
            f"invalid {context} fields; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _parse_positive_int_sequence(value: Any, context: str, *, allow_zero: bool = False, maximum: int | None = None) -> tuple[int, ...]:
    result = tuple(
        _parse_non_negative_int(item, f"{context}[{index}]")
        if allow_zero
        else _parse_positive_int(item, f"{context}[{index}]")
        for index, item in enumerate(_require_sequence(value, context))
    )
    if maximum is not None and any(item > maximum for item in result):
        raise TensorRTValidationError(f"{context} values must be <= {maximum}")
    return result


def _parse_positive_int(value: Any, context: str) -> int:
    result = _parse_non_negative_int(value, context)
    if result == 0:
        raise TensorRTValidationError(f"{context} must be positive")
    return result


def _parse_non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TensorRTValidationError(f"{context} must be a non-negative integer")
    return value


def _parse_non_negative_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise TensorRTValidationError(f"{context} must be a non-negative number")
    return float(value)


def _parse_fraction(value: Any, context: str) -> float:
    result = _parse_non_negative_float(value, context)
    if result > 1:
        raise TensorRTValidationError(f"{context} must be between 0 and 1")
    return result


def _parse_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise TensorRTValidationError(f"{context} must be boolean")
    return value
