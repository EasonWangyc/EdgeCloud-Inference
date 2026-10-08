"""后端无关的推理执行与失败记录。"""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import AssessmentValidationError, ParkingAssessment
from parksight_vlm.casebook import CasebookValidationError, ParkingCase
from parksight_vlm.workload import FrozenWorkload


class RuntimeFailureCategory(str, Enum):
    """研究和报告使用的稳定失败类别。"""

    INPUT_ERROR = "input_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    JSON_PARSE_ERROR = "json_parse_error"
    MODEL_REFUSAL = "model_refusal"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    RUNTIME_ERROR = "runtime_error"


class RuntimeDependencyError(RuntimeError):
    """当缺少必要的 Runtime 依赖时由 Adapter 抛出。"""


class RuntimeInputError(RuntimeError):
    """图片内容无效，不能作为运行时输入。"""


class RuntimeUnsupportedError(RuntimeError):
    """当 Runtime 无法执行模型算子或配置时抛出。"""


class RuntimeRefusalError(RuntimeError):
    """当后端明确识别出模型拒答时抛出。"""


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """标识一个可执行模型 Runtime 的配置。"""

    backend: str
    backend_revision: str
    model_id: str
    model_revision: str
    adapter_revision: str
    precision: str

    def __post_init__(self) -> None:
        for field_name in (
            "backend",
            "backend_revision",
            "model_id",
            "model_revision",
            "adapter_revision",
            "precision",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"runtime identity {field_name} must not be blank")

    def to_mapping(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "backend_revision": self.backend_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_revision": self.adapter_revision,
            "precision": self.precision,
        }


@dataclass(frozen=True, slots=True)
class StageTimings:
    """可选的阶段级时延事实，单位为毫秒。"""

    preprocess_ms: float | None = None
    vision_encode_ms: float | None = None
    model_generate_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    http_round_trip_ms: float | None = None
    backend_end_to_end_ms: float | None = None
    end_to_end_ms: float | None = None
    time_to_first_token_ms: float | None = None
    routing_ms: float | None = None
    client_total_ttft_ms: float | None = None
    client_tpot_ms: float | None = None

    def __post_init__(self) -> None:
        for field_name, value in self.to_mapping().items():
            _validate_measurement(value, field_name)

    def to_mapping(self) -> dict[str, float | None]:
        return {
            "preprocess_ms": self.preprocess_ms,
            "vision_encode_ms": self.vision_encode_ms,
            "model_generate_ms": self.model_generate_ms,
            "prefill_ms": self.prefill_ms,
            "decode_ms": self.decode_ms,
            "http_round_trip_ms": self.http_round_trip_ms,
            "backend_end_to_end_ms": self.backend_end_to_end_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "routing_ms": self.routing_ms,
            "client_total_ttft_ms": self.client_total_ttft_ms,
            "client_tpot_ms": self.client_tpot_ms,
        }


@dataclass(frozen=True, slots=True)
class StreamTimings:
    """非空 SSE 文本事件到达时间，距 HTTP 请求开始的毫秒数；不是 token 边界。"""

    content_arrival_ms: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for value in self.content_arrival_ms:
            if value is None:
                raise ValueError("stream arrival time must be numeric")
            _validate_measurement(value, "content_arrival_ms")
        if any(b < a for a, b in zip(self.content_arrival_ms, self.content_arrival_ms[1:])):
            raise ValueError("stream arrival times must be monotonic")

    @property
    def chunk_intervals_ms(self) -> tuple[float, ...]:
        return tuple(b - a for a, b in zip(self.content_arrival_ms, self.content_arrival_ms[1:]))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "content_chunk_count": len(self.content_arrival_ms),
            "content_arrival_ms": list(self.content_arrival_ms),
            "chunk_intervals_ms": list(self.chunk_intervals_ms),
        }


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """由后端或板端探针采集的可选 Runtime 资源事实。"""

    peak_memory_mb: float | None = None
    average_power_w: float | None = None
    peak_temperature_c: float | None = None

    def __post_init__(self) -> None:
        for field_name, value in self.to_mapping().items():
            _validate_measurement(value, field_name)

    def to_mapping(self) -> dict[str, float | None]:
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "average_power_w": self.average_power_w,
            "peak_temperature_c": self.peak_temperature_c,
        }


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    """分类保存的 Runtime 失败，用于避免伪造业务输出。"""

    category: RuntimeFailureCategory
    message: str
    exception_type: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "category": self.category.value,
            "message": self.message,
            "exception_type": self.exception_type,
        }


@dataclass(frozen=True, slots=True)
class RuntimeGeneration:
    """后端原始生成结果及执行期间的实测事实。"""

    raw_output: str
    stage_timings: StageTimings = field(default_factory=StageTimings)
    resource_snapshot: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    output_tokens: int | None = None
    input_tokens: int | None = None
    stream_timings: StreamTimings = field(default_factory=StreamTimings)
    decode_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.raw_output, str):
            raise TypeError("raw_output must be a string")
        if self.output_tokens is not None and (
            isinstance(self.output_tokens, bool)
            or not isinstance(self.output_tokens, int)
            or self.output_tokens < 0
        ):
            raise ValueError("output_tokens must be a non-negative integer")
        if self.input_tokens is not None and (
            isinstance(self.input_tokens, bool)
            or not isinstance(self.input_tokens, int)
            or self.input_tokens < 0
        ):
            raise ValueError("input_tokens must be a non-negative integer")
        _validate_decode_tokens(self.decode_tokens, self.output_tokens)


@dataclass(frozen=True, slots=True)
class InferenceRecord:
    """一次 Runtime 执行产生的全部可用事实。"""

    case_id: str
    runtime_identity: RuntimeIdentity
    workload_identity: str
    started_at_utc: str
    assessment: ParkingAssessment | None
    failure: RuntimeFailure | None
    raw_output: str | None
    stage_timings: StageTimings
    resource_snapshot: ResourceSnapshot
    output_tokens: int | None
    routing_decision: dict[str, Any] | None = None
    input_tokens: int | None = None
    stream_timings: StreamTimings = field(default_factory=StreamTimings)
    decode_tokens: int | None = None

    def __post_init__(self) -> None:
        if (self.assessment is None) == (self.failure is None):
            raise ValueError("record must contain exactly one of assessment or failure")
        _validate_decode_tokens(self.decode_tokens, self.output_tokens)

    @property
    def succeeded(self) -> bool:
        return self.assessment is not None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "runtime_identity": self.runtime_identity.to_mapping(),
            "workload_identity": self.workload_identity,
            "started_at_utc": self.started_at_utc,
            "assessment": (
                self.assessment.to_mapping() if self.assessment is not None else None
            ),
            "failure": self.failure.to_mapping() if self.failure is not None else None,
            "raw_output": self.raw_output,
            "stage_timings": self.stage_timings.to_mapping(),
            "resource_snapshot": self.resource_snapshot.to_mapping(),
            "output_tokens": self.output_tokens,
            "routing_decision": self.routing_decision,
            "input_tokens": self.input_tokens,
            "stream_timings": self.stream_timings.to_mapping(),
            "decode_tokens": self.decode_tokens,
        }


class RiskRuntime(ABC):
    """将后端执行转换为推理记录的模板接口。"""

    def __init__(self, *, data_root: Path, identity: RuntimeIdentity) -> None:
        self._data_root = data_root
        self._identity = identity

    @property
    def identity(self) -> RuntimeIdentity:
        return self._identity

    def close(self) -> None:
        """释放运行时资源；没有连接资源的 Adapter 无需实现。"""

    def analyze(self, case: ParkingCase, workload: FrozenWorkload) -> InferenceRecord:
        """执行一个样本，并保存校验后的输出或失败事实。"""
        started_at_utc = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        generation: RuntimeGeneration | None = None
        raw_output: str | None = None
        assessment: ParkingAssessment | None = None
        failure: RuntimeFailure | None = None
        try:
            image_path = case.resolve_image(self._data_root, require_exists=True)
            generation = self._generate(image_path=image_path, workload=workload)
            raw_output = generation.raw_output
            payload = json.loads(raw_output)
            assessment = ParkingAssessment.from_mapping(payload)
        except Exception as error:  # 推理记录是明确的 Runtime 失败边界。
            failure = _classify_failure(error)

        end_to_end_ms = (time.perf_counter() - start_time) * 1000.0
        if generation is None:
            timings = StageTimings(end_to_end_ms=end_to_end_ms)
            resources = ResourceSnapshot()
            output_tokens = None
        else:
            timings = replace(generation.stage_timings, end_to_end_ms=end_to_end_ms)
            resources = generation.resource_snapshot
            output_tokens = generation.output_tokens

        return InferenceRecord(
            case_id=case.case_id,
            runtime_identity=self.identity,
            workload_identity=workload.identity,
            started_at_utc=started_at_utc,
            assessment=assessment,
            failure=failure,
            raw_output=raw_output,
            stage_timings=timings,
            resource_snapshot=resources,
            output_tokens=output_tokens,
            input_tokens=generation.input_tokens if generation is not None else None,
            stream_timings=generation.stream_timings if generation is not None else StreamTimings(),
            decode_tokens=generation.decode_tokens if generation is not None else None,
        )

    @abstractmethod
    def _generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        """调用具体后端，并且只返回实际测得的事实。"""


def _validate_decode_tokens(value: int | None, output_tokens: int | None) -> None:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("decode_tokens must be a non-negative integer")
        if output_tokens is not None and value > output_tokens:
            raise ValueError("decode_tokens cannot exceed output_tokens")


def _validate_measurement(value: float | None, field_name: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be finite and non-negative")


def _classify_failure(error: Exception) -> RuntimeFailure:
    message = str(error) or error.__class__.__name__
    lowered_message = message.lower()
    if isinstance(error, (CasebookValidationError, RuntimeInputError)):
        category = RuntimeFailureCategory.INPUT_ERROR
    elif isinstance(error, RuntimeDependencyError):
        category = RuntimeFailureCategory.DEPENDENCY_UNAVAILABLE
    elif isinstance(error, (json.JSONDecodeError, AssessmentValidationError)):
        category = RuntimeFailureCategory.JSON_PARSE_ERROR
    elif isinstance(error, RuntimeRefusalError):
        category = RuntimeFailureCategory.MODEL_REFUSAL
    elif isinstance(error, TimeoutError):
        category = RuntimeFailureCategory.TIMEOUT
    elif isinstance(error, MemoryError) or "out of memory" in lowered_message:
        category = RuntimeFailureCategory.OUT_OF_MEMORY
    elif isinstance(error, RuntimeUnsupportedError) or "unsupported operator" in lowered_message:
        category = RuntimeFailureCategory.UNSUPPORTED_OPERATOR
    else:
        category = RuntimeFailureCategory.RUNTIME_ERROR
    return RuntimeFailure(
        category=category,
        message=message,
        exception_type=error.__class__.__name__,
    )
