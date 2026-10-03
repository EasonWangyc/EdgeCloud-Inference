"""可审计的 Edge-Cloud Runtime 选择、调用和失败回退。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit

from parksight_vlm.assessment import ParkingAssessment, ParkingRiskEvent, RiskLevel
from parksight_vlm.casebook import ParkingCase
from parksight_vlm.workload import FrozenWorkload

from .runtime import (
    InferenceRecord,
    ResourceSnapshot,
    RiskRuntime,
    RuntimeFailure,
    RuntimeFailureCategory,
    RuntimeGeneration,
    RuntimeIdentity,
    StageTimings,
)


class RoutingSignalsProvider(Protocol):
    """每次请求提供隐私授权、设备状态和云链路状态。"""

    def get_signals(
        self, *, case: ParkingCase, workload: FrozenWorkload
    ) -> "RoutingSignals":
        """返回本次决策使用的状态快照。"""


@dataclass(frozen=True, slots=True)
class RoutingSignals:
    """路由决策输入；cloud_allowed 是上传图像的显式授权。"""

    cloud_allowed: bool
    network_available: bool
    edge_available: bool
    cloud_available: bool
    edge_free_memory_mb: float | None = None
    edge_temperature_c: float | None = None
    network_rtt_ms: float | None = None
    packet_loss_percent: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "cloud_allowed",
            "network_available",
            "edge_available",
            "cloud_available",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"routing signal {name} must be boolean")
        for name in (
            "edge_free_memory_mb",
            "edge_temperature_c",
            "network_rtt_ms",
            "packet_loss_percent",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"routing signal {name} must be non-negative")
        if self.packet_loss_percent is not None and self.packet_loss_percent > 100:
            raise ValueError("routing signal packet_loss_percent must be <= 100")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RoutingSignals":
        allowed = {
            "cloud_allowed",
            "network_available",
            "edge_available",
            "cloud_available",
            "edge_free_memory_mb",
            "edge_temperature_c",
            "network_rtt_ms",
            "packet_loss_percent",
        }
        unexpected = set(payload) - allowed
        if unexpected:
            raise ValueError(f"unsupported routing signals: {sorted(unexpected)}")
        required = {
            "cloud_allowed",
            "network_available",
            "edge_available",
            "cloud_available",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"missing routing signals: {sorted(missing)}")
        return cls(**payload)

    def to_mapping(self) -> dict[str, bool | float | None]:
        return {
            "cloud_allowed": self.cloud_allowed,
            "network_available": self.network_available,
            "edge_available": self.edge_available,
            "cloud_available": self.cloud_available,
            "edge_free_memory_mb": self.edge_free_memory_mb,
            "edge_temperature_c": self.edge_temperature_c,
            "network_rtt_ms": self.network_rtt_ms,
            "packet_loss_percent": self.packet_loss_percent,
        }


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """基于 SLA 与资源门槛选择 primary，并限制自动回退条件。"""

    preference: str = "edge_first"
    min_edge_free_memory_mb: float = 1024.0
    max_edge_temperature_c: float = 82.0
    max_network_rtt_ms: float = 250.0
    max_packet_loss_percent: float = 2.0
    cloud_escalation_risk_levels: tuple[str, ...] = (RiskLevel.HIGH.value,)
    cloud_escalation_events: tuple[str, ...] = ()
    fallback_on: tuple[RuntimeFailureCategory, ...] = (
        RuntimeFailureCategory.DEPENDENCY_UNAVAILABLE,
        RuntimeFailureCategory.JSON_PARSE_ERROR,
        RuntimeFailureCategory.OUT_OF_MEMORY,
        RuntimeFailureCategory.TIMEOUT,
        RuntimeFailureCategory.UNSUPPORTED_OPERATOR,
        RuntimeFailureCategory.RUNTIME_ERROR,
    )

    def __post_init__(self) -> None:
        if self.preference not in {"edge_first", "cloud_first"}:
            raise ValueError("routing preference must be edge_first or cloud_first")
        for name in (
            "min_edge_free_memory_mb",
            "max_edge_temperature_c",
            "max_network_rtt_ms",
            "max_packet_loss_percent",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if any(
            level not in {item.value for item in RiskLevel}
            for level in self.cloud_escalation_risk_levels
        ):
            raise ValueError("cloud_escalation_risk_levels contains an unknown risk level")
        if any(
            event not in {item.value for item in ParkingRiskEvent}
            for event in self.cloud_escalation_events
        ):
            raise ValueError("cloud_escalation_events contains an unknown event")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RoutingPolicy":
        allowed = {
            "preference",
            "min_edge_free_memory_mb",
            "max_edge_temperature_c",
            "max_network_rtt_ms",
            "max_packet_loss_percent",
            "cloud_escalation_risk_levels",
            "cloud_escalation_events",
            "fallback_on",
        }
        unexpected = set(payload) - allowed
        if unexpected:
            raise ValueError(f"unsupported routing policy fields: {sorted(unexpected)}")
        values = dict(payload)
        for field_name in (
            "cloud_escalation_risk_levels",
            "cloud_escalation_events",
        ):
            if field_name in values:
                items = values[field_name]
                if not isinstance(items, list) or any(
                    not isinstance(item, str) for item in items
                ):
                    raise ValueError(f"{field_name} must be an array of strings")
                values[field_name] = tuple(items)
        if "fallback_on" in values:
            categories = values["fallback_on"]
            if not isinstance(categories, list) or any(
                not isinstance(value, str) for value in categories
            ):
                raise ValueError("fallback_on must be an array of failure category names")
            try:
                values["fallback_on"] = tuple(
                    RuntimeFailureCategory(value) for value in categories
                )
            except ValueError as error:
                raise ValueError("fallback_on contains an unknown failure category") from error
        return cls(**values)

    def eligible_routes(self, signals: RoutingSignals) -> tuple[bool, bool, str, str]:
        edge_ok = signals.edge_available
        if (
            signals.edge_free_memory_mb is not None
            and signals.edge_free_memory_mb < self.min_edge_free_memory_mb
        ):
            edge_ok = False
        if (
            signals.edge_temperature_c is not None
            and signals.edge_temperature_c > self.max_edge_temperature_c
        ):
            edge_ok = False
        edge_reason = "edge_healthy" if edge_ok else "edge_unavailable_or_resource_limit"

        network_ok = signals.network_available
        if (
            signals.network_rtt_ms is not None
            and signals.network_rtt_ms > self.max_network_rtt_ms
        ):
            network_ok = False
        if (
            signals.packet_loss_percent is not None
            and signals.packet_loss_percent > self.max_packet_loss_percent
        ):
            network_ok = False
        cloud_ok = signals.cloud_allowed and network_ok and signals.cloud_available
        if not signals.cloud_allowed:
            cloud_reason = "cloud_upload_not_authorized"
        elif not network_ok:
            cloud_reason = "network_unavailable_or_over_sla"
        elif not signals.cloud_available:
            cloud_reason = "cloud_runtime_unavailable"
        else:
            cloud_reason = "cloud_healthy"
        return edge_ok, cloud_ok, edge_reason, cloud_reason

    def to_mapping(self) -> dict[str, Any]:
        return {
            "preference": self.preference,
            "min_edge_free_memory_mb": self.min_edge_free_memory_mb,
            "max_edge_temperature_c": self.max_edge_temperature_c,
            "max_network_rtt_ms": self.max_network_rtt_ms,
            "max_packet_loss_percent": self.max_packet_loss_percent,
            "cloud_escalation_risk_levels": list(
                self.cloud_escalation_risk_levels
            ),
            "cloud_escalation_events": list(self.cloud_escalation_events),
            "fallback_on": [category.value for category in self.fallback_on],
        }

    def cloud_escalation_reason(
        self, assessment: ParkingAssessment
    ) -> str | None:
        if assessment.risk_level.value in self.cloud_escalation_risk_levels:
            return f"risk_level:{assessment.risk_level.value}"
        triggered_events = sorted(
            event.value
            for event in assessment.events
            if event.value in self.cloud_escalation_events
        )
        if triggered_events:
            return "events:" + ",".join(triggered_events)
        return None


class StaticRoutingSignalsProvider:
    """Study 配置适配器；运行时状态可以替换为实时采样 provider。"""

    def __init__(self, signals: RoutingSignals) -> None:
        self._signals = signals

    def get_signals(
        self, *, case: ParkingCase, workload: FrozenWorkload
    ) -> RoutingSignals:
        return self._signals


class HttpHealthRoutingSignalsProvider:
    """Refreshes Edge/vLLM readiness over HTTP for each routed request.

    Device headroom and privacy authorization remain explicit inputs. Readiness
    probes make service availability current; an actual inference failure still
    controls fallback, since health endpoints cannot guarantee model execution.
    """

    def __init__(
        self,
        *,
        edge_base_url: str,
        cloud_base_url: str,
        base_signals: RoutingSignals,
        cloud_api_key: str | None = None,
        timeout_seconds: float = 1.0,
        health_path: str = "/health",
    ) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("health probe timeout must be finite and positive")
        if not health_path.startswith("/"):
            raise ValueError("health_path must start with '/'")
        self._edge_health_url = self._health_url(edge_base_url, health_path)
        self._cloud_health_url = self._health_url(cloud_base_url, health_path)
        self._base_signals = base_signals
        self._cloud_api_key = cloud_api_key
        self._timeout_seconds = timeout_seconds

    def get_signals(
        self, *, case: ParkingCase, workload: FrozenWorkload
    ) -> RoutingSignals:
        edge_available = self._probe(self._edge_health_url)
        cloud_available = False
        if self._base_signals.cloud_allowed and self._base_signals.network_available:
            cloud_headers = (
                {"Authorization": f"Bearer {self._cloud_api_key}"}
                if self._cloud_api_key
                else None
            )
            cloud_available = self._probe(
                self._cloud_health_url, headers=cloud_headers
            )
        return replace(
            self._base_signals,
            edge_available=edge_available,
            cloud_available=cloud_available,
        )

    def _probe(self, url: str, *, headers: dict[str, str] | None = None) -> bool:
        request_headers = {"Accept": "application/json"}
        if headers:
            request_headers.update(headers)
        request = Request(url, headers=request_headers, method="GET")
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                return 200 <= int(getattr(response, "status", 200)) < 400
        except Exception:
            return False

    @staticmethod
    def _health_url(base_url: str, health_path: str) -> str:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("health probe base URL must be absolute HTTP(S)")
        base_path = parsed.path.rstrip("/")
        for suffix in ("/v1/chat/completions", "/v1"):
            if base_path.endswith(suffix):
                base_path = base_path[: -len(suffix)]
                break
        normalized_path = f"{base_path}{health_path}"
        return urlunsplit(
            (parsed.scheme, parsed.netloc, normalized_path, "", "")
        )


class EdgeCloudRouterRuntime(RiskRuntime):
    """将两个 Runtime Adapter 组合成可回退、保留决策证据的路由 Runtime。"""

    def __init__(
        self,
        *,
        data_root: Path,
        edge_runtime: RiskRuntime,
        cloud_runtime: RiskRuntime,
        signals_provider: RoutingSignalsProvider,
        policy: RoutingPolicy,
        backend_revision: str,
        model_id: str,
        model_revision: str,
        adapter_revision: str = "none",
    ) -> None:
        super().__init__(
            data_root=data_root,
            identity=RuntimeIdentity(
                backend="edge_vllm_router",
                backend_revision=backend_revision,
                model_id=model_id,
                model_revision=model_revision,
                adapter_revision=adapter_revision,
                precision="routed",
            ),
        )
        self._edge_runtime = edge_runtime
        self._cloud_runtime = cloud_runtime
        self._signals_provider = signals_provider
        self._policy = policy

    def _generate(
        self, *, image_path: Path, workload: FrozenWorkload
    ) -> RuntimeGeneration:
        raise RuntimeError("EdgeCloudRouterRuntime dispatches through child runtimes")

    def analyze(self, case: ParkingCase, workload: FrozenWorkload) -> InferenceRecord:
        route_started = time.perf_counter()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        try:
            signals = self._signals_provider.get_signals(case=case, workload=workload)
            edge_ok, cloud_ok, edge_reason, cloud_reason = self._policy.eligible_routes(
                signals
            )
            primary_name, decision_reason = self._select_primary(
                edge_ok=edge_ok,
                cloud_ok=cloud_ok,
                edge_reason=edge_reason,
                cloud_reason=cloud_reason,
            )
        except Exception as error:
            return self._routing_failure(
                case=case,
                workload=workload,
                started_at_utc=started_at_utc,
                route_started=route_started,
                reason="routing_signal_provider_failed",
                message=str(error) or error.__class__.__name__,
            )

        decision_ms = (time.perf_counter() - route_started) * 1000.0
        if primary_name is None:
            return self._routing_failure(
                case=case,
                workload=workload,
                started_at_utc=started_at_utc,
                route_started=route_started,
                reason="no_eligible_runtime",
                message=f"edge={edge_reason}; cloud={cloud_reason}",
                routing_decision={
                    "policy": self._policy.to_mapping(),
                    "signals": signals.to_mapping(),
                    "selected_runtime": None,
                    "reason": decision_reason,
                    "attempts": [],
                    "fallback_used": False,
                },
                decision_ms=decision_ms,
            )

        runtimes = {"edge": self._edge_runtime, "cloud": self._cloud_runtime}
        primary_record = runtimes[primary_name].analyze(case, workload)
        attempts = [self._attempt_mapping(primary_name, primary_record)]
        selected_record = primary_record
        fallback_used = False
        escalation_attempted = False
        escalation_succeeded = False
        escalation_reason: str | None = None
        escalation_skipped_reason: str | None = None
        fallback_name = "cloud" if primary_name == "edge" else "edge"
        fallback_ok = cloud_ok if fallback_name == "cloud" else edge_ok
        fallback_allowed = (
            not primary_record.succeeded
            and primary_record.failure is not None
            and primary_record.failure.category in self._policy.fallback_on
            and fallback_ok
        )
        if fallback_allowed:
            fallback_used = True
            selected_record = runtimes[fallback_name].analyze(case, workload)
            attempts.append(self._attempt_mapping(fallback_name, selected_record))
        elif primary_name == "edge" and primary_record.assessment is not None:
            escalation_reason = self._policy.cloud_escalation_reason(
                primary_record.assessment
            )
            if escalation_reason is not None:
                if signals.cloud_allowed and cloud_ok:
                    escalation_attempted = True
                    cloud_record = self._cloud_runtime.analyze(case, workload)
                    attempts.append(self._attempt_mapping("cloud", cloud_record))
                    if cloud_record.succeeded:
                        selected_record = cloud_record
                        escalation_succeeded = True
                elif not signals.cloud_allowed:
                    escalation_skipped_reason = "cloud_upload_not_authorized"
                else:
                    escalation_skipped_reason = cloud_reason

        routing_decision = {
            "policy": self._policy.to_mapping(),
            "signals": signals.to_mapping(),
            "selected_runtime": (
                selected_record.runtime_identity.backend
            ),
            "primary_runtime": primary_record.runtime_identity.backend,
            "reason": decision_reason,
            "fallback_used": fallback_used,
            "fallback_reason": (
                primary_record.failure.category.value if fallback_used else None
            ),
            "scene_escalation_reason": escalation_reason,
            "scene_escalation_attempted": escalation_attempted,
            "scene_escalation_succeeded": escalation_succeeded,
            "scene_escalation_skipped_reason": escalation_skipped_reason,
            "attempts": attempts,
        }
        elapsed_ms = (time.perf_counter() - route_started) * 1000.0
        timings = replace(
            selected_record.stage_timings,
            end_to_end_ms=elapsed_ms,
            routing_ms=decision_ms,
        )
        return replace(
            selected_record,
            started_at_utc=started_at_utc,
            stage_timings=timings,
            routing_decision=routing_decision,
        )

    def _select_primary(
        self,
        *,
        edge_ok: bool,
        cloud_ok: bool,
        edge_reason: str,
        cloud_reason: str,
    ) -> tuple[str | None, str]:
        preferred = ("edge", "cloud") if self._policy.preference == "edge_first" else (
            "cloud",
            "edge",
        )
        eligible = {"edge": edge_ok, "cloud": cloud_ok}
        reasons = {"edge": edge_reason, "cloud": cloud_reason}
        for name in preferred:
            if eligible[name]:
                return name, f"{self._policy.preference}:{reasons[name]}"
        return None, f"no_eligible_route:edge={edge_reason};cloud={cloud_reason}"

    @staticmethod
    def _attempt_mapping(runtime_name: str, record: InferenceRecord) -> dict[str, Any]:
        return {
            "route": runtime_name,
            "runtime_identity": record.runtime_identity.to_mapping(),
            "succeeded": record.succeeded,
            "failure": record.failure.to_mapping() if record.failure else None,
            "stage_timings": record.stage_timings.to_mapping(),
            "output_tokens": record.output_tokens,
        }

    def _routing_failure(
        self,
        *,
        case: ParkingCase,
        workload: FrozenWorkload,
        started_at_utc: str,
        route_started: float,
        reason: str,
        message: str,
        routing_decision: dict[str, Any] | None = None,
        decision_ms: float | None = None,
    ) -> InferenceRecord:
        elapsed_ms = (time.perf_counter() - route_started) * 1000.0
        if routing_decision is None:
            routing_decision = {
                "policy": self._policy.to_mapping(),
                "selected_runtime": None,
                "reason": reason,
                "attempts": [],
                "fallback_used": False,
            }
        return InferenceRecord(
            case_id=case.case_id,
            runtime_identity=self.identity,
            workload_identity=workload.identity,
            started_at_utc=started_at_utc,
            assessment=None,
            failure=RuntimeFailure(
                category=RuntimeFailureCategory.DEPENDENCY_UNAVAILABLE,
                message=message,
                exception_type="RoutingUnavailableError",
            ),
            raw_output=None,
            stage_timings=StageTimings(
                end_to_end_ms=elapsed_ms,
                routing_ms=(elapsed_ms if decision_ms is None else decision_ms),
            ),
            resource_snapshot=ResourceSnapshot(),
            output_tokens=None,
            routing_decision=routing_decision,
        )
