"""vLLM HTTP Adapter 与端云路由的无硬件契约测试。"""

from __future__ import annotations

import base64
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, patch

from parksight_vlm.assessment import ParkingAssessment
from parksight_vlm.casebook import DatasetSplit, ParkingCase
from parksight_vlm.inference import (
    EdgeCloudRouterRuntime,
    HttpHealthRoutingSignalsProvider,
    ResourceSnapshot,
    RiskRuntime,
    RoutingPolicy,
    RoutingSignals,
    RuntimeFailure,
    RuntimeFailureCategory,
    RuntimeGeneration,
    RuntimeIdentity,
    StageTimings,
    VllmHttpBackend,
    VllmRuntime,
)
from parksight_vlm.workload import FrozenWorkload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = FrozenWorkload.load(
    PROJECT_ROOT / "configs" / "workloads" / "parking_risk_v1.json"
)
IMAGE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "inference" / "scene.jpg"


def assessment(risk_level: str, events: list[str]) -> ParkingAssessment:
    return ParkingAssessment.from_mapping(
        {
            "schema_version": "parking_risk_v1",
            "risk_level": risk_level,
            "events": events,
            "evidence": ["Visible evidence supports this assessment."],
            "driver_advice": ["slow_down"] if risk_level != "low" else ["maintain_observation"],
        }
    )


def test_case() -> ParkingCase:
    return ParkingCase(
        case_id="case-router-001",
        image_ref=PurePosixPath("scene.jpg"),
        source_group_id="group-router-001",
        split=DatasetSplit.TEST,
        reference_assessment=None,
    )


class ScriptedRuntime(RiskRuntime):
    def __init__(self, backend: str, outcomes: list[object]) -> None:
        super().__init__(
            data_root=PROJECT_ROOT,
            identity=RuntimeIdentity(
                backend=backend,
                backend_revision="test-revision",
                model_id="test-model",
                model_revision="model-revision",
                adapter_revision="none",
                precision="test",
            ),
        )
        self.outcomes = list(outcomes)
        self.calls = 0

    def analyze(self, case: ParkingCase, workload: FrozenWorkload):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        started_at = datetime.now(timezone.utc).isoformat()
        if isinstance(outcome, RuntimeFailure):
            return self._record(
                case,
                workload,
                started_at,
                assessment_value=None,
                failure=outcome,
                raw_output=None,
            )
        raw_output = json.dumps(outcome.to_mapping(), ensure_ascii=False)
        return self._record(
            case,
            workload,
            started_at,
            assessment_value=outcome,
            failure=None,
            raw_output=raw_output,
        )

    def _record(
        self,
        case: ParkingCase,
        workload: FrozenWorkload,
        started_at: str,
        *,
        assessment_value: ParkingAssessment | None,
        failure: RuntimeFailure | None,
        raw_output: str | None,
    ):
        from parksight_vlm.inference import InferenceRecord

        return InferenceRecord(
            case_id=case.case_id,
            runtime_identity=self.identity,
            workload_identity=workload.identity,
            started_at_utc=started_at,
            assessment=assessment_value,
            failure=failure,
            raw_output=raw_output,
            stage_timings=StageTimings(end_to_end_ms=10.0),
            resource_snapshot=ResourceSnapshot(),
            output_tokens=8 if raw_output is not None else None,
        )

    def _generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        raise AssertionError("ScriptedRuntime overrides analyze")


class VllmAdapterTests(unittest.TestCase):
    def test_builds_openai_multimodal_json_stream_payload(self) -> None:
        backend = VllmHttpBackend(
            base_url="http://127.0.0.1:8000/v1",
            model_name="served-qwen",
            stream_responses=True,
            json_mode=True,
        )

        payload = backend.build_request_payload(
            image_path=IMAGE_PATH,
            workload=WORKLOAD,
            stream=True,
        )

        self.assertEqual(payload["model"], "served-qwen")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        user_parts = payload["messages"][1]["content"]
        image_url = user_parts[0]["image_url"]["url"]
        encoded = image_url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), IMAGE_PATH.read_bytes())
        self.assertTrue(backend._endpoint.endswith("/v1/chat/completions"))

    def test_normalizes_completion_endpoint_without_duplicating_v1(self) -> None:
        backend = VllmHttpBackend(
            base_url="https://vllm.example/api/v1?tenant=parking",
            model_name="served-qwen",
        )

        self.assertEqual(
            backend._endpoint,
            "https://vllm.example/api/v1/chat/completions?tenant=parking",
        )
        self.assertEqual(
            backend._http_endpoint_path,
            "/api/v1/chat/completions?tenant=parking",
        )

    def test_api_key_requires_tls_for_non_loopback_host(self) -> None:
        with self.assertRaisesRegex(ValueError, "require HTTPS"):
            VllmHttpBackend(
                base_url="http://192.168.1.10:8000",
                model_name="served-qwen",
                api_key="secret",
            )

        VllmHttpBackend(
            base_url="http://localhost:8000",
            model_name="served-qwen",
            api_key="local-test-key",
        )

    def test_streaming_http_response_becomes_validated_inference_record(self) -> None:
        output = json.dumps(
            {
                "schema_version": "parking_risk_v1",
                "risk_level": "medium",
                "events": ["narrow_passage"],
                "evidence": ["Vehicles leave a narrow maneuvering corridor."],
                "driver_advice": ["slow_down"],
            }
        )
        first_chunk = output[: len(output) // 2]
        second_chunk = output[len(output) // 2 :]
        events = [
            {"choices": [{"delta": {"content": first_chunk}}]},
            {"choices": [{"delta": {"content": second_chunk}}]},
            {"choices": [], "usage": {"completion_tokens": 21}},
        ]

        class Headers:
            @staticmethod
            def get_content_type() -> str:
                return "text/event-stream"

        class Response:
            status = 200
            headers = Headers()
            will_close = True

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def __iter__(self):
                for event in events:
                    yield b"data: " + json.dumps(event).encode("utf-8") + b"\n\n"
                yield b"data: [DONE]\n\n"

        backend = VllmHttpBackend(
            base_url="https://vllm.example/v1",
            model_name="served-qwen",
            stream_responses=True,
        )
        runtime = VllmRuntime(
            data_root=IMAGE_PATH.parent,
            backend=backend,
            backend_revision="vllm-test",
            model_id="test-model",
            model_revision="model-revision",
            adapter_revision="none",
            precision="bf16",
        )
        with patch(
            "parksight_vlm.inference.edge_llm.urlopen", return_value=Response()
        ) as post:
            record = runtime.analyze(test_case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(record.assessment.risk_level.value, "medium")
        self.assertEqual(record.assessment.events[0].value, "narrow_passage")
        self.assertEqual(record.output_tokens, 21)
        self.assertIsNotNone(record.stage_timings.time_to_first_token_ms)
        self.assertEqual(
            post.call_args.args[0].full_url,
            "https://vllm.example/v1/chat/completions",
        )


class EdgeCloudRoutingTests(unittest.TestCase):
    def _router(
        self,
        *,
        edge_outcomes: list[object],
        cloud_outcomes: list[object],
        signals: RoutingSignals,
        policy: RoutingPolicy | None = None,
    ) -> tuple[EdgeCloudRouterRuntime, ScriptedRuntime, ScriptedRuntime]:
        edge = ScriptedRuntime("tensorrt_edge_llm", edge_outcomes)
        cloud = ScriptedRuntime("vllm", cloud_outcomes)
        router = EdgeCloudRouterRuntime(
            data_root=PROJECT_ROOT,
            edge_runtime=edge,
            cloud_runtime=cloud,
            signals_provider=Mock(get_signals=Mock(return_value=signals)),
            policy=policy or RoutingPolicy(),
            backend_revision="router-test",
            model_id="test-model",
            model_revision="model-revision",
        )
        return router, edge, cloud

    @staticmethod
    def _signals(*, cloud_allowed: bool) -> RoutingSignals:
        return RoutingSignals(
            cloud_allowed=cloud_allowed,
            network_available=True,
            edge_available=True,
            cloud_available=True,
        )

    def test_low_risk_uses_edge_only(self) -> None:
        router, edge, cloud = self._router(
            edge_outcomes=[assessment("low", [])],
            cloud_outcomes=[],
            signals=self._signals(cloud_allowed=True),
        )

        record = router.analyze(test_case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(edge.calls, 1)
        self.assertEqual(cloud.calls, 0)
        self.assertEqual(record.routing_decision["selected_runtime"], "tensorrt_edge_llm")
        self.assertFalse(record.routing_decision["scene_escalation_attempted"])

    def test_high_risk_escalates_to_cloud_when_authorized(self) -> None:
        router, edge, cloud = self._router(
            edge_outcomes=[assessment("high", ["fixed_obstacle_near_path"])],
            cloud_outcomes=[assessment("medium", ["fixed_obstacle_near_path"])],
            signals=self._signals(cloud_allowed=True),
        )

        record = router.analyze(test_case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(edge.calls, 1)
        self.assertEqual(cloud.calls, 1)
        self.assertEqual(record.runtime_identity.backend, "vllm")
        self.assertEqual(record.routing_decision["scene_escalation_reason"], "risk_level:high")
        self.assertTrue(record.routing_decision["scene_escalation_succeeded"])
        self.assertEqual(len(record.routing_decision["attempts"]), 2)
        self.assertIsNotNone(record.stage_timings.routing_ms)

    def test_configured_event_escalates_even_below_high_risk(self) -> None:
        router, edge, cloud = self._router(
            edge_outcomes=[assessment("medium", ["parking_space_conflict"])],
            cloud_outcomes=[assessment("medium", ["parking_space_conflict"])],
            signals=self._signals(cloud_allowed=True),
            policy=RoutingPolicy(cloud_escalation_events=("parking_space_conflict",)),
        )

        record = router.analyze(test_case(), WORKLOAD)

        self.assertEqual(edge.calls, 1)
        self.assertEqual(cloud.calls, 1)
        self.assertEqual(
            record.routing_decision["scene_escalation_reason"],
            "events:parking_space_conflict",
        )

    def test_failed_cloud_escalation_preserves_successful_edge_assessment(self) -> None:
        edge_assessment = assessment("high", ["fixed_obstacle_near_path"])
        router, edge, cloud = self._router(
            edge_outcomes=[edge_assessment],
            cloud_outcomes=[
                RuntimeFailure(
                    category=RuntimeFailureCategory.TIMEOUT,
                    message="cloud timed out",
                    exception_type="TimeoutError",
                )
            ],
            signals=self._signals(cloud_allowed=True),
        )

        record = router.analyze(test_case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(record.runtime_identity.backend, "tensorrt_edge_llm")
        self.assertEqual(record.assessment, edge_assessment)
        self.assertTrue(record.routing_decision["scene_escalation_attempted"])
        self.assertFalse(record.routing_decision["scene_escalation_succeeded"])
        self.assertEqual(record.routing_decision["attempts"][1]["failure"]["category"], "timeout")

    def test_cloud_first_selects_cloud_when_consent_and_services_are_available(self) -> None:
        router, edge, cloud = self._router(
            edge_outcomes=[assessment("low", [])],
            cloud_outcomes=[assessment("low", [])],
            signals=self._signals(cloud_allowed=True),
            policy=RoutingPolicy(preference="cloud_first"),
        )

        record = router.analyze(test_case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(edge.calls, 0)
        self.assertEqual(cloud.calls, 1)
        self.assertEqual(record.runtime_identity.backend, "vllm")

    def test_privacy_gate_prevents_health_probe_escalation_and_fallback(self) -> None:
        router, edge, cloud = self._router(
            edge_outcomes=[assessment("high", ["vru_near_maneuver_path"])],
            cloud_outcomes=[],
            signals=self._signals(cloud_allowed=False),
        )

        record = router.analyze(test_case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(edge.calls, 1)
        self.assertEqual(cloud.calls, 0)
        self.assertEqual(
            record.routing_decision["scene_escalation_skipped_reason"],
            "cloud_upload_not_authorized",
        )

    def test_runtime_failure_falls_back_and_preserves_attempts(self) -> None:
        router, edge, cloud = self._router(
            edge_outcomes=[
                RuntimeFailure(
                    category=RuntimeFailureCategory.TIMEOUT,
                    message="edge timed out",
                    exception_type="TimeoutError",
                )
            ],
            cloud_outcomes=[assessment("medium", ["narrow_passage"])],
            signals=self._signals(cloud_allowed=True),
        )

        record = router.analyze(test_case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(edge.calls, 1)
        self.assertEqual(cloud.calls, 1)
        self.assertTrue(record.routing_decision["fallback_used"])
        self.assertEqual(record.routing_decision["fallback_reason"], "timeout")
        self.assertEqual(
            record.routing_decision["attempts"][0]["failure"]["category"],
            "timeout",
        )

    def test_refusal_and_input_failures_do_not_trigger_cloud_fallback(self) -> None:
        for category in (
            RuntimeFailureCategory.MODEL_REFUSAL,
            RuntimeFailureCategory.INPUT_ERROR,
        ):
            with self.subTest(category=category.value):
                router, edge, cloud = self._router(
                    edge_outcomes=[
                        RuntimeFailure(
                            category=category,
                            message="non-retryable request result",
                            exception_type="TestFailure",
                        )
                    ],
                    cloud_outcomes=[],
                    signals=self._signals(cloud_allowed=True),
                )

                record = router.analyze(test_case(), WORKLOAD)

                self.assertFalse(record.succeeded)
                self.assertEqual(edge.calls, 1)
                self.assertEqual(cloud.calls, 0)
                self.assertFalse(record.routing_decision["fallback_used"])

    def test_no_healthy_or_authorized_route_returns_auditable_failure(self) -> None:
        router, edge, cloud = self._router(
            edge_outcomes=[],
            cloud_outcomes=[],
            signals=RoutingSignals(
                cloud_allowed=False,
                network_available=False,
                edge_available=False,
                cloud_available=False,
            ),
        )

        record = router.analyze(test_case(), WORKLOAD)

        self.assertFalse(record.succeeded)
        self.assertEqual(record.failure.category, RuntimeFailureCategory.DEPENDENCY_UNAVAILABLE)
        self.assertEqual(edge.calls, 0)
        self.assertEqual(cloud.calls, 0)
        self.assertIsNone(record.routing_decision["selected_runtime"])
        self.assertEqual(record.routing_decision["attempts"], [])

    def test_health_probe_does_not_touch_cloud_without_upload_permission(self) -> None:
        provider = HttpHealthRoutingSignalsProvider(
            edge_base_url="http://edge.local:8000/v1",
            cloud_base_url="http://cloud.local:8000/v1",
            base_signals=self._signals(cloud_allowed=False),
        )
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)

        with patch("parksight_vlm.inference.routing.urlopen", return_value=response) as get:
            signals = provider.get_signals(case=test_case(), workload=WORKLOAD)

        get.assert_called_once()
        self.assertEqual(get.call_args.args[0].full_url, "http://edge.local:8000/health")
        self.assertTrue(signals.edge_available)
        self.assertFalse(signals.cloud_available)

    def test_health_probe_sends_cloud_api_key_when_authorized(self) -> None:
        provider = HttpHealthRoutingSignalsProvider(
            edge_base_url="http://edge.local:8000",
            cloud_base_url="https://cloud.local:8000/v1",
            base_signals=self._signals(cloud_allowed=True),
            cloud_api_key="secret",
        )
        response = Mock(status=200)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)

        with patch("parksight_vlm.inference.routing.urlopen", return_value=response) as get:
            signals = provider.get_signals(case=test_case(), workload=WORKLOAD)

        self.assertEqual(get.call_count, 2)
        cloud_request = get.call_args_list[1].args[0]
        self.assertEqual(cloud_request.full_url, "https://cloud.local:8000/health")
        self.assertEqual(cloud_request.get_header("Authorization"), "Bearer secret")
        self.assertTrue(signals.cloud_available)


if __name__ == "__main__":
    unittest.main()
