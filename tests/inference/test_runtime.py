"""Runtime 执行记录与失败分类测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from parksight_vlm.assessment import ParkingAssessment
from parksight_vlm.casebook import DatasetSplit, ParkingCase
from parksight_vlm.inference import (
    EdgeLlmRuntime,
    EdgeLlmHttpBackend,
    ResourceSnapshot,
    RuntimeDependencyError,
    RuntimeFailureCategory,
    RuntimeGeneration,
    RuntimeRefusalError,
    StageTimings,
    TransformersRuntime,
)
from parksight_vlm.inference.transformers import (
    _FirstTokenTimingProbe,
    _ForwardPhaseProfiler,
    _find_profile_module,
    _require_cuda_architecture,
    build_qwen3_vl_chat_messages,
)
from parksight_vlm.workload import FrozenWorkload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "inference"
WORKLOAD = FrozenWorkload.load(
    PROJECT_ROOT / "configs" / "workloads" / "parking_risk_v1.json"
)


class StaticBackend:
    def __init__(self, generation: RuntimeGeneration | Exception) -> None:
        self.generation = generation
        self.calls: list[Path] = []

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration: # 与TransformersRuntime定义一致
        self.calls.append(image_path)
        if isinstance(self.generation, Exception):
            raise self.generation
        return self.generation


class RuntimeTests(unittest.TestCase):
    def test_forward_phase_profiler_separates_prefill_and_decode(self) -> None:
        class Handle:
            def remove(self) -> None:
                return None

        class Module:
            def __init__(self) -> None:
                self.before: list[object] = []
                self.after: list[object] = []

            def register_forward_pre_hook(self, callback: object) -> Handle:
                self.before.append(callback)
                return Handle()

            def register_forward_hook(self, callback: object) -> Handle:
                self.after.append(callback)
                return Handle()

            def run(self) -> None:
                for callback in self.before:
                    callback(self, ())  # type: ignore[operator]
                for callback in self.after:
                    callback(self, (), None)  # type: ignore[operator]

        class Model:
            def __init__(self) -> None:
                self.visual = Module()
                self.language = Module()

            def named_modules(self) -> list[tuple[str, Module]]:
                return [
                    ("model.visual", self.visual),
                    ("model.language_model", self.language),
                ]

        class Cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class Torch:
            cuda = Cuda()

        model = Model()
        with _ForwardPhaseProfiler(model, Torch()) as profiler:
            model.visual.run()
            model.language.run()
            model.language.run()

        self.assertIsNotNone(profiler.vision_encode_ms)
        self.assertIsNotNone(profiler.prefill_ms)
        self.assertIsNotNone(profiler.decode_ms)
        self.assertIs(
            _find_profile_module(dict(model.named_modules()), "visual"),
            model.visual,
        )

    def test_first_token_probe_records_first_logits_timestamp(self) -> None:
        class Cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class Torch:
            cuda = Cuda()

        probe = _FirstTokenTimingProbe(Torch(), generation_start=0.0)
        scores = object()

        self.assertIs(probe(object(), scores), scores)
        self.assertIsNotNone(probe.elapsed_ms)
        first_elapsed_ms = probe.elapsed_ms
        probe(object(), object())
        self.assertEqual(probe.elapsed_ms, first_elapsed_ms)

    def test_qwen3_vl_messages_use_typed_content_items(self) -> None:
        image = object()

        messages = build_qwen3_vl_chat_messages(image=image, workload=WORKLOAD)

        self.assertEqual(
            messages[0],
            {
                "role": "system",
                "content": [{"type": "text", "text": WORKLOAD.system_prompt}],
            },
        )
        self.assertIs(messages[1]["content"][0]["image"], image)
        self.assertEqual(messages[1]["content"][1]["type"], "text")
        self.assertEqual(
            messages[1]["content"][1]["text"],
            WORKLOAD.render_user_prompt(),
        )

    def test_transformers_backend_rejects_incompatible_cuda_wheel(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def get_device_capability() -> tuple[int, int]:
                return (8, 7)

            @staticmethod
            def get_arch_list() -> list[str]:
                return ["sm_80", "sm_90"]

        class FakeTorch:
            __version__ = "2.9.1+cu126"
            cuda = FakeCuda()

            @staticmethod
            def ones(*args: object, **kwargs: object) -> object:
                raise RuntimeError("no compatible kernel image")

        with self.assertRaisesRegex(
            RuntimeDependencyError,
            "does not include CUDA kernels for sm_87",
        ):
            _require_cuda_architecture(FakeTorch())

    def test_transformers_runtime_records_validated_success(self) -> None:
        backend = StaticBackend(
            RuntimeGeneration(
                raw_output=json.dumps(self._assessment_payload()),
                stage_timings=StageTimings(preprocess_ms=2.0, decode_ms=5.0),
                resource_snapshot=ResourceSnapshot(peak_memory_mb=1024.0),
                output_tokens=42,
            )
        )
        runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=backend,
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )

        record = runtime.analyze(self._case(), WORKLOAD)

        self.assertTrue(record.succeeded)
        self.assertEqual(record.assessment.risk_level.value, "medium")
        self.assertEqual(record.output_tokens, 42)
        self.assertEqual(record.resource_snapshot.peak_memory_mb, 1024.0)
        self.assertIsNotNone(record.stage_timings.end_to_end_ms)
        self.assertEqual(record.workload_identity, WORKLOAD.identity)
        self.assertEqual(backend.calls, [FIXTURE_ROOT / "scene.jpg"])
        self.assertEqual(record.to_mapping()["failure"], None)

    def test_edge_llm_runtime_preserves_invalid_json_failure(self) -> None:
        backend = StaticBackend(RuntimeGeneration(raw_output="not-json", output_tokens=2))
        runtime = EdgeLlmRuntime(
            data_root=FIXTURE_ROOT,
            backend=backend,
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
            adapter_revision="edge-test",
            precision="fp16",
        )

        record = runtime.analyze(self._case(), WORKLOAD)

        self.assertFalse(record.succeeded)
        self.assertEqual(record.failure.category, RuntimeFailureCategory.JSON_PARSE_ERROR)
        self.assertEqual(record.raw_output, "not-json")
        self.assertEqual(record.output_tokens, 2)

    def test_runtime_records_timeout_and_missing_input(self) -> None:
        timeout_runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=StaticBackend(TimeoutError("generation timed out")),
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )
        timeout_record = timeout_runtime.analyze(self._case(), WORKLOAD)
        self.assertEqual(timeout_record.failure.category, RuntimeFailureCategory.TIMEOUT)

        missing_case = self._case(image_ref="missing.jpg")
        missing_record = timeout_runtime.analyze(missing_case, WORKLOAD)
        self.assertEqual(missing_record.failure.category, RuntimeFailureCategory.INPUT_ERROR)

    def test_runtime_records_explicit_model_refusal(self) -> None:
        runtime = TransformersRuntime(
            data_root=FIXTURE_ROOT,
            backend=StaticBackend(RuntimeRefusalError("request refused")),
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test-revision",
        )

        record = runtime.analyze(self._case(), WORKLOAD)

        self.assertEqual(
            record.failure.category, RuntimeFailureCategory.MODEL_REFUSAL
        )

    def test_edge_llm_http_backend_uses_multimodal_chat_request(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps(self_payload)}}
                        ],
                        "usage": {"completion_tokens": 19},
                        "timings_ms": {
                            "prefill_ms": 120,
                            "ttft_ms": 180,
                            "decode_ms": 400,
                            "server_e2e_ms": 650,
                        },
                    }
                ).encode("utf-8")

        self_payload = self._assessment_payload()
        backend = EdgeLlmHttpBackend(base_url="http://127.0.0.1:8000")
        with patch(
            "parksight_vlm.inference.edge_llm.urlopen",
            return_value=FakeResponse(),
        ) as mocked_urlopen:
            generation = backend.generate(
                image_path=FIXTURE_ROOT / "scene.jpg",
                workload=WORKLOAD,
            )

        request = mocked_urlopen.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        system_message = request_payload["messages"][0]
        image_item = request_payload["messages"][1]["content"][0]
        text_item = request_payload["messages"][1]["content"][1]
        self.assertEqual(
            system_message,
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": WORKLOAD.system_prompt}
                ],
            },
        )
        self.assertEqual(image_item["type"], "image")
        self.assertEqual(text_item["type"], "text")
        self.assertEqual(text_item["text"], WORKLOAD.render_user_prompt())
        self.assertEqual(generation.output_tokens, 19)
        self.assertIsNotNone(generation.stage_timings.preprocess_ms)
        self.assertIsNotNone(generation.stage_timings.http_round_trip_ms)
        self.assertEqual(generation.stage_timings.prefill_ms, 120.0)
        self.assertEqual(generation.stage_timings.time_to_first_token_ms, 180.0)
        self.assertEqual(generation.stage_timings.decode_ms, 400.0)
        self.assertEqual(generation.stage_timings.backend_end_to_end_ms, 650.0)
        self.assertEqual(json.loads(generation.raw_output), self_payload)

    def test_edge_llm_http_backend_rejects_invalid_server_timing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            EdgeLlmHttpBackend.parse_server_timings(
                {"timings_ms": {"decode_ms": -1}}
            )

    def test_edge_llm_http_backend_validates_base_url(self) -> None:
        with self.assertRaisesRegex(ValueError, r"absolute http\(s\) URL"):
            EdgeLlmHttpBackend(base_url="127.0.0.1:8000")

    def test_edge_llm_http_backend_persistent_connection_uses_http_client(self) -> None:
        from unittest.mock import patch

        class FakeResponse:
            status = 200
            headers = {"content-type": "application/json"}

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return (
                    b'{"choices":[{"message":{"content":"{}"}}],'
                    b'"usage":{"completion_tokens":1}}'
                )

        class FakeConnection:
            instances: list["FakeConnection"] = []

            def __init__(self, host: str, *, timeout: float) -> None:
                self.host = host
                self.timeout = timeout
                self.requests: list[dict[str, object]] = []
                self.closed = False
                self.instances.append(self)

            def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
                self.requests.append(
                    {"method": method, "path": path, "body": body, "headers": headers}
                )

            def getresponse(self) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                self.closed = True

        workload = WORKLOAD
        image_path = Path("image.jpg")
        backend = EdgeLlmHttpBackend(
            base_url="http://127.0.0.1:8000",
            stream_responses=False,
            reuse_http_connection=True,
        )
        with patch(
            "parksight_vlm.inference.edge_llm.http.client.HTTPConnection",
            FakeConnection,
        ):
            backend.generate(image_path=image_path, workload=workload)
            backend.generate(image_path=image_path, workload=workload)

        self.assertEqual(len(FakeConnection.instances), 1)
        self.assertEqual(len(FakeConnection.instances[0].requests), 2)
        self.assertEqual(
            FakeConnection.instances[0].requests[0]["path"],
            "/v1/chat/completions",
        )
        self.assertEqual(
            FakeConnection.instances[0].requests[0]["headers"]["Connection"],
            "keep-alive",
        )

    def test_edge_llm_http_backend_measures_ttft_from_first_stream_delta(self) -> None:
        self_payload = self._assessment_payload()
        raw_payload = json.dumps(self_payload)

        class FakeHeaders:
            @staticmethod
            def get_content_type() -> str:
                return "text/event-stream"

        class FakeResponse:
            headers = FakeHeaders()

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def __iter__(self):
                events = [
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": raw_payload[:12]}}]},
                    {"choices": [{"delta": {"content": raw_payload[12:]}}]},
                    {
                        "choices": [],
                        "usage": {"completion_tokens": 19},
                        "timings_ms": {
                            "prefill_ms": 120,
                            "decode_ms": 400,
                            "server_e2e_ms": 650,
                        },
                    },
                ]
                for event in events:
                    encoded = ("data: " + json.dumps(event) + "\n\n").encode("utf-8")
                    split_at = max(1, len(encoded) // 2)
                    yield encoded[:split_at]
                    yield encoded[split_at:]
                yield b"data: [DONE]\n\n"

        backend = EdgeLlmHttpBackend(base_url="http://127.0.0.1:8000")
        with (
            patch(
                "parksight_vlm.inference.edge_llm.urlopen",
                return_value=FakeResponse(),
            ) as mocked_urlopen,
            patch(
                "parksight_vlm.inference.edge_llm.time.perf_counter",
                side_effect=[0.0, 0.0, 10.0, 11.0, 12.0, 20.0],
            ),
        ):
            generation = backend.generate(
                image_path=FIXTURE_ROOT / "scene.jpg",
                workload=WORKLOAD,
            )

        request_payload = json.loads(mocked_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertTrue(request_payload["stream"])
        self.assertEqual(generation.output_tokens, 19)
        self.assertEqual(generation.stage_timings.time_to_first_token_ms, 1000.0)
        self.assertEqual(generation.stage_timings.http_round_trip_ms, 10000.0)
        self.assertEqual(generation.stage_timings.prefill_ms, 120.0)
        self.assertEqual(generation.stage_timings.decode_ms, 400.0)
        self.assertEqual(generation.stage_timings.backend_end_to_end_ms, 650.0)
        self.assertEqual(json.loads(generation.raw_output), self_payload)

    @staticmethod
    def _assessment_payload() -> dict[str, object]:
        return {
            "schema_version": "parking_risk_v1",
            "risk_level": "medium",
            "events": ["narrow_passage"],
            "evidence": ["Vehicles leave a narrow maneuvering corridor."],
            "driver_advice": ["slow_down"],
        }

    @classmethod
    def _case(cls, image_ref: str = "scene.jpg") -> ParkingCase:
        return ParkingCase(
            case_id="case-001",
            image_ref=PurePosixPath(image_ref),
            source_group_id="sequence-a",
            split=DatasetSplit.TEST,
            reference_assessment=ParkingAssessment.from_mapping(cls._assessment_payload()),
        )


if __name__ == "__main__":
    unittest.main()
