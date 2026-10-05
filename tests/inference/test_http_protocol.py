"""真实协议边界：拆包、提前断流、明确拒答与持久连接恢复。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, patch

from parksight_vlm.casebook import DatasetSplit, ParkingCase
from parksight_vlm.inference import (
    EdgeLlmHttpBackend,
    EdgeLlmRuntime,
    ResourceSnapshot,
    RuntimeGeneration,
    StageTimings,
    StreamTimings,
)
from parksight_vlm.inference.edge_llm import OpenAICompatibleHttpBackend
from parksight_vlm.workload import FrozenWorkload


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = FrozenWorkload.load(ROOT / "configs/workloads/parking_risk_v1.json")
IMAGE = ROOT / "tests/fixtures/inference/scene.jpg"
ASSESSMENT = {
    "schema_version": "parking_risk_v1",
    "risk_level": "medium",
    "events": ["narrow_passage"],
    "evidence": ["两侧车辆之间的通道狭窄"],
    "driver_advice": ["slow_down"],
}


def event(payload: dict[str, object]) -> bytes:
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\r\n\r\n").encode()


class Response:
    status = 200
    will_close = False

    def __init__(self, chunks: list[bytes | Exception], *, stream: bool = True) -> None:
        self.chunks = chunks
        self.headers = {"Content-Type": "text/event-stream" if stream else "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def read(self) -> bytes:
        return b"".join(self.chunks)


class HttpProtocolTests(unittest.TestCase):
    def test_stream_timing_uses_content_events_and_server_token_usage(self) -> None:
        response = Response([
            event({"choices": [{"delta": {"role": "assistant"}}]}),
            event({"choices": [{"delta": {"content": "hello"}}]}),
            event({"choices": [{"delta": {"content": " world"}}]}),
            event({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 5}}),
            b"data: [DONE]\n\n",
        ])
        with patch("parksight_vlm.inference.edge_llm.urlopen", return_value=response), patch(
            "parksight_vlm.inference.edge_llm.time.perf_counter",
            side_effect=[0.0, 0.01, 1.0, 2.0, 2.05, 2.1],
        ):
            generation = EdgeLlmHttpBackend().generate(image_path=IMAGE, workload=WORKLOAD)
        self.assertEqual(generation.input_tokens, 20)
        self.assertEqual(generation.output_tokens, 5)
        self.assertEqual(len(generation.stream_timings.content_arrival_ms), 2)
        self.assertAlmostEqual(generation.stream_timings.chunk_intervals_ms[0], 50.0)
        self.assertAlmostEqual(generation.stage_timings.time_to_first_token_ms, 1000.0)
        self.assertAlmostEqual(generation.stage_timings.client_total_ttft_ms, 1010.0)
        self.assertAlmostEqual(generation.stage_timings.client_tpot_ms, 12.5)

    def test_single_content_chunk_does_not_fabricate_tpot(self) -> None:
        record = self.record(Response([
            event({"choices": [{"delta": {"content": json.dumps(ASSESSMENT)}}]}),
            event({"choices": [], "usage": {"completion_tokens": 31}}),
            b"data: [DONE]\n\n",
        ]))
        self.assertIsNone(record.stage_timings.client_tpot_ms)
        self.assertEqual(record.stream_timings.chunk_intervals_ms, ())

    def test_stream_arrivals_reject_non_monotonic_or_invalid_values(self) -> None:
        for arrivals in ((2.0, 1.0), (None,), (True,), (float("nan"),)):
            with self.subTest(arrivals=arrivals), self.assertRaises(ValueError):
                StreamTimings(arrivals)

    def record(self, response: Response):
        runtime = EdgeLlmRuntime(
            data_root=IMAGE.parent,
            backend=EdgeLlmHttpBackend(),
            backend_revision="test",
            model_id="test",
            model_revision="test",
            adapter_revision="none",
            precision="fp16",
        )
        case = ParkingCase(
            case_id="protocol-test",
            image_ref=PurePosixPath(IMAGE.name),
            source_group_id="protocol-test",
            split=DatasetSplit.TEST,
            reference_assessment=None,
        )
        with patch("parksight_vlm.inference.edge_llm.urlopen", return_value=response):
            return runtime.analyze(case, WORKLOAD)

    def test_chinese_utf8_split_at_every_byte_round_trips(self) -> None:
        wire = b"\xef\xbb\xbf: keepalive\r\n\r\n" + event(
            {"choices": [{"delta": {"content": json.dumps(ASSESSMENT, ensure_ascii=False)}}]}
        ) + event({"choices": [], "usage": {"completion_tokens": 31}}) + b"data: [DONE]\r\n\r\n"
        record = self.record(Response([wire[i:i + 1] for i in range(len(wire))]))
        self.assertTrue(record.succeeded)
        self.assertEqual(record.assessment.evidence, tuple(ASSESSMENT["evidence"]))
        self.assertEqual(record.output_tokens, 31)
        self.assertIsNotNone(record.stage_timings.time_to_first_token_ms)

    def test_valid_json_followed_by_premature_eof_is_not_success(self) -> None:
        record = self.record(Response([event(
            {"choices": [{"delta": {"content": json.dumps(ASSESSMENT)}}]}
        )]))
        self.assertEqual(record.failure.category.value, "runtime_error")
        self.assertIn("before [DONE]", record.failure.message)
        self.assertIsNone(record.assessment)

    def test_server_error_event_is_not_a_business_json_failure(self) -> None:
        record = self.record(Response([
            event({"error": {"message": "engine failed"}}),
            b"data: [DONE]\n\n",
        ]))
        self.assertEqual(record.failure.category.value, "runtime_error")
        self.assertIn("engine failed", record.failure.message)

    def test_malformed_protocol_json_is_not_a_business_json_failure(self) -> None:
        for stream in (True, False):
            with self.subTest(stream=stream):
                chunks = [b"data: not-json\n\n", b"data: [DONE]\n\n"] if stream else [b"not-json"]
                record = self.record(Response(chunks, stream=stream))
                self.assertEqual(record.failure.category.value, "runtime_error")
                self.assertIn("JSON envelope", record.failure.message)

    def test_explicit_refusal_is_preserved_for_stream_and_json_response(self) -> None:
        payloads = [
            {"choices": [{"delta": {"refusal": "request refused"}}]},
            {"choices": [{"message": {"content": None, "refusal": "request refused"}}]},
            {"choices": [{"finish_reason": "content_filter", "delta": {}}]},
        ]
        for payload in payloads:
            for stream in (True, False):
                with self.subTest(payload=payload, stream=stream):
                    chunks = [event(payload), b"data: [DONE]\n\n"] if stream else [json.dumps(payload).encode()]
                    record = self.record(Response(chunks, stream=stream))
                    self.assertEqual(record.failure.category.value, "model_refusal")

    def test_timeout_during_body_read_discards_connection_before_next_request(self) -> None:
        first_connection, second_connection = Mock(), Mock()
        first_connection.getresponse.return_value = Response([TimeoutError("read timed out")])
        second_connection.getresponse.return_value = Response([
            event({"choices": [{"delta": {"content": "{}"}}]}),
            b"data: [DONE]\n\n",
        ])
        backend = EdgeLlmHttpBackend(reuse_http_connection=True)
        with patch(
            "parksight_vlm.inference.edge_llm.http.client.HTTPConnection",
            side_effect=[first_connection, second_connection],
        ) as factory:
            with self.assertRaises(TimeoutError):
                backend.generate(image_path=IMAGE, workload=WORKLOAD)
            first_connection.close.assert_called_once()
            self.assertIsNone(backend._http_connection)
            generation = backend.generate(image_path=IMAGE, workload=WORKLOAD)
            self.assertEqual(generation.raw_output, "{}")
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(first_connection.request.call_count, 1)
        backend.close()

    def test_usage_does_not_accept_boolean_negative_or_fractional_tokens(self) -> None:
        for value in (True, -1, 1.5):
            with self.subTest(value=value):
                record = self.record(Response([json.dumps({
                    "choices": [{"message": {"content": json.dumps(ASSESSMENT)}}],
                    "usage": {"completion_tokens": value},
                }).encode()], stream=False))
                self.assertEqual(record.failure.category.value, "runtime_error")

    def test_multiline_data_and_terminal_event_without_final_newline(self) -> None:
        chunks = [b'data: {"choices":\n', b'data: []}\n\n', b'data: [DONE]']
        self.assertEqual(list(OpenAICompatibleHttpBackend.iter_sse_payloads(chunks)), [{"choices": []}])

    def test_invalid_event_shape_and_data_after_done_are_rejected(self) -> None:
        for chunks in ([b"data: []\n\n", b"data: [DONE]\n\n"], [b"data: [DONE]\n\n", event({})]):
            with self.subTest(chunks=chunks), self.assertRaises(RuntimeError):
                list(OpenAICompatibleHttpBackend.iter_sse_payloads(chunks))

    def test_timeout_and_measurements_require_finite_numeric_values(self) -> None:
        for value in (True, float("nan"), float("inf"), -1, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EdgeLlmHttpBackend(timeout_seconds=value)
                with self.assertRaises(ValueError):
                    StageTimings(decode_ms=value)
                with self.assertRaises(ValueError):
                    ResourceSnapshot(peak_memory_mb=value)
                with self.assertRaises(RuntimeError):
                    EdgeLlmHttpBackend.parse_server_timings({"timings_ms": {"decode_ms": value}})
        with self.assertRaises(ValueError):
            RuntimeGeneration(raw_output="{}", output_tokens=True)


if __name__ == "__main__":
    unittest.main()
