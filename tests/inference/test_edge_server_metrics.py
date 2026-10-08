"""SSE usage from native token IDs, including empty text and stream cleanup."""

import json
import types
import unittest

from parksight_vlm.inference.edge_server_metrics import install_stream_usage


def event(payload):
    return "data: " + json.dumps(payload) + "\n\n"


def native_server(llm_instance, messages, params, response_id, **kwargs):
    for delta in llm_instance.generate_stream(messages, params, **kwargs):
        yield event({"choices": [{"delta": {"content": delta.text}}]})
    if kwargs.get("existing_usage"):
        yield event({"choices": [], "usage": {"completion_tokens": 99}})
    yield event({"choices": [{"delta": {}, "finish_reason": kwargs.get("finish", "stop")}]})
    yield "data: [DONE]\n\n"


class FakeLlm:
    def __init__(self, ids):
        self.ids = ids
        self.closed = False

    def generate_stream(self, *args, **kwargs):
        try:
            for ids in self.ids:
                yield types.SimpleNamespace(text="" if ids == [] else "text", token_ids=ids)
        finally:
            self.closed = True


def usage_events(events):
    return [json.loads(e[5:]) for e in events
            if e.startswith("data:") and e.strip() != "data: [DONE]"
            and "usage" in json.loads(e[5:])]


class EdgeServerMetricsTests(unittest.TestCase):
    def setUp(self):
        self.api = types.SimpleNamespace(_generate_stream_sse=native_server)
        install_stream_usage(self.api)

    def test_counts_ids_and_preserves_every_existing_sse_event(self):
        llm = FakeLlm([[1, 2], [], [3]])
        actual = list(self.api._generate_stream_sse(llm, [], None, "request-1"))
        expected = list(native_server(FakeLlm([[1, 2], [], [3]]), [], None, "request-1"))
        self.assertEqual(actual[:-2] + actual[-1:], expected)
        self.assertEqual(usage_events(actual)[0]["usage"], {"completion_tokens": 3})
        self.assertEqual(usage_events(actual)[0]["choices"], [])
        self.assertTrue(llm.closed)

    def test_interleaved_streams_have_independent_counters(self):
        first = self.api._generate_stream_sse(FakeLlm([[1], [2]]), [], None, "first")
        second = self.api._generate_stream_sse(FakeLlm([[3, 4, 5]]), [], None, "second")
        first_event, second_event = next(first), next(second)
        self.assertEqual(usage_events([first_event, *first])[0]["usage"]["completion_tokens"], 2)
        self.assertEqual(usage_events([second_event, *second])[0]["usage"]["completion_tokens"], 3)

    def test_unknown_ids_and_error_finish_do_not_claim_completed_usage(self):
        for ids, kwargs in (([None], {}), ([[True]], {}), ([[1]], {"finish": "error"}), ([], {})):
            with self.subTest(ids=ids, kwargs=kwargs):
                events = list(self.api._generate_stream_sse(FakeLlm(ids), [], None, "x", **kwargs))
                self.assertEqual(usage_events(events), [])

    def test_existing_usage_is_preserved_and_zero_tokens_are_explicit(self):
        events = list(self.api._generate_stream_sse(FakeLlm([[1]]), [], None, "x", existing_usage=True))
        self.assertEqual(len(usage_events(events)), 1)
        self.assertEqual(usage_events(events)[0]["usage"]["completion_tokens"], 99)
        events = list(self.api._generate_stream_sse(FakeLlm([[]]), [], None, "x"))
        self.assertEqual(usage_events(events)[0]["usage"]["completion_tokens"], 0)

    def test_disconnect_closes_original_stream(self):
        llm = FakeLlm([[1], [2]])
        stream = self.api._generate_stream_sse(llm, [], None, "x")
        next(stream)
        stream.close()
        self.assertTrue(llm.closed)

    def test_install_is_idempotent_and_rejects_unsupported_boundary(self):
        installed = self.api._generate_stream_sse
        install_stream_usage(self.api)
        self.assertIs(self.api._generate_stream_sse, installed)
        for module in (types.SimpleNamespace(), types.SimpleNamespace(_generate_stream_sse=lambda: None)):
            with self.assertRaises(RuntimeError):
                install_stream_usage(module)
