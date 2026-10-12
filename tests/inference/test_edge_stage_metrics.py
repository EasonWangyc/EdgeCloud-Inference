"""Exclusive response lifetime, native quiescence and phase/count consistency."""

import asyncio
import copy
import json
import threading
import time
import unittest
from types import SimpleNamespace

from parksight_vlm.inference.edge_llm import OpenAICompatibleHttpBackend
from parksight_vlm.inference.edge_server_metrics import install_stream_usage
from parksight_vlm.inference.edge_stage_metrics import SerialInferenceApp, SerialStageObserver
from tests.inference.test_edge_server_metrics import native_server, usage_events


class FakeNative:
    def __init__(self):
        self.runs = 0
        self.generated = 0
        self.stages = {}
        self.snapshot_calls = 0
        self.profile = True
    def get_profiling_enabled(self):
        return self.profile
    def set_profiling_enabled(self, enabled):
        self.profile = enabled
    def get_stage_timing_snapshot(self):
        self.snapshot_calls += 1
        return copy.deepcopy(self.stages)
    def get_prefill_metrics(self):
        return SimpleNamespace(get_total_runs=lambda: self.runs, computed_tokens=self.runs * 5, reused_tokens=0)
    def get_generation_metrics(self):
        return SimpleNamespace(get_total_runs=lambda: self.runs, generated_tokens=self.generated)
    def handle_request(self, token_count):
        self.runs += 1
        self.generated += token_count
        for stage, count, duration in [('vision_encoder', 1, 2.0), ('llm_prefill', 1, 3.0),
                                       ('llm_generation', token_count - 1, 7.0)]:
            previous = self.stages.get(stage, {'run_count': 0, 'total_gpu_time_ms': 0.0})
            self.stages[stage] = {'run_count': previous['run_count'] + count,
                                  'total_gpu_time_ms': previous['total_gpu_time_ms'] + duration}


def measured_llm():
    runtime = FakeNative()
    llm = SimpleNamespace(has_draft_model=False, _batch_scheduler=None, _runtime=runtime, _rt=runtime)
    def stream(*args, **kwargs):
        llm._runtime.handle_request(3)
        yield SimpleNamespace(text='answer', token_ids=[1, 2, 3])
    llm.generate_stream = stream
    return llm, runtime


class StageMetricsTests(unittest.TestCase):
    def test_real_client_parser_receives_paired_stage_and_usage_extensions(self):
        llm, runtime = measured_llm()
        observer = SerialStageObserver(llm)
        observer.http_window_active = True
        api = SimpleNamespace(_generate_stream_sse=native_server)
        install_stream_usage(api, stage_observer=observer)
        for _ in range(2):
            events = list(api._generate_stream_sse(llm, [], None, 'request'))
            payload, _, timings, _ = OpenAICompatibleHttpBackend.read_stream_response(
                (event.encode() for event in events), request_start=time.perf_counter())
            self.assertEqual(payload['usage'], {'completion_tokens': 3, 'prompt_tokens': 5, 'decode_tokens': 2})
            self.assertEqual(timings, {'vision_encode_ms': 2.0, 'prefill_ms': 3.0, 'decode_ms': 7.0})
            self.assertEqual(payload['parksight_measurement']['status'], 'valid')
        observer.close()
        self.assertIs(llm._runtime, runtime)

    def test_counter_mismatch_preserves_completion_usage_without_inventing_phases(self):
        llm, runtime = measured_llm()
        original = llm.generate_stream
        def mismatched(*args, **kwargs):
            yield from original(*args, **kwargs)
            runtime.generated += 1
        llm.generate_stream = mismatched
        observer = SerialStageObserver(llm)
        observer.http_window_active = True
        api = SimpleNamespace(_generate_stream_sse=native_server)
        install_stream_usage(api, stage_observer=observer)
        payload = usage_events(list(api._generate_stream_sse(llm, [], None, 'request')))[0]
        self.assertEqual(payload['usage'], {'completion_tokens': 3})
        self.assertNotIn('timings_ms', payload)
        self.assertEqual(payload['parksight_measurement']['status'], 'unavailable')

    def test_snapshot_refuses_busy_worker_and_failed_shutdown_poisons_admission(self):
        llm, runtime = measured_llm()
        entered, release = threading.Event(), threading.Event()
        def blocked(count):
            entered.set()
            release.wait(timeout=5)
        runtime.handle_request = blocked
        observer = SerialStageObserver(llm)
        observer.http_window_active = True
        worker = threading.Thread(target=llm._runtime.handle_request, args=(3,))
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            with self.assertRaisesRegex(RuntimeError, 'quiescent'):
                observer.snapshot()
            self.assertEqual(runtime.snapshot_calls, 0)
            observer.check_termination()
            self.assertTrue(observer.poisoned)
        finally:
            release.set()
            worker.join(timeout=1)
        with self.assertRaises(RuntimeError):
            observer.snapshot()

    def test_rejects_uncontrolled_runtime_or_timer_reset(self):
        for draft, scheduler in ((True, None), (False, object())):
            llm, _ = measured_llm()
            llm.has_draft_model, llm._batch_scheduler = draft, scheduler
            with self.assertRaises(ValueError):
                SerialStageObserver(llm)
        llm, runtime = measured_llm()
        observer = SerialStageObserver(llm)
        observer.http_window_active = True
        llm._runtime.handle_request(3)
        before = observer.snapshot()
        runtime.runs = 0
        with self.assertRaises(ValueError):
            observer.finish(before, 3)
        runtime.profile = False
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            observer.snapshot()

    def test_stream_preparation_also_blocks_snapshot_and_late_native_execution(self):
        llm, runtime = measured_llm()
        observer = SerialStageObserver(llm)
        observer.http_window_active = True
        observer.stream_started()
        with self.assertRaisesRegex(RuntimeError, 'quiescent'):
            observer.snapshot()
        self.assertEqual(runtime.snapshot_calls, 0)
        observer.check_termination()
        observer.stream_stopped()
        with self.assertRaises(RuntimeError):
            llm._runtime.handle_request(3)
        self.assertEqual(runtime.runs, 0)


class SerialAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_response_with_live_native_worker_closes_further_admission(self):
        llm, runtime = measured_llm()
        entered, release = threading.Event(), threading.Event()
        def blocked(count):
            entered.set()
            release.wait(timeout=5)
        runtime.handle_request = blocked
        observer = SerialStageObserver(llm)
        worker = None
        ready = asyncio.Event()
        async def app(scope, receive, send):
            nonlocal worker
            worker = threading.Thread(target=llm._runtime.handle_request, args=(3,))
            worker.start()
            self.assertTrue(entered.wait(timeout=1))
            ready.set()
            await asyncio.Event().wait()
        wrapper = SerialInferenceApp(app, observer)
        request = asyncio.create_task(wrapper({'type':'http','path':'/v1/chat/completions'}, None, None))
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            self.assertTrue(observer.poisoned)
            self.assertFalse(observer.http_window_active)
            responses = []
            async def send(message):
                responses.append(message)
            await wrapper({'type':'http','path':'/v1/chat/completions'}, None, send)
            self.assertEqual(responses[0]['status'], 503)
        finally:
            release.set()
            if worker is not None:
                worker.join(timeout=1)

    async def test_serializes_entire_response_and_releases_admission_after_failure(self):
        llm, _ = measured_llm()
        observer = SerialStageObserver(llm)
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def app(scope, receive, send):
            calls.append(scope['id'])
            self.assertTrue(observer.http_window_active)
            if scope['id'] == 1:
                started.set()
                await release.wait()
                raise RuntimeError('response failed')
        wrapper = SerialInferenceApp(app, observer)
        scope = lambda n: {'type':'http', 'path':'/v1/chat/completions', 'id':n}
        first = asyncio.create_task(wrapper(scope(1), None, None))
        await started.wait()
        second = asyncio.create_task(wrapper(scope(2), None, None))
        await asyncio.sleep(0)
        self.assertEqual(calls, [1])
        release.set()
        with self.assertRaisesRegex(RuntimeError, 'response failed'):
            await first
        await second
        self.assertEqual(calls, [1, 2])
        self.assertFalse(observer.http_window_active)

    async def test_poisoned_service_rejects_inference_but_preserves_health(self):
        llm, _ = measured_llm()
        observer = SerialStageObserver(llm)
        observer.poisoned = True
        calls, responses = [], []
        async def app(scope, receive, send):
            calls.append(scope['path'])
        async def send(message):
            responses.append(message)
        wrapper = SerialInferenceApp(app, observer)
        await wrapper({'type':'http','path':'/v1/chat/completions'}, None, send)
        self.assertEqual(responses[0]['status'], 503)
        await wrapper({'type':'http','path':'/health'}, None, send)
        self.assertEqual(calls, ['/health'])
