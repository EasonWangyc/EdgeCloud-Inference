"""Exclusive HTTP measurement of the inspected single-sequence native runtime."""

from __future__ import annotations

import asyncio
import math
import threading
from functools import wraps
from typing import Any


class _RuntimeActivity:
    def __init__(self, target: Any, owner: "SerialStageObserver") -> None:
        self.target = target
        self.owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def handle_request(self, *args: Any, **kwargs: Any) -> Any:
        with self.owner.mutex:
            if self.owner.active or self.owner.poisoned or not self.owner.http_window_active:
                raise RuntimeError("serial measurement forbids overlapping native execution")
            self.owner.active = True
        try:
            return self.target.handle_request(*args, **kwargs)
        finally:
            with self.owner.mutex:
                self.owner.active = False


class SerialStageObserver:
    """Read Timer only while native execution is idle and HTTP admission is held."""

    def __init__(self, llm: Any) -> None:
        if llm.has_draft_model or getattr(llm, "_batch_scheduler", None) is not None:
            raise ValueError("serial stage metrics require vanilla decoding without a batch scheduler")
        if not callable(getattr(llm._rt, "get_stage_timing_snapshot", None)):
            raise ValueError("native binding does not expose stage timing snapshots")
        self.llm = llm
        self.mutex = threading.Lock()
        self.active = False
        self.streaming = False
        self.poisoned = False
        self.http_window_active = False
        self.original_runtime = llm._runtime
        self.runtime_proxy = _RuntimeActivity(llm._runtime, self)
        llm._runtime = self.runtime_proxy

    def snapshot(self) -> dict[str, Any]:
        with self.mutex:
            if not self.http_window_active or self.active or self.streaming or self.poisoned:
                raise RuntimeError("native workers are not quiescent in an exclusive HTTP window")
            if not self.llm._rt.get_profiling_enabled():
                raise RuntimeError("native profiling is disabled")
            prefill = self.original_runtime.get_prefill_metrics()
            generation = self.original_runtime.get_generation_metrics()
            counts = {
                "prefill_runs": prefill.get_total_runs(),
                "computed_tokens": prefill.computed_tokens,
                "reused_tokens": prefill.reused_tokens,
                "generation_runs": generation.get_total_runs(),
                "generated_tokens": generation.generated_tokens,
            }
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts.values()):
                raise ValueError("invalid native cumulative token counts")
            return {"counts": counts, "stages": self.llm._rt.get_stage_timing_snapshot()}

    def check_termination(self) -> None:
        with self.mutex:
            if self.active or self.streaming:
                # A disconnected stream may outlive upstream's five-second join.
                # Reject further inference rather than measure around that worker.
                self.poisoned = True

    def stream_started(self) -> None:
        with self.mutex:
            if not self.http_window_active or self.poisoned or self.streaming:
                raise RuntimeError("stream generation requires exclusive HTTP admission")
            self.streaming = True

    def stream_stopped(self) -> None:
        with self.mutex:
            self.streaming = False

    def close(self) -> None:
        self.check_termination()
        if self.llm._runtime is self.runtime_proxy:
            self.llm._runtime = self.original_runtime

    def finish(self, before: dict[str, Any], completion_tokens: int) -> dict[str, Any]:
        after = self.snapshot()
        delta = {k: after["counts"][k] - v for k, v in before["counts"].items()}
        if min(delta.values()) < 0 or delta["prefill_runs"] != 1 or delta["generation_runs"] != 1:
            raise ValueError("native counter window does not contain exactly one complete request")
        if delta["generated_tokens"] != completion_tokens:
            raise ValueError("native generation and SSE token counts disagree")
        phases = {}
        for stage, field in (("vision_encoder", "vision_encode_ms"),
                             ("llm_prefill", "prefill_ms"), ("llm_generation", "decode_ms")):
            previous = before["stages"].get(stage, {"run_count": 0, "total_gpu_time_ms": 0.0})
            current = after["stages"].get(stage, previous)
            values = [previous["total_gpu_time_ms"], current["total_gpu_time_ms"]]
            runs = [previous["run_count"], current["run_count"]]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("invalid native cumulative CUDA time")
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in runs):
                raise ValueError("invalid native cumulative stage count")
            count, duration = runs[1] - runs[0], values[1] - values[0]
            if count < 0 or duration < 0:
                raise ValueError("native Timer reset during request")
            if count == 0 and duration != 0:
                raise ValueError("native Timer changed without a stage measurement")
            phases[field] = {"run_count": count, "gpu_time_ms": duration}
        if phases["prefill_ms"]["run_count"] != 1 or phases["vision_encode_ms"]["run_count"] not in (0, 1):
            raise ValueError("unexpected prefill or vision stage boundary")
        decode = phases["decode_ms"]["run_count"]
        if completion_tokens < 1 or decode != completion_tokens - 1:
            raise ValueError("vanilla decode stage count does not match subsequent token count")
        timings = {k: v["gpu_time_ms"] for k, v in phases.items() if v["run_count"] > 0}
        return {
            "usage": {"prompt_tokens": delta["computed_tokens"] + delta["reused_tokens"],
                      "decode_tokens": decode},
            "timings_ms": timings,
            "parksight_measurement": {"status": "valid", "mode": "serial_native_cuda_events",
                                      "before": before, "after": after},
        }


class SerialInferenceApp:
    """Hold admission until the entire ASGI response, including SSE cleanup, ends."""

    def __init__(self, app: Any, observer: SerialStageObserver) -> None:
        self.app = app
        self.observer = observer
        self.lock = asyncio.Lock()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path") in ("/health", "/v1/models"):
            await self.app(scope, receive, send)
            return
        async with self.lock:
            if self.observer.poisoned:
                await send({"type": "http.response.start", "status": 503,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"error":"native worker did not stop; restart measurement service"}'})
                return
            self.observer.http_window_active = True
            try:
                await self.app(scope, receive, send)
            finally:
                self.observer.check_termination()
                self.observer.http_window_active = False


def install_serial_stage_metrics(api_server: Any, llm: Any) -> SerialStageObserver:
    """Patch this process only; all inference routes share one admission lock."""
    from parksight_vlm.inference.edge_server_metrics import install_stream_usage

    original = getattr(api_server, "_create_app", None)
    if not callable(original) or getattr(original, "_parksight_serial_metrics", False):
        raise RuntimeError("unsupported or already configured Edge-LLM app factory")
    observer = SerialStageObserver(llm)
    try:
        install_stream_usage(api_server, stage_observer=observer)
    except BaseException:
        observer.close()
        raise

    @wraps(original)
    def create_app(llm_instance: Any):
        if llm_instance is not llm:
            raise RuntimeError("serial measurement app received a different runtime")
        return SerialInferenceApp(original(llm_instance), observer)

    create_app._parksight_serial_metrics = True
    api_server._create_app = create_app
    return observer
