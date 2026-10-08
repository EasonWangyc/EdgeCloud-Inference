"""Opt-in SSE usage from Edge-LLM's actual per-delta token IDs."""

from __future__ import annotations

import inspect
import json
from functools import wraps
from typing import Any


class _TokenCountingProxy:
    def __init__(self, target: Any) -> None:
        self._target = target
        self.token_count = 0
        self.observed = False
        self.valid = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def generate_stream(self, *args: Any, **kwargs: Any):
        source = iter(self._target.generate_stream(*args, **kwargs))
        try:
            for delta in source:
                ids = getattr(delta, "token_ids", None)
                if not isinstance(ids, (list, tuple)) or any(
                    isinstance(token, bool) or not isinstance(token, int) or token < 0
                    for token in ids
                ):
                    self.valid = False
                else:
                    self.observed = True
                    self.token_count += len(ids)
                yield delta
        finally:
            close = getattr(source, "close", None)
            if callable(close):
                close()


def install_stream_usage(api_server: Any) -> None:
    """Wrap one server module in memory; leave source files and native runtime intact.

    Only the known Edge-LLM generator boundary is supported. Each HTTP stream
    owns its counter. Existing usage is preserved, and missing IDs remain unknown.
    """
    original = getattr(api_server, "_generate_stream_sse", None)
    if not callable(original):
        raise RuntimeError("Edge-LLM server has no supported SSE generator")
    if getattr(original, "_parksight_stream_usage", False):
        return
    signature = inspect.signature(original)
    if not {"llm_instance", "response_id"}.issubset(signature.parameters):
        raise RuntimeError("unsupported Edge-LLM SSE generator signature")

    @wraps(original)
    def measured(*args: Any, **kwargs: Any):
        bound = signature.bind(*args, **kwargs)
        proxy = _TokenCountingProxy(bound.arguments["llm_instance"])
        bound.arguments["llm_instance"] = proxy
        stream = iter(original(*bound.args, **bound.kwargs))
        has_usage = False
        server_error = False
        try:
            for event in stream:
                if isinstance(event, str) and event.strip() == "data: [DONE]":
                    if proxy.observed and proxy.valid and not has_usage and not server_error:
                        payload = {
                            "id": bound.arguments["response_id"],
                            "object": "chat.completion.chunk",
                            "choices": [],
                            # Prompt/phase counts are not exposed by this boundary.
                            "usage": {"completion_tokens": proxy.token_count},
                        }
                        yield "data: " + json.dumps(payload) + "\n\n"
                    yield event
                    continue
                if isinstance(event, str) and event.startswith("data:"):
                    try:
                        payload = json.loads(event[5:].strip())
                    except (ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict):
                        has_usage |= isinstance(payload.get("usage"), dict)
                        choices = payload.get("choices", [])
                        if isinstance(choices, list):
                            server_error |= any(
                                isinstance(choice, dict) and choice.get("finish_reason") == "error"
                                for choice in choices
                            )
                yield event
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    measured._parksight_stream_usage = True
    api_server._generate_stream_sse = measured
