"""TensorRT Edge-LLM Runtime Adapter。"""

from __future__ import annotations

import codecs
import http.client
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator, Protocol
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from parksight_vlm.workload import FrozenWorkload

from .runtime import (
    RiskRuntime,
    RuntimeGeneration,
    RuntimeIdentity,
    RuntimeRefusalError,
    StageTimings,
    StreamTimings,
)


class EdgeLlmBackend(Protocol):
    """面向已安装 Jetson Runtime 实现的可执行 Edge-LLM 接口。"""

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        """调用 engine 并返回原始输出和实测事实。"""


class OpenAICompatibleHttpBackend:
    """OpenAI-compatible Chat Completions 的共享 HTTP/SSE 传输层。"""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        model_name: str = "local",
        timeout_seconds: float = 120.0,
        stream_responses: bool = True,
        reuse_http_connection: bool = False,
        api_key: str | None = None,
    ) -> None:
        parsed_url = urlsplit(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("base_url must be an absolute http(s) URL")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        base_path = parsed_url.path.rstrip("/")
        if base_path.endswith("/v1/chat/completions"):
            endpoint_path = base_path
        elif base_path.endswith("/v1"):
            endpoint_path = base_path + "/chat/completions"
        else:
            endpoint_path = base_path + "/v1/chat/completions"
        self._endpoint = urlunsplit(
            (parsed_url.scheme, parsed_url.netloc, endpoint_path, parsed_url.query, "")
        )
        self._http_endpoint_path = urlunsplit(
            ("", "", endpoint_path, parsed_url.query, "")
        )
        self._http_scheme = parsed_url.scheme
        self._http_host = parsed_url.netloc
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._stream_responses = stream_responses
        self._reuse_http_connection = reuse_http_connection
        self._api_key = api_key
        self._http_connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None

    def close(self) -> None:
        """关闭可选的持久 HTTP 连接。"""
        if self._http_connection is not None:
            self._http_connection.close()
            self._http_connection = None

    def __del__(self) -> None:
        # Best effort only: interpreter shutdown may already have torn down
        # http.client globals. Explicit close() remains the deterministic path.
        try:
            self.close()
        except Exception:
            pass

    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        """执行请求；任何传输或解析失败都丢弃持久连接，不隐式重试 POST。"""
        try:
            return self._generate_response(image_path=image_path, workload=workload)
        except Exception:
            self.close()
            raise

    def _generate_response(
        self, *, image_path: Path, workload: FrozenWorkload
    ) -> RuntimeGeneration:
        request_build_start = time.perf_counter()
        payload = self.build_request_payload(
            image_path=image_path,
            workload=workload,
            stream=self._stream_responses,
        )
        headers = {
            "Accept": "text/event-stream" if self._stream_responses else "application/json",
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        request_build_ms = (time.perf_counter() - request_build_start) * 1000.0
        http_start = time.perf_counter()
        response_context = (
            self._open_persistent_response(request_body=request.data, headers=request.headers)
            if self._reuse_http_connection
            else urlopen(request, timeout=self._timeout_seconds)
        )
        response_will_close = False
        with response_context as response:
            status = getattr(response, "status", 200)
            if isinstance(status, int) and not 200 <= status < 300:
                self.close()
                raise RuntimeError(
                    f"OpenAI-compatible inference HTTP request failed with status {status}"
                )
            response_will_close = bool(getattr(response, "will_close", False))
            if self._stream_responses and self.is_event_stream_response(response):
                response_payload, measured_ttft_ms, stream_timings, arrivals = (
                    self.read_stream_response(response, request_start=http_start)
                )
            else:
                response_body = response.read()
                try:
                    response_payload = json.loads(response_body.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise RuntimeError("invalid OpenAI-compatible JSON envelope") from error
                measured_ttft_ms = None
                stream_timings = {}
                arrivals = StreamTimings()
        self.validate_response_payload(response_payload)
        if self._reuse_http_connection and response_will_close:
            self.close()
        http_round_trip_ms = (time.perf_counter() - http_start) * 1000.0
        try:
            raw_output = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenAI-compatible response has no assistant content") from error
        if not isinstance(raw_output, str):
            raise RuntimeError("OpenAI-compatible assistant content must be a string")
        usage = response_payload.get("usage", {})
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if output_tokens is not None and (
            isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens < 0
        ):
            raise RuntimeError("completion_tokens must be a non-negative integer")
        input_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if input_tokens is not None and (
            isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0
        ):
            raise RuntimeError("prompt_tokens must be a non-negative integer")
        client_tpot_ms = None
        if output_tokens is not None and output_tokens > 1 and len(arrivals.content_arrival_ms) > 1:
            client_tpot_ms = (
                arrivals.content_arrival_ms[-1] - arrivals.content_arrival_ms[0]
            ) / (output_tokens - 1)
        server_timings = dict(stream_timings)
        server_timings.update(self.parse_server_timings(response_payload))
        return RuntimeGeneration(
            raw_output=raw_output,
            stage_timings=StageTimings(
                # 客户端构造请求耗时；启用 workload_resize 时包含解码与缩放，
                # 不包含服务端处理。
                preprocess_ms=request_build_ms,
                vision_encode_ms=server_timings.get("vision_encode_ms"),
                model_generate_ms=server_timings.get("model_generate_ms"),
                prefill_ms=server_timings.get("prefill_ms"),
                decode_ms=server_timings.get("decode_ms"),
                # 非流式 OpenAI-compatible 响应只能可靠测到完整 HTTP 往返时间。
                http_round_trip_ms=http_round_trip_ms,
                backend_end_to_end_ms=server_timings.get("backend_end_to_end_ms"),
                end_to_end_ms=server_timings.get("end_to_end_ms"),
                # 流式路径测量客户端从发起请求到收到首个非空 token 的时间；
                # 非流式路径只有服务端显式返回 TTFT 时才填充该字段。
                time_to_first_token_ms=(
                    measured_ttft_ms
                    if measured_ttft_ms is not None
                    else server_timings.get("time_to_first_token_ms")
                ),
                client_total_ttft_ms=(request_build_ms + measured_ttft_ms if measured_ttft_ms is not None else None),
                client_tpot_ms=client_tpot_ms,
            ),
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            stream_timings=arrivals,
        )

    def _open_persistent_response(
        self,
        *,
        request_body: bytes | None,
        headers: dict[str, str],
    ) -> http.client.HTTPResponse:
        """通过同一 HTTP/1.1 连接发送一次请求，减少逐请求 TCP 建连。"""
        if request_body is None:
            raise RuntimeError("HTTP request body is required")
        if self._http_connection is None:
            connection_type = (
                http.client.HTTPSConnection
                if self._http_scheme == "https"
                else http.client.HTTPConnection
            )
            self._http_connection = connection_type(
                self._http_host,
                timeout=self._timeout_seconds,
            )
        try:
            self._http_connection.request(
                "POST",
                self._http_endpoint_path,
                body=request_body,
                headers={**headers, "Connection": "keep-alive"},
            )
            response = self._http_connection.getresponse()
        except Exception:
            # A broken keep-alive connection must not be reused for a later
            # sample. Do not retry a POST implicitly: retrying could duplicate
            # a request whose server-side execution already started.
            self.close()
            raise
        return response

    @staticmethod
    def is_event_stream_response(response: object) -> bool:
        """判断 HTTP 响应是否为 OpenAI-compatible Server-Sent Events。"""
        headers = getattr(response, "headers", None)
        if headers is None:
            return False
        get_content_type = getattr(headers, "get_content_type", None)
        if callable(get_content_type):
            return str(get_content_type()).lower() == "text/event-stream"
        if isinstance(headers, dict):
            content_type = headers.get("Content-Type", headers.get("content-type", ""))
            return "text/event-stream" in str(content_type).lower()
        return False

    @classmethod
    def read_stream_response(
        cls,
        response: Any,
        *,
        request_start: float,
    ) -> tuple[dict[str, object], float | None, dict[str, float], StreamTimings]:
        """读取 SSE 响应并在首个非空文本 delta 到达时记录 TTFT。"""
        content_parts: list[str] = []
        usage: dict[str, object] | None = None
        timings: dict[str, float] = {}
        first_token_ms: float | None = None
        last_payload: dict[str, object] = {}
        arrivals: list[float] = []

        for payload in cls.iter_sse_payloads(response):
            cls.validate_response_payload(payload)
            last_payload = payload
            timings.update(cls.parse_server_timings(payload))
            raw_usage = payload.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
            contents = list(cls.extract_stream_content(payload))
            if any(contents):
                arrival = (time.perf_counter() - request_start) * 1000.0
                arrivals.append(arrival)
                if first_token_ms is None:
                    first_token_ms = arrival
            for content in contents:
                content_parts.append(content)

        result: dict[str, object] = {
            "choices": [{"message": {"content": "".join(content_parts)}}]
        }
        if usage is not None:
            result["usage"] = usage
        # 保留最后一个事件中除 choices/usage 以外的服务端扩展字段，
        # 以便继续解析 timings_ms/performance。
        for key, value in last_payload.items():
            if key not in {"choices", "usage"}:
                result[key] = value
        return result, first_token_ms, timings, StreamTimings(tuple(arrivals))

    @classmethod
    def iter_sse_payloads(cls, response: Any) -> Iterator[dict[str, object]]:
        """增量解码 SSE；只有收到 [DONE] 才将流视为完整返回。"""
        completed = False
        for data in cls._iter_sse_data(response):
            if data == "[DONE]":
                completed = True
                continue
            if completed:
                raise RuntimeError("SSE response contains data after [DONE]")
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as error:
                raise RuntimeError("invalid SSE JSON envelope") from error
            if not isinstance(parsed, dict):
                raise RuntimeError("SSE response payload must be an object")
            yield parsed
        if not completed:
            raise RuntimeError("SSE response ended before [DONE]")

    @staticmethod
    def _iter_sse_data(response: Any) -> Iterator[str]:
        """保留跨 chunk 的 UTF-8 状态、行边界和多行 data 事件。"""
        decoder = codecs.getincrementaldecoder("utf-8-sig")()
        data_lines: list[str] = []
        line_buffer = ""
        for raw_chunk in response:
            chunk_bytes = (
                raw_chunk
                if isinstance(raw_chunk, bytes)
                else str(raw_chunk).encode("utf-8")
            )
            line_buffer += decoder.decode(chunk_bytes)
            # HTTPResponse 通常逐行迭代，但代理也可能将一行拆成多个 chunk。
            lines = line_buffer.split("\n")
            line_buffer = lines.pop()
            for line in lines:
                line = line.rstrip("\r")
                if not line:
                    if data_lines:
                        payload = "\n".join(data_lines)
                        data_lines = []
                        yield payload
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].removeprefix(" "))
        line_buffer += decoder.decode(b"", final=True)
        if line_buffer:
            if line_buffer.startswith("data:"):
                data_lines.append(line_buffer[5:].removeprefix(" ").rstrip("\r"))
        if data_lines:
            yield "\n".join(data_lines)

    @staticmethod
    def validate_response_payload(payload: object) -> None:
        """保留服务端错误和明确拒答的语义，不将其归入业务 JSON 错误。"""
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI-compatible response must be an object")
        if payload.get("error") is not None:
            error = payload["error"]
            message = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(f"OpenAI-compatible server error: {message}")
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            raise RuntimeError("OpenAI-compatible choices must be an array")
        for choice in choices:
            if not isinstance(choice, dict):
                raise RuntimeError("OpenAI-compatible choice must be an object")
            if choice.get("finish_reason") == "content_filter":
                raise RuntimeRefusalError("model response was filtered")
            for name in ("message", "delta"):
                part = choice.get(name)
                if isinstance(part, dict) and part.get("refusal"):
                    raise RuntimeRefusalError(str(part["refusal"]))

    @staticmethod
    def extract_stream_content(payload: dict[str, object]) -> Iterator[str]:
        """提取标准 delta.content，也兼容旧式 text 字段。"""
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            candidate: object = delta.get("content") if isinstance(delta, dict) else None
            if candidate is None:
                candidate = choice.get("text")
            if isinstance(candidate, str):
                yield candidate

    @staticmethod
    def parse_server_timings(payload: dict[str, object]) -> dict[str, float]:
        """解析服务端扩展返回的实际阶段时延。

        标准 OpenAI-compatible 响应不包含这些字段，因此缺失时返回空字典；
        已知字段若存在但类型错误则拒绝该响应，避免将无效数字写入证据。
        """
        raw_timings = payload.get("timings_ms")
        if raw_timings is None:
            raw_timings = payload.get("performance", {})
        if raw_timings is None:
            return {}
        if not isinstance(raw_timings, dict):
            raise RuntimeError("server timings must be an object")
        aliases = {
            "ttft_ms": "time_to_first_token_ms",
            "e2e_ms": "backend_end_to_end_ms",
            "end_to_end_ms": "backend_end_to_end_ms",
            "server_e2e_ms": "backend_end_to_end_ms",
        }
        supported = {
            "vision_encode_ms",
            "model_generate_ms",
            "prefill_ms",
            "decode_ms",
            "backend_end_to_end_ms",
            "time_to_first_token_ms",
        }
        parsed: dict[str, float] = {}
        for name, value in raw_timings.items():
            normalized_name = aliases.get(name, name)
            if normalized_name not in supported:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise RuntimeError(f"server timing must be finite and non-negative: {name}")
            parsed[normalized_name] = float(value)
        return parsed

    def build_request_payload(
        self,
        *,
        image_path: Path,
        workload: FrozenWorkload,
        stream: bool | None = None,
    ) -> dict[str, object]:
        """构造后端请求；具体消息格式由协议适配器实现。"""
        raise NotImplementedError


class EdgeLlmHttpBackend(OpenAICompatibleHttpBackend):
    """TensorRT Edge-LLM server 的 OpenAI-compatible HTTP Adapter。"""

    def build_request_payload(
        self,
        *,
        image_path: Path,
        workload: FrozenWorkload,
        stream: bool | None = None,
    ) -> dict[str, object]:
        """构造 Edge-LLM server 当前接受的可审计请求体。"""
        payload: dict[str, object] = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": workload.system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": workload.render_user_prompt()},
                    ],
                },
            ],
            "max_tokens": workload.generation.max_new_tokens,
        }
        if not workload.generation.do_sample:
            payload["temperature"] = 0.0
        if stream is not None:
            payload["stream"] = stream
        return payload


class EdgeLlmRuntime(RiskRuntime):
    """记录 TensorRT Edge-LLM 后端实际执行事实的 Adapter。"""

    def __init__(
        self,
        *,
        data_root: Path,
        backend: EdgeLlmBackend,
        backend_revision: str,
        model_id: str,
        model_revision: str,
        adapter_revision: str,
        precision: str,
    ) -> None:
        super().__init__(
            data_root=data_root,
            identity=RuntimeIdentity(
                backend="tensorrt_edge_llm",
                backend_revision=backend_revision,
                model_id=model_id,
                model_revision=model_revision,
                adapter_revision=adapter_revision,
                precision=precision,
            ),
        )
        self._backend = backend

    def _generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        return self._backend.generate(image_path=image_path, workload=workload)
