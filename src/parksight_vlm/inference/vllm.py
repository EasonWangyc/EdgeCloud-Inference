"""vLLM OpenAI-compatible HTTP Runtime Adapter."""

from __future__ import annotations

import base64
import ipaddress
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit

from parksight_vlm.workload import FrozenWorkload

from .edge_llm import OpenAICompatibleHttpBackend
from .runtime import RiskRuntime, RuntimeGeneration, RuntimeIdentity


class VllmHttpBackend(OpenAICompatibleHttpBackend):
    """通过 vLLM OpenAI-compatible Chat Completions API 执行图像推理。

    HTTP 传输、SSE 解码和响应计时复用 OpenAI-compatible 基类；本类负责
    把本地图片编码成标准 data URI，并生成 vLLM 请求体。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 120.0,
        stream_responses: bool = True,
        reuse_http_connection: bool = False,
        api_key: str | None = None,
        json_mode: bool = True,
    ) -> None:
        parsed_url = urlsplit(base_url)
        if api_key and parsed_url.scheme == "http":
            hostname = parsed_url.hostname or ""
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = hostname.lower() == "localhost"
            if not is_loopback:
                raise ValueError("vLLM API keys require HTTPS except on loopback")
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
            stream_responses=stream_responses,
            reuse_http_connection=reuse_http_connection,
            api_key=api_key,
        )
        self._json_mode = json_mode

    def build_request_payload(
        self,
        *,
        image_path: Path,
        workload: FrozenWorkload,
        stream: bool | None = None,
    ) -> dict[str, object]:
        """构造标准 OpenAI 多模态消息；图片字节仅在请求时编码上传。"""
        mime_type = mimetypes.guess_type(image_path.name)[0]
        if mime_type is None or not mime_type.startswith("image/"):
            raise ValueError(f"unsupported image MIME type: {image_path.name}")
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload: dict[str, object] = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "system",
                    "content": workload.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": workload.render_user_prompt(),
                        },
                    ],
                },
            ],
            "max_tokens": workload.generation.max_new_tokens,
        }
        if not workload.generation.do_sample:
            payload["temperature"] = 0.0
        if self._json_mode:
            # 语法约束不代替 RiskRuntime 对 ParkingAssessment 的 schema 校验。
            payload["response_format"] = {"type": "json_object"}
        if stream is not None:
            payload["stream"] = stream
            if stream:
                payload["stream_options"] = {"include_usage": True}
        return payload


class VllmRuntime(RiskRuntime):
    """记录云侧 vLLM 执行事实并统一校验 ParkingAssessment。"""

    def __init__(
        self,
        *,
        data_root: Path,
        backend: VllmHttpBackend,
        backend_revision: str,
        model_id: str,
        model_revision: str,
        adapter_revision: str,
        precision: str,
    ) -> None:
        super().__init__(
            data_root=data_root,
            identity=RuntimeIdentity(
                backend="vllm",
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
