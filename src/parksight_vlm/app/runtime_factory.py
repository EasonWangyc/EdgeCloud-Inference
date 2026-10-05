"""根据已校验的应用配置组合 Runtime Adapter。"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parksight_vlm.inference import (
    EdgeLlmHttpBackend,
    EdgeLlmRuntime,
    EdgeCloudRouterRuntime,
    HuggingFaceQwen3VlBackend,
    HttpHealthRoutingSignalsProvider,
    RiskRuntime,
    RoutingPolicy,
    RoutingSignals,
    TransformersRuntime,
    VllmHttpBackend,
    VllmRuntime,
)

from .config import AppConfigError, RuntimeConfig


def build_runtime(config: RuntimeConfig, *, data_root: Path) -> RiskRuntime:
    """构建 Runtime，并将模型权重加载延迟到首次执行。"""
    if (
        config.backend_revision == "main"
        or config.backend_revision.startswith("replace-with-")
    ):
        raise AppConfigError(
            "runtime.backend_revision must identify the installed backend version "
            "or immutable commit"
        )
    if (
        config.model_revision == "main"
        or config.model_revision.startswith("replace-with-")
    ):
        raise AppConfigError(
            "runtime.model_revision must be an immutable model commit, not a "
            "branch or template placeholder"
        )
    if config.backend == "transformers":
        _require_allowed_options(
            config.options,
            {
                "device_map",
                "dtype",
                "attn_implementation",
                "adapter_path",
                "profile_stages",
            },
            "transformers",
        )
        backend = HuggingFaceQwen3VlBackend(
            model_id=config.model_id,
            model_revision=config.model_revision,
            device_map=_option_text(config.options, "device_map", "auto"),
            dtype=_option_text(config.options, "dtype", "auto"),
            attn_implementation=_option_text(
                config.options, "attn_implementation", "sdpa"
            ),
            adapter_path=(
                _option_text(config.options, "adapter_path", "")
                if "adapter_path" in config.options
                else None
            ),
            profile_stages=_option_bool(config.options, "profile_stages", False),
        )
        return TransformersRuntime(
            data_root=data_root,
            backend=backend,
            backend_revision=config.backend_revision,
            model_id=config.model_id,
            model_revision=config.model_revision,
            adapter_revision=config.adapter_revision,
            precision=config.precision,
        )
    if config.backend == "tensorrt_edge_llm_http":
        _require_allowed_options(
            config.options,
            {
                "base_url",
                "model_name",
                "timeout_seconds",
                "stream_responses",
                "reuse_http_connection",
            },
            "tensorrt_edge_llm_http",
        )
        timeout_seconds = _positive_number(
            config.options.get("timeout_seconds", 120.0),
            "runtime.options.timeout_seconds",
        )
        stream_responses = config.options.get("stream_responses", True)
        if not isinstance(stream_responses, bool):
            raise AppConfigError("runtime.options.stream_responses must be a boolean")
        reuse_http_connection = config.options.get("reuse_http_connection", False)
        if not isinstance(reuse_http_connection, bool):
            raise AppConfigError(
                "runtime.options.reuse_http_connection must be a boolean"
            )
        backend = EdgeLlmHttpBackend(
            base_url=_option_text(config.options, "base_url", "http://127.0.0.1:8000"),
            model_name=_option_text(config.options, "model_name", "local"),
            timeout_seconds=float(timeout_seconds),
            stream_responses=stream_responses,
            reuse_http_connection=reuse_http_connection,
        )
        return EdgeLlmRuntime(
            data_root=data_root,
            backend=backend,
            backend_revision=config.backend_revision,
            model_id=config.model_id,
            model_revision=config.model_revision,
            adapter_revision=config.adapter_revision,
            precision=config.precision,
        )
    if config.backend == "vllm_http":
        _require_allowed_options(
            config.options,
            {
                "base_url",
                "model_name",
                "timeout_seconds",
                "stream_responses",
                "reuse_http_connection",
                "api_key_env",
                "json_mode",
                "image_preprocessing",
            },
            "vllm_http",
        )
        timeout_seconds = _positive_number(
            config.options.get("timeout_seconds", 180.0),
            "runtime.options.timeout_seconds",
        )
        stream_responses = _option_bool(
            config.options, "stream_responses", True
        )
        reuse_http_connection = _option_bool(
            config.options, "reuse_http_connection", True
        )
        json_mode = _option_bool(config.options, "json_mode", True)
        image_preprocessing = _option_text(config.options, "image_preprocessing", "source")
        if image_preprocessing not in ("source", "workload_resize"):
            raise AppConfigError("runtime.options.image_preprocessing must be source or workload_resize")
        api_key_env = config.options.get("api_key_env")
        api_key = None
        if api_key_env is not None:
            api_key_env = _option_text(config.options, "api_key_env", "")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise AppConfigError(
                    f"environment variable {api_key_env!r} must contain the vLLM API key"
                )
        backend = VllmHttpBackend(
            base_url=_option_text(
                config.options, "base_url", "http://127.0.0.1:8000"
            ),
            model_name=_option_text(
                config.options, "model_name", config.model_id
            ),
            timeout_seconds=float(timeout_seconds),
            stream_responses=stream_responses,
            reuse_http_connection=reuse_http_connection,
            api_key=api_key,
            json_mode=json_mode,
            image_preprocessing=image_preprocessing,
        )
        return VllmRuntime(
            data_root=data_root,
            backend=backend,
            backend_revision=config.backend_revision,
            model_id=config.model_id,
            model_revision=config.model_revision,
            adapter_revision=config.adapter_revision,
            precision=config.precision,
        )
    if config.backend == "edge_vllm_router":
        _require_allowed_options(
            config.options,
            {
                "edge_runtime",
                "cloud_runtime",
                "policy",
                "signals",
                "health_probe_timeout_seconds",
            },
            "edge_vllm_router",
        )
        edge_payload = config.options.get("edge_runtime")
        cloud_payload = config.options.get("cloud_runtime")
        policy_payload = config.options.get("policy", {})
        signals_payload = config.options.get("signals")
        if not isinstance(edge_payload, Mapping):
            raise AppConfigError("runtime.options.edge_runtime must be a mapping")
        if not isinstance(cloud_payload, Mapping):
            raise AppConfigError("runtime.options.cloud_runtime must be a mapping")
        if not isinstance(policy_payload, Mapping):
            raise AppConfigError("runtime.options.policy must be a mapping")
        if not isinstance(signals_payload, Mapping):
            raise AppConfigError("runtime.options.signals must be a mapping")
        edge_config = RuntimeConfig.from_mapping(edge_payload)
        cloud_config = RuntimeConfig.from_mapping(cloud_payload)
        if edge_config.backend != "tensorrt_edge_llm_http":
            raise AppConfigError(
                "runtime.options.edge_runtime.backend must be tensorrt_edge_llm_http"
            )
        if cloud_config.backend != "vllm_http":
            raise AppConfigError(
                "runtime.options.cloud_runtime.backend must be vllm_http"
            )
        try:
            policy = RoutingPolicy.from_mapping(dict(policy_payload))
            signals = RoutingSignals.from_mapping(dict(signals_payload))
        except (TypeError, ValueError) as error:
            raise AppConfigError(f"invalid edge-vLLM router options: {error}") from error
        edge_runtime = build_runtime(edge_config, data_root=data_root)
        cloud_runtime = build_runtime(cloud_config, data_root=data_root)
        cloud_api_key_env = cloud_config.options.get("api_key_env")
        cloud_api_key = (
            os.environ.get(_option_text(cloud_config.options, "api_key_env", ""))
            if cloud_api_key_env is not None
            else None
        )
        return EdgeCloudRouterRuntime(
            data_root=data_root,
            edge_runtime=edge_runtime,
            cloud_runtime=cloud_runtime,
            signals_provider=HttpHealthRoutingSignalsProvider(
                edge_base_url=_option_text(
                    edge_config.options, "base_url", "http://127.0.0.1:8000"
                ),
                cloud_base_url=_option_text(
                    cloud_config.options, "base_url", "http://127.0.0.1:8000"
                ),
                base_signals=signals,
                cloud_api_key=cloud_api_key,
                timeout_seconds=_positive_number(
                    config.options.get("health_probe_timeout_seconds", 1.0),
                    "runtime.options.health_probe_timeout_seconds",
                ),
            ),
            policy=policy,
            backend_revision=config.backend_revision,
            model_id=config.model_id,
            model_revision=config.model_revision,
            adapter_revision=config.adapter_revision,
        )
    raise AppConfigError(f"unsupported runtime backend: {config.backend!r}")


def _require_allowed_options(
    options: Mapping[str, Any], allowed_options: set[str], backend: str
) -> None:
    unexpected_options = set(options) - allowed_options
    if unexpected_options:
        raise AppConfigError(
            f"unsupported {backend} options: {sorted(unexpected_options)}"
        )
def _option_text(options: Mapping[str, Any], key: str, default: str) -> str:
    value = options.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise AppConfigError(f"runtime.options.{key} must be a non-blank string")
    return value.strip()


def _option_bool(options: Mapping[str, Any], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if not isinstance(value, bool):
        raise AppConfigError(f"runtime.options.{key} must be a boolean")
    return value


def _positive_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise AppConfigError(f"{field_name} must be a finite positive number")
    return float(value)
