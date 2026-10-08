"""Select reproducibility fields from vLLM's optional server-info endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_CONFIG_FIELDS = {
    "model_config": (
        "model", "revision", "tokenizer_revision", "dtype", "max_model_len",
        "quantization", "enforce_eager", "seed",
    ),
    "cache_config": (
        "enable_prefix_caching", "gpu_memory_utilization", "cache_dtype", "block_size",
    ),
    "scheduler_config": (
        "max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill",
    ),
    "compilation_config": (
        "mode", "cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size",
    ),
    "parallel_config": (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
    ),
}
_MM_FIELDS = (
    "mm_processor_cache_gb", "mm_processor_cache_type", "limit_per_prompt",
    "mm_encoder_tp_mode",
)


def extract_server_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep selected runtime settings while omitting credential fields.

    Requires ``/server_info?config_format=json``. Missing optional fields stay
    absent; their absence must not be interpreted as a cache/Graph default.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("server_info response must be a JSON mapping")
    config = payload.get("vllm_config")
    if not isinstance(config, Mapping):
        raise ValueError("server_info must contain a JSON vllm_config mapping")
    result: dict[str, Any] = {}
    for section, fields in _CONFIG_FIELDS.items():
        values = config.get(section)
        if not isinstance(values, Mapping):
            raise ValueError(f"server_info is missing the {section} mapping")
        result[section] = {key: values[key] for key in fields if key in values}
    multimodal = config["model_config"].get("multimodal_config")
    if isinstance(multimodal, Mapping):
        result["multimodal_config"] = {
            key: multimodal[key] for key in _MM_FIELDS if key in multimodal
        }
        kwargs = multimodal.get("mm_processor_kwargs")
        if isinstance(kwargs, Mapping):
            result["multimodal_config"]["mm_processor_kwargs"] = {
                key: value for key, value in kwargs.items()
                if key in ("min_pixels", "max_pixels") and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
    return result
