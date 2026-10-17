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


_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16", "torch.bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16", "torch.float16": "fp16",
    "fp32": "fp32", "float32": "fp32", "torch.float32": "fp32",
}


def compare_server_identity(
    config: Mapping[str, Any], *, model_revision: str, precision: str,
) -> dict[str, bool | None]:
    """Compare explicit server identity with the study claim; absent values stay unknown."""
    model = config.get("model_config")
    if not isinstance(model, Mapping):
        raise ValueError("server config is missing model_config")
    revision = model.get("revision")
    server_dtype = model.get("dtype")
    expected_dtype = _DTYPE_ALIASES.get(precision.lower())
    actual_dtype = _DTYPE_ALIASES.get(server_dtype.lower()) if isinstance(server_dtype, str) else None
    return {
        "model_revision_matches_client": (
            revision == model_revision if isinstance(revision, str) and revision else None
        ),
        "precision_matches_client": (
            actual_dtype == expected_dtype if actual_dtype is not None and expected_dtype is not None else None
        ),
    }
