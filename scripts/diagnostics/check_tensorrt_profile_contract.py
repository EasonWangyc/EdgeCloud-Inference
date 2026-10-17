"""检查 TensorRT 输入长度与 KV cache profile 是否覆盖冻结 workload。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import (
    TensorRTValidationError,
    estimate_kv_cache_bytes,
    validate_profile_contract,
)


def check_prompt_contracts(
    *,
    prompt_contract_paths: list[Path],
    max_input_len: int,
    max_kv_cache_capacity: int,
    max_generate_length: int,
    max_batch_size: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    element_size_bytes: int = 2,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    reasons: list[str] = []
    for path in prompt_contract_paths:
        payload = _read_object(path)
        prompt_tokens = payload.get("prompt_tokens")
        if not isinstance(prompt_tokens, Mapping):
            raise TensorRTValidationError(
                f"prompt contract has no prompt_tokens object: {path}"
            )
        observed = prompt_tokens.get("count")
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise TensorRTValidationError(
                f"prompt_tokens.count must be a non-negative integer: {path}"
            )
        check = validate_profile_contract(
            max_input_len=max_input_len,
            max_kv_cache_capacity=max_kv_cache_capacity,
            observed_input_tokens=observed,
            max_generate_length=max_generate_length,
        )
        observation = {
            "path": str(path),
            "workload_identity": payload.get("workload_identity"),
            **check,
        }
        observations.append(observation)
        reasons.extend(
            f"{path}: {reason}" for reason in check["reasons"]
        )
    return {
        "schema_version": "parksight_tensorrt_profile_contract_v1",
        "profile": {
            "max_input_len": max_input_len,
            "max_kv_cache_capacity": max_kv_cache_capacity,
            "max_generate_length": max_generate_length,
        },
        "observations": observations,
        "kv_cache": _kv_cache_summary(
            max_batch_size=max_batch_size,
            max_kv_cache_capacity=max_kv_cache_capacity,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            element_size_bytes=element_size_bytes,
        ),
        "valid": not reasons,
        "reasons": reasons,
        "evidence_boundary": (
            "This is a prompt/KV capacity contract check; it does not prove that an "
            "engine builds, loads, or is faster"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-contract", action="append", required=True, type=Path)
    parser.add_argument("--max-input-len", required=True, type=int)
    parser.add_argument("--max-kv-cache-capacity", required=True, type=int)
    parser.add_argument("--max-generate-length", required=True, type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--element-size-bytes", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = check_prompt_contracts(
        prompt_contract_paths=args.prompt_contract,
        max_input_len=args.max_input_len,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        max_generate_length=args.max_generate_length,
        max_batch_size=args.max_batch_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        element_size_bytes=args.element_size_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0 if result["valid"] else 2


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TensorRTValidationError(f"cannot read prompt contract: {path}") from error
    if not isinstance(payload, dict):
        raise TensorRTValidationError(f"prompt contract must be an object: {path}")
    return payload


def _kv_cache_summary(
    *,
    max_batch_size: int | None,
    max_kv_cache_capacity: int,
    num_kv_heads: int | None,
    head_dim: int | None,
    element_size_bytes: int,
) -> dict[str, Any] | None:
    dimensions = (max_batch_size, num_kv_heads, head_dim)
    if all(value is None for value in dimensions):
        return None
    if any(value is None for value in dimensions):
        raise TensorRTValidationError(
            "max-batch-size, num-kv-heads, and head-dim must be provided together"
        )
    bytes_count = estimate_kv_cache_bytes(
        max_batch_size=max_batch_size,
        max_kv_cache_capacity=max_kv_cache_capacity,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        element_size_bytes=element_size_bytes,
    )
    return {
        "layout": "[batch, 2, num_kv_heads, max_kv_cache_capacity, head_dim]",
        "max_batch_size": max_batch_size,
        "max_kv_cache_capacity": max_kv_cache_capacity,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "element_size_bytes": element_size_bytes,
        "bytes": bytes_count,
        "mib": bytes_count / (1024 * 1024),
        "evidence_boundary": "estimate excludes allocator alignment and other runtime buffers",
    }


if __name__ == "__main__":
    raise SystemExit(main())
