"""Validate a TensorRT Edge-LLM reduced-vocabulary mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def validate_mapping(
    *,
    map_path: Path,
    metadata_path: Path,
    original_vocab_size: int,
    required_token_ids: tuple[int, ...] = (),
    packed_int4_lm_head: bool = False,
) -> dict[str, Any]:
    """Validate ``vocab_map.safetensors`` without importing torch/safetensors."""
    reasons: list[str] = []
    map_bytes = map_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("reduced vocabulary metadata must be an object")
    declared_vocab_size = metadata.get("vocab_size")
    declared_reduced_size = metadata.get("reduced_vocab_size")
    if declared_vocab_size != original_vocab_size:
        reasons.append(
            f"metadata vocab_size {declared_vocab_size!r} != "
            f"original_vocab_size {original_vocab_size}"
        )
    if (
        isinstance(declared_reduced_size, bool)
        or not isinstance(declared_reduced_size, int)
        or declared_reduced_size <= 0
    ):
        reasons.append("metadata reduced_vocab_size must be a positive integer")
    if (
        packed_int4_lm_head
        and isinstance(declared_reduced_size, int)
        and not isinstance(declared_reduced_size, bool)
        and declared_reduced_size % 128 != 0
    ):
        reasons.append(
            "packed INT4 AWQ lm_head requires reduced_vocab_size to be a multiple "
            f"of 128: observed {declared_reduced_size}"
        )

    tensor = _read_safetensors_1d_int32(map_bytes, "vocab_map")
    token_ids = tensor["values"]
    if declared_reduced_size != len(token_ids):
        reasons.append(
            f"metadata reduced_vocab_size {declared_reduced_size!r} != "
            f"map length {len(token_ids)}"
        )
    if any(token_id < 0 or token_id >= original_vocab_size for token_id in token_ids):
        reasons.append("map contains a token id outside the original vocabulary")
    if any(left >= right for left, right in zip(token_ids, token_ids[1:])):
        reasons.append("map token ids must be strictly increasing")

    missing_required = sorted(set(required_token_ids) - set(token_ids))
    invalid_required = sorted(
        token_id
        for token_id in set(required_token_ids)
        if token_id < 0 or token_id >= original_vocab_size
    )
    if invalid_required:
        reasons.append(f"required token ids outside vocabulary: {invalid_required}")
    if missing_required:
        reasons.append(f"required token ids missing from map: {missing_required}")

    return {
        "schema_version": "parksight_reduced_vocab_validation_v1",
        "valid": not reasons,
        "map": {
            "path": str(map_path),
            "sha256": hashlib.sha256(map_bytes).hexdigest(),
            "dtype": tensor["dtype"],
            "count": len(token_ids),
            "first_token_id": token_ids[0] if token_ids else None,
            "last_token_id": token_ids[-1] if token_ids else None,
        },
        "metadata": {
            "path": str(metadata_path),
            "vocab_size": declared_vocab_size,
            "reduced_vocab_size": declared_reduced_size,
        },
        "required_token_ids": list(required_token_ids),
        "packed_int4_lm_head": packed_int4_lm_head,
        "packed_int4_group_size": 128 if packed_int4_lm_head else None,
        "missing_required_token_ids": missing_required,
        "reasons": reasons,
    }


def read_mapping_token_ids(map_path: Path) -> list[int]:
    """Read the sorted original token IDs from a validated map file."""
    tensor = _read_safetensors_1d_int32(map_path.read_bytes(), "vocab_map")
    return [int(token_id) for token_id in tensor["values"]]


def _read_safetensors_1d_int32(data: bytes, name: str) -> dict[str, Any]:
    if len(data) < 8:
        raise ValueError("safetensors file is missing its header length")
    header_length = struct.unpack_from("<Q", data, 0)[0]
    header_start = 8
    data_start = header_start + header_length
    if data_start > len(data):
        raise ValueError("safetensors header extends past the file")
    try:
        header = json.loads(data[header_start:data_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid safetensors header") from error
    if not isinstance(header, dict) or name not in header:
        raise ValueError(f"safetensors tensor {name!r} is missing")
    descriptor = header[name]
    if not isinstance(descriptor, dict):
        raise ValueError(f"safetensors tensor {name!r} descriptor is invalid")
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    offsets = descriptor.get("data_offsets")
    if dtype != "I32" or shape is None or offsets is None:
        raise ValueError(f"{name} must be a one-dimensional I32 tensor")
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int):
        raise ValueError(f"{name} must have a one-dimensional shape")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"{name} data_offsets are invalid")
    start, end = offsets
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < start
        or data_start + end > len(data)
        or end - start != shape[0] * 4
    ):
        raise ValueError(f"{name} data_offsets do not match its shape")
    values = list(
        struct.unpack(
            f"<{shape[0]}i",
            data[data_start + start : data_start + end],
        )
    )
    return {"dtype": dtype, "values": values}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, dest="map_path")
    parser.add_argument("--metadata", required=True, type=Path, dest="metadata_path")
    parser.add_argument("--original-vocab-size", required=True, type=int)
    parser.add_argument("--required-token-id", action="append", type=int, default=[])
    parser.add_argument(
        "--packed-int4-lm-head",
        action="store_true",
        help="require reduced_vocab_size to be aligned to packed INT4 group size 128",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.original_vocab_size <= 0:
        parser.error("--original-vocab-size must be positive")
    result = validate_mapping(
        map_path=args.map_path,
        metadata_path=args.metadata_path,
        original_vocab_size=args.original_vocab_size,
        required_token_ids=tuple(args.required_token_id),
        packed_int4_lm_head=args.packed_int4_lm_head,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(args.output)
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
