"""比较两个 TensorRT Engine Inspector JSON 的 layer/tactic 证据。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def compare_inspectors(left_path: Path | str, right_path: Path | str) -> dict[str, Any]:
    """按稳定 layer name 优先、index 兜底比较两个 Inspector 报告。"""
    left = _load_report(left_path)
    right = _load_report(right_path)
    left_layers = _layer_signatures(left.get("layers", []))
    right_layers = _layer_signatures(right.get("layers", []))
    aligned_layers = _align_layers(left_layers, right_layers)
    changed_layers: list[dict[str, Any]] = []
    for alignment in aligned_layers:
        left_index = alignment["left_index"]
        right_index = alignment["right_index"]
        left_signature = (
            left_layers.get(left_index) if left_index is not None else None
        )
        right_signature = (
            right_layers.get(right_index) if right_index is not None else None
        )
        if left_signature != right_signature:
            changed_layers.append(
                {
                    "key": alignment["key"],
                    "index": (
                        left_index
                        if left_index is not None and left_index == right_index
                        else None
                    ),
                    "left_index": left_index,
                    "right_index": right_index,
                    "left": left_signature,
                    "right": right_signature,
                }
            )

    left_engine = left.get("engine", {})
    right_engine = right.get("engine", {})
    alignment_summary = _alignment_summary(aligned_layers, left_layers, right_layers)
    return {
        "schema_version": "parksight_tensorrt_engine_inspector_comparison_v1",
        "inputs": {
            "left": str(Path(left_path)),
            "right": str(Path(right_path)),
        },
        "engine": {
            "left_sha256": left_engine.get("sha256"),
            "right_sha256": right_engine.get("sha256"),
            "left_size_bytes": left_engine.get("size_bytes"),
            "right_size_bytes": right_engine.get("size_bytes"),
            "left_device_memory_size": left_engine.get("device_memory_size"),
            "right_device_memory_size": right_engine.get("device_memory_size"),
            "left_device_memory_size_v2": left_engine.get("device_memory_size_v2"),
            "right_device_memory_size_v2": right_engine.get("device_memory_size_v2"),
        },
        "layers": {
            "left_count": len(left_layers),
            "right_count": len(right_layers),
            "changed_count": len(changed_layers),
            "changed": changed_layers,
            "alignment": {
                "strategy": "unique_layer_name_then_index",
                "matched_count": sum(
                    1
                    for item in aligned_layers
                    if item["left_index"] is not None
                    and item["right_index"] is not None
                ),
                "added_count": sum(
                    1 for item in aligned_layers if item["left_index"] is None
                ),
                "removed_count": sum(
                    1 for item in aligned_layers if item["right_index"] is None
                ),
            },
            "diff_summary": alignment_summary,
            "left_operator_counts": _operator_counts(left_layers),
            "right_operator_counts": _operator_counts(right_layers),
            "operator_count_delta": _operator_count_delta(left_layers, right_layers),
        },
        "tactic_visibility": {
            "left": _visibility(left_layers, "tactic"),
            "right": _visibility(right_layers, "tactic"),
            "workspace": {
                "left": _visibility(left_layers, "workspace"),
                "right": _visibility(right_layers, "workspace"),
            },
            "timing": {
                "left": _visibility(left_layers, "timing"),
                "right": _visibility(right_layers, "timing"),
            },
        },
        "evidence_boundary": (
            "A changed layer signature identifies serialized inspector metadata that differs; "
            "name alignment is only a structural comparison and may report added/removed "
            "layers when TensorRT fuses or splits operators. "
            "It does not prove that the layer is the runtime bottleneck. Use matching workload "
            "timings and Nsight Systems/Compute to attribute latency or kernel occupancy."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_inspectors(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


def _load_report(path: Path | str) -> Mapping[str, Any]:
    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid TensorRT inspector report: {report_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"TensorRT inspector report must be an object: {report_path}")
    if payload.get("schema_version") != "parksight_tensorrt_engine_inspector_v1":
        raise ValueError(f"unsupported TensorRT inspector schema: {report_path}")
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"TensorRT inspector report layers must be an array: {report_path}")
    return payload


def _layer_signatures(layers: list[Any]) -> dict[int, dict[str, Any]]:
    signatures: dict[int, dict[str, Any]] = {}
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            signatures[index] = {"raw_type": type(layer).__name__}
            continue
        signatures[index] = {
            "name": _first_value(layer, "name", "layer_name"),
            "type": _first_value(layer, "type", "layer_type", "op", "operation"),
            "tactic": _first_value(layer, "tactic", "tactic_name", "tactic_value"),
            "workspace": _first_value(
                layer,
                "workspace",
                "workspace_size",
                "workspace_size_bytes",
                "workspace_memory",
            ),
            "timing": _first_value(
                layer,
                "timing",
                "timing_ms",
                "average_ms",
                "average_time_ms",
                "latency_ms",
                "time_ms",
            ),
        }
    return signatures


def _align_layers(
    left: Mapping[int, Mapping[str, Any]], right: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """生成稳定的 layer 配对，避免 fusion 后仅按 index 误配。"""
    left_names = _unique_name_indexes(left)
    right_names = _unique_name_indexes(right)
    alignments: list[dict[str, Any]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()

    for name in sorted(set(left_names) & set(right_names)):
        left_index = left_names[name]
        right_index = right_names[name]
        alignments.append(
            {"key": f"name:{name}", "left_index": left_index, "right_index": right_index}
        )
        used_left.add(left_index)
        used_right.add(right_index)

    unique_left_indexes = set(left_names.values())
    unique_right_indexes = set(right_names.values())
    for index in sorted(set(left) & set(right)):
        if index in used_left or index in used_right:
            continue
        # If both sides have unique names, a name change represents a structural
        # add/remove (often fusion or split), not the same layer by coincidence.
        # Index fallback is reserved for missing or repeated names.
        if index in unique_left_indexes and index in unique_right_indexes:
            continue
        alignments.append(
            {"key": f"index:{index}", "left_index": index, "right_index": index}
        )
        used_left.add(index)
        used_right.add(index)

    for index in sorted(set(left) - used_left):
        alignments.append(
            {"key": f"left-index:{index}", "left_index": index, "right_index": None}
        )
    for index in sorted(set(right) - used_right):
        alignments.append(
            {"key": f"right-index:{index}", "left_index": None, "right_index": index}
        )
    return alignments


def _unique_name_indexes(
    layers: Mapping[int, Mapping[str, Any]]
) -> dict[str, int]:
    by_name: dict[str, list[int]] = {}
    for index, signature in layers.items():
        name = signature.get("name")
        if name is None or str(name) == "":
            continue
        by_name.setdefault(str(name), []).append(index)
    return {
        name: indexes[0]
        for name, indexes in by_name.items()
        if len(indexes) == 1
    }


def _first_value(payload: Mapping[str, Any], *names: str) -> Any:
    wanted = {_normalize_key(name) for name in names}
    for key, value in payload.items():
        if _normalize_key(str(key)) in wanted:
            return value
    for value in payload.values():
        if isinstance(value, Mapping):
            nested = _first_value(value, *names)
            if nested is not None:
                return nested
    return None


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _operator_counts(layers: Mapping[int, Mapping[str, Any]]) -> dict[str, int]:
    values = [signature["type"] for signature in layers.values() if signature.get("type")]
    return dict(sorted(Counter(str(value) for value in values).items()))


def _alignment_summary(
    alignments: list[dict[str, Any]],
    left: Mapping[int, Mapping[str, Any]],
    right: Mapping[int, Mapping[str, Any]],
) -> dict[str, int]:
    """统计结构变化和可见字段变化，便于筛选 profiling 重点。"""
    summary = {
        "matched_count": 0,
        "added_count": 0,
        "removed_count": 0,
        "type_changed_count": 0,
        "tactic_changed_count": 0,
        "workspace_changed_count": 0,
        "timing_changed_count": 0,
    }
    for alignment in alignments:
        left_index = alignment["left_index"]
        right_index = alignment["right_index"]
        if left_index is None:
            summary["added_count"] += 1
            continue
        if right_index is None:
            summary["removed_count"] += 1
            continue
        summary["matched_count"] += 1
        left_signature = left[left_index]
        right_signature = right[right_index]
        for field_name in ("type", "tactic", "workspace", "timing"):
            if left_signature.get(field_name) != right_signature.get(field_name):
                summary[f"{field_name}_changed_count"] += 1
    return summary


def _operator_count_delta(
    left: Mapping[int, Mapping[str, Any]], right: Mapping[int, Mapping[str, Any]]
) -> dict[str, int]:
    """返回 right-left 的算子数量变化；缺失算子类型不进入统计。"""
    left_counts = _operator_counts(left)
    right_counts = _operator_counts(right)
    names = sorted(set(left_counts) | set(right_counts))
    return {
        name: right_counts.get(name, 0) - left_counts.get(name, 0)
        for name in names
        if right_counts.get(name, 0) != left_counts.get(name, 0)
    }


def _visibility(layers: Mapping[int, Mapping[str, Any]], field: str) -> str:
    return "present" if any(signature.get(field) is not None for signature in layers.values()) else "missing"


if __name__ == "__main__":
    raise SystemExit(main())
