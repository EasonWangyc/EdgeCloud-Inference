"""Stage an isolated paged-KV ONNX export without changing the dense source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


TOKENS_PER_PAGE = 128


def expected_attention_count(config_path: Path) -> int:
    """Read the exported model's layer count before accepting a paged graph."""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"ONNX model config is missing or invalid: {config_path}") from error
    count = config.get("num_hidden_layers") if isinstance(config, dict) else None
    if type(count) is not int or count <= 0:
        raise ValueError(f"ONNX model config has no positive num_hidden_layers: {config_path}")
    return count


def validate_paged_graph_contract(
    model: Any, *, attention_count: int, max_batch_size: int,
    max_pages_per_sequence: int,
    max_kv_pool_pages: int, int32_type: int,
) -> dict[str, int]:
    """Check the page-table binding and cache shapes before publishing the stage."""
    graph = model.graph
    attention_nodes = [node for node in graph.node if node.op_type == "AttentionPlugin"]
    if len(attention_nodes) != attention_count or attention_count == 0:
        raise ValueError("staged ONNX AttentionPlugin count does not match the patch")
    for node in attention_nodes:
        if len(node.input) != 8 or node.input[7] != "kv_page_table":
            raise ValueError("staged ONNX AttentionPlugin has an invalid page-table binding")
        if not any(attribute.name == "enable_paged_kv" and attribute.i == 1
                   for attribute in node.attribute):
            raise ValueError("staged ONNX AttentionPlugin has paged KV disabled")

    page_tables = [value for value in graph.input if value.name == "kv_page_table"]
    if len(page_tables) != 1:
        raise ValueError("staged ONNX must have one kv_page_table input")
    table = page_tables[0].type.tensor_type
    dimensions = list(table.shape.dim)
    if (table.elem_type != int32_type or len(dimensions) != 3
            or not (dimensions[0].dim_param or dimensions[0].dim_value >= max_batch_size)
            or dimensions[1].dim_value != 2
            or dimensions[2].dim_value != max_pages_per_sequence):
        raise ValueError("staged ONNX kv_page_table dtype or shape is invalid")

    cache_inputs = [value for value in graph.input
                    if value.name.startswith("past_key_values_")]
    cache_outputs = [value for value in graph.output
                     if value.name.startswith("present_key_values_")]
    if len(cache_inputs) != attention_count or len(cache_outputs) != attention_count:
        raise ValueError("staged ONNX KV cache bindings do not match AttentionPlugin count")
    for value in (*cache_inputs, *cache_outputs):
        dims = list(value.type.tensor_type.shape.dim)
        if (len(dims) != 5
                or [dim.dim_value for dim in dims[:3]]
                != [2, max_kv_pool_pages, TOKENS_PER_PAGE]
                or any(dim.dim_value <= 0 for dim in dims[3:])):
            raise ValueError(f"staged ONNX paged KV pool shape is invalid: {value.name}")

    def by_layer(values: list[Any], prefix: str) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        for value in values:
            suffix = value.name[len(prefix):]
            if not suffix or suffix in layers:
                raise ValueError(f"staged ONNX has duplicate or unnamed KV cache layer: {value.name}")
            layers[suffix] = value.type.tensor_type
        return layers

    inputs_by_layer = by_layer(cache_inputs, "past_key_values_")
    outputs_by_layer = by_layer(cache_outputs, "present_key_values_")
    if inputs_by_layer.keys() != outputs_by_layer.keys():
        raise ValueError("staged ONNX past/present KV cache layers do not match")
    for layer, past in inputs_by_layer.items():
        present = outputs_by_layer[layer]
        past_shape = [dim.dim_value for dim in past.shape.dim]
        present_shape = [dim.dim_value for dim in present.shape.dim]
        if past.elem_type != present.elem_type or past_shape != present_shape:
            raise ValueError(f"staged ONNX past/present KV cache shape or dtype differs: {layer}")
    return {
        "attention_plugin_count": attention_count,
        "kv_cache_input_count": len(cache_inputs),
        "kv_cache_output_count": len(cache_outputs),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_paged_onnx(
    source_root: Path,
    destination_root: Path,
    *,
    max_batch_size: int,
    max_kv_cache_capacity: int,
    max_kv_pool_pages: int,
) -> dict[str, object]:
    """Patch a copy of the LLM graph and atomically publish its external data."""
    if min(max_batch_size, max_kv_cache_capacity, max_kv_pool_pages) <= 0:
        raise ValueError("batch, KV capacity and pool pages must be positive")
    max_pages_per_sequence = (max_kv_cache_capacity + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE
    required_pool_pages = max_batch_size * max_pages_per_sequence
    if max_kv_pool_pages < required_pool_pages:
        raise ValueError(
            f"max_kv_pool_pages={max_kv_pool_pages} is below required {required_pool_pages}"
        )
    source_llm = source_root.resolve() / "llm"
    source_graph = source_llm / "model.onnx"
    if not source_graph.is_file():
        raise FileNotFoundError(source_graph)
    layer_count = expected_attention_count(source_llm / "config.json")
    destination_root = destination_root.resolve()
    if destination_root.exists():
        raise FileExistsError(f"refusing to overwrite staged ONNX: {destination_root}")
    destination_root.parent.mkdir(parents=True, exist_ok=True)

    # Import ONNX only for this explicit export stage; normal no-hardware
    # tests and dense engine builds do not require the package.
    import onnx
    if __package__:
        from scripts.experiments.patch_edgellm_paged_onnx import patch_paged_attention_graph
    else:
        from patch_edgellm_paged_onnx import patch_paged_attention_graph

    with tempfile.TemporaryDirectory(
        prefix=".paged-onnx-", dir=destination_root.parent
    ) as temporary:
        staged_root = Path(temporary)
        staged_llm = staged_root / "llm"
        staged_llm.mkdir()
        for entry in source_llm.iterdir():
            if entry.is_file() and entry.name not in {"model.onnx", "model.onnx.data"}:
                shutil.copy2(entry, staged_llm / entry.name)
        staged_graph = staged_llm / "model.onnx"
        node_count = patch_paged_attention_graph(
            source_graph,
            staged_graph,
            max_pages_per_sequence=max_pages_per_sequence,
            max_pool_pages=max_kv_pool_pages,
        )
        if node_count != layer_count:
            raise ValueError(
                f"paged ONNX AttentionPlugin count {node_count} differs from "
                f"num_hidden_layers {layer_count}"
            )
        graph = onnx.load(str(staged_graph), load_external_data=False)
        graph_validation = validate_paged_graph_contract(
            graph,
            attention_count=node_count,
            max_batch_size=max_batch_size,
            max_pages_per_sequence=max_pages_per_sequence,
            max_kv_pool_pages=max_kv_pool_pages,
            int32_type=onnx.TensorProto.INT32,
        )
        external_locations = {
            entry.value
            for initializer in graph.graph.initializer
            for entry in initializer.external_data
            if entry.key == "location"
        }
        if not external_locations or any(not (staged_llm / name).is_file()
                                         for name in external_locations):
            raise ValueError("staged ONNX external weights are missing")
        report: dict[str, object] = {
            "schema_version": "parksight_paged_onnx_stage_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root.resolve()),
            "destination_root": str(destination_root),
            "attention_plugin_count": node_count,
            "graph_validation": graph_validation,
            "tokens_per_page": TOKENS_PER_PAGE,
            "max_pages_per_sequence": max_pages_per_sequence,
            "max_kv_pool_pages": max_kv_pool_pages,
            "max_batch_size": max_batch_size,
            "source_graph_sha256": _sha256(source_graph),
            "staged_files_sha256": {
                str(path.relative_to(staged_root)): _sha256(path)
                for path in sorted(staged_root.rglob("*")) if path.is_file()
            },
        }
        (staged_root / "paged_onnx_provenance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staged_root, destination_root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--max-batch-size", required=True, type=int)
    parser.add_argument("--max-kv-cache-capacity", required=True, type=int)
    parser.add_argument("--max-kv-pool-pages", required=True, type=int)
    args = parser.parse_args()
    report = prepare_paged_onnx(
        args.source_root,
        args.destination_root,
        max_batch_size=args.max_batch_size,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        max_kv_pool_pages=args.max_kv_pool_pages,
    )
    print(json.dumps({"destination_root": report["destination_root"],
                      "attention_plugin_count": report["attention_plugin_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
