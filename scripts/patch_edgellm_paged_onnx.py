"""Inject the v0.9.1 AttentionPlugin page-table input into a Qwen3-VL ONNX graph.

TensorRT Edge-LLM v0.9.1 exports AttentionPlugin with seven inputs in the
separate-Q/K/V layout.  The paged runtime adds one input after those seven;
this helper patches a copy of the graph so the builder and plugin see the same
contract without modifying the source ONNX artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def patch_paged_attention_graph(
    source: Path,
    destination: Path,
    *,
    max_pages_per_sequence: int,
    max_pool_pages: int,
    page_table_name: str = "kv_page_table",
) -> int:
    """Patch a Qwen3-VL v0.9.1 graph and return the AttentionPlugin count."""
    if max_pages_per_sequence <= 0:
        raise ValueError("max_pages_per_sequence must be positive")
    if max_pool_pages <= 0:
        raise ValueError("max_pool_pages must be positive")
    if not page_table_name:
        raise ValueError("page_table_name must not be empty")

    model = onnx.load(str(source))
    attention_nodes = [
        node for node in model.graph.node if node.op_type == "AttentionPlugin"
    ]
    if not attention_nodes:
        raise ValueError("ONNX graph contains no AttentionPlugin nodes")
    if any(len(node.input) != 7 for node in attention_nodes):
        raise ValueError(
            "paged v0.9.1 patch requires exactly seven original AttentionPlugin inputs"
        )

    graph_inputs = {value.name for value in model.graph.input}

    def set_paged_pool_shape(value_info: onnx.ValueInfoProto) -> None:
        shape = value_info.type.tensor_type.shape
        old_dims = list(shape.dim)
        if len(old_dims) != 5:
            raise ValueError(f"expected five KV dimensions for {value_info.name}")
        num_kv_heads = old_dims[2].dim_value
        head_dim = old_dims[4].dim_value
        if not num_kv_heads or not head_dim:
            raise ValueError(f"KV head dimensions must be static for {value_info.name}")
        shape.ClearField("dim")
        for dim in (2, max_pool_pages, 128, num_kv_heads, head_dim):
            shape.dim.add().dim_value = dim

    # The original export describes each cache as [B, 2, Hkv, past_len, D].
    # Paged mode owns a shared pool instead: [2, num_pages, tokens_per_page,
    # Hkv, D].  Update both the graph inputs and declared outputs so TensorRT
    # does not compare the paged profile against the old dense axis order.
    for value_info in model.graph.input:
        if value_info.name.startswith("past_key_values_"):
            set_paged_pool_shape(value_info)
    for value_info in model.graph.output:
        if value_info.name.startswith("present_key_values_"):
            set_paged_pool_shape(value_info)

    if page_table_name not in graph_inputs:
        model.graph.input.append(
            helper.make_tensor_value_info(
                page_table_name,
                TensorProto.INT32,
                ["batch", 2, max_pages_per_sequence],
            )
        )

    for node in attention_nodes:
        node.input.append(page_table_name)
        attribute_names = {attribute.name for attribute in node.attribute}
        if "enable_paged_kv" in attribute_names:
            for attribute in node.attribute:
                if attribute.name == "enable_paged_kv":
                    attribute.i = 1
                    break
        else:
            node.attribute.append(helper.make_attribute("enable_paged_kv", 1))

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the INT4 weights external.  Plain ``onnx.save`` materializes the
    # 1.3 GiB initializer blob into model.onnx, which duplicates the export
    # and makes the staging copy needlessly large.
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )
    return len(attention_nodes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--max-pages-per-sequence", type=int, required=True)
    parser.add_argument("--max-pool-pages", type=int, required=True)
    args = parser.parse_args()
    count = patch_paged_attention_graph(
        args.source,
        args.destination,
        max_pages_per_sequence=args.max_pages_per_sequence,
        max_pool_pages=args.max_pool_pages,
    )
    print(f"patched {count} AttentionPlugin nodes: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
