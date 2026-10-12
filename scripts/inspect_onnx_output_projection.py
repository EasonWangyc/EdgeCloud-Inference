"""Inspect an ONNX output projection without loading external weight files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def projection_from_graph(graph: Any, output_name: str = "logits") -> dict[str, Any]:
    """Follow a supported output chain to MatMul and report its weight metadata."""
    if output_name not in {value.name for value in graph.output}:
        raise ValueError(f"graph has no output named {output_name!r}")
    producers = {}
    for node in graph.node:
        for name in node.output:
            if not name:
                continue
            if name in producers:
                raise ValueError(f"duplicate producer for {name!r}")
            producers[name] = node
    initializers = {}
    for tensor in graph.initializer:
        if tensor.name in initializers:
            raise ValueError(f"duplicate initializer {tensor.name!r}")
        initializers[tensor.name] = tensor

    path = []
    seen = set()
    current = output_name
    while True:
        if current in seen:
            raise ValueError("cycle in output projection path")
        seen.add(current)
        node = producers.get(current)
        if node is None:
            raise ValueError(f"missing producer for {current!r}")
        if node.domain not in ("", "ai.onnx"):
            raise ValueError(f"unsupported operator domain {node.domain!r}")
        path.append({"name": node.name, "op_type": node.op_type,
                     "inputs": list(node.input), "outputs": list(node.output)})
        if node.op_type == "MatMul":
            if len(node.input) != 2 or node.input[1] not in initializers:
                raise ValueError("MatMul weight must be a direct initializer")
            weight = initializers[node.input[1]]
            if len(weight.dims) != 2 or any(dim <= 0 for dim in weight.dims):
                raise ValueError("projection weight must have two positive dimensions")
            return {
                "output_name": output_name,
                "output_to_projection_path": path,
                "weight": {
                    "name": weight.name, "data_type": int(weight.data_type),
                    "dimensions": list(weight.dims),
                    "data_location": int(weight.data_location),
                    "external_data": {item.key: item.value for item in weight.external_data},
                    "raw_data_bytes": len(weight.raw_data),
                },
                "scope": "Graph metadata only; does not bind TensorRT tactics or engine weights.",
            }
        if node.op_type not in {"Cast", "Identity", "Reshape", "Squeeze", "Unsqueeze", "Transpose"}:
            raise ValueError(f"unsupported output path operator {node.op_type!r}")
        if not node.input or not node.input[0]:
            raise ValueError("output path operator has no data input")
        current = node.input[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-name", default="logits")
    args = parser.parse_args()
    payload = args.model.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.model_sha256.lower():
        parser.error("model SHA-256 mismatch")
    if args.output.exists():
        parser.error("output already exists; use a fresh evidence path")
    try:
        import onnx
    except ImportError:
        parser.error("ONNX is optional; install the documented inspection dependency separately")
    # Parse the verified protobuf bytes directly; never resolve external_data paths.
    model = onnx.load_model_from_string(payload)
    result = projection_from_graph(model.graph, args.output_name)
    result.update({
        "schema_version": "parksight_onnx_output_projection_v1",
        "onnx_version": onnx.__version__,
        "model": {"path": str(args.model), "sha256": digest, "size_bytes": len(payload)},
        "external_weights_loaded": False,
        "graph_node_count": len(model.graph.node),
        "graph_initializer_count": len(model.graph.initializer),
    })
    result["weight"]["data_type_name"] = onnx.TensorProto.DataType.Name(result["weight"]["data_type"])
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
