"""Prepare a QKV/gate-up fusion graph; keep the original graph and weights unchanged."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import struct


_LAYOUT_SOURCES = {
    "cpp/kernels/int4GroupwiseGemmKernels/int4WoQGemvCuda.cu": "d1fb5cca18095616069dcf564bf7102bbaf031c73ed4e8b19988ef976770ca61",
    "cpp/kernels/int4GroupwiseGemmKernels/int4WoqGemmCuda.cu": "461ed9c98a0979f24a12ac7136eace02bf4f238578230544704be8d429ca1cb3",
    "cpp/plugins/int4GroupwiseGemmPlugin/int4GroupwiseGemmPlugin.cpp": "3547c288685698ef88cdb4a3a7ef99d9e9771f8492feb0c14317ceec24f480a8",
}


def prepare(model_path: Path, evidence_path: Path, output: Path) -> dict:
    import onnx

    raw = model_path.read_bytes()
    evidence = json.loads(evidence_path.read_text())
    if hashlib.sha256(raw).hexdigest() != evidence["graph_sha256"]:
        raise ValueError("graph does not match scale evidence")
    if output.exists():
        raise ValueError("output directory already exists")
    for suffix, expected in _LAYOUT_SOURCES.items():
        matches = [item for path, item in evidence.get("sources", {}).items() if path.endswith("/" + suffix)]
        if len(matches) != 1 or matches[0]["sha256"] != expected:
            raise ValueError("unsupported or unverified plugin packing source")
    model = onnx.load_model_from_string(raw)
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}
    producers = {name: node for node in graph.node for name in node.output}
    if len(initializers) != len(graph.initializer):
        raise ValueError("duplicate initializer")
    if len({n.name for n in graph.node}) != len(graph.node):
        raise ValueError("ambiguous graph node names")
    scales = {item["name"]: item for item in evidence["scales"]}
    groups = {}
    for node in graph.node:
        if node.domain != "trt_edgellm" or node.op_type != "Int4GroupwiseGemmPlugin":
            continue
        if len(node.input) != 3 or len(node.output) != 1:
            raise ValueError("unsupported INT4 plugin interface")
        match = re.fullmatch(r"_model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)_proj\.weight", node.input[1])
        if not match:
            raise ValueError("unsupported INT4 projection name")
        layer, section, projection = match.groups()
        family = "qkv" if section == "self_attn" and projection in {"q", "k", "v"} else (
            "gate_up" if section == "mlp" and projection in {"gate", "up"} else None)
        if family:
            key = (int(layer), family)
            if projection in groups.setdefault(key, {}):
                raise ValueError("duplicate projection in fusion group")
            groups[key][projection] = node

    def external(tensor, dtype, dimensions):
        info = {item.key: item.value for item in tensor.external_data}
        size = math.prod(dimensions) * (2 if dtype == 10 else 1)
        if (tensor.data_type != dtype or list(tensor.dims) != dimensions or tensor.raw_data
                or tensor.data_location != onnx.TensorProto.EXTERNAL
                or info.get("location") != "model.onnx.data"
                or int(info.get("length", -1)) != size):
            raise ValueError(f"unsupported external tensor layout: {tensor.name}")
        offset = int(info.get("offset", -1))
        if offset < 0 or offset + size > evidence["external_weights"]["size_bytes"]:
            raise ValueError("external tensor range out of bounds")
        return {"offset": offset, "length": size}

    operations = []
    replacements = {}
    removed = set()
    total_bytes = 0
    group_report = []
    existing_names = {t.name for t in graph.initializer} | set(producers) | {n.name for n in graph.node}
    for (layer, family), members in sorted(groups.items()):
        order = ("q", "k", "v") if family == "qkv" else ("gate", "up")
        if set(members) != set(order):
            raise ValueError("incomplete fusion group")
        nodes = [members[name] for name in order]
        attributes = [{a.name: onnx.helper.get_attribute_value(a) for a in n.attribute} for n in nodes]
        if any(set(a) != {"gemm_n", "gemm_k", "group_size"} for a in attributes):
            raise ValueError("unsupported plugin attributes")
        k = attributes[0]["gemm_k"]
        widths = [a["gemm_n"] for a in attributes]
        if (k <= 0 or k % 128 or any(n <= 0 or n % 128 for n in widths)
                or any(a["gemm_k"] != k or a["group_size"] != 128 for a in attributes)):
            raise ValueError("fusion requires equal K, group128, and N aligned to128")
        hidden = []
        scale_bytes = []
        for node in nodes:
            mul = producers[node.input[0]]
            if mul.domain not in ("", "ai.onnx") or mul.op_type != "Mul" or len(mul.input) != 2:
                raise ValueError("activation is not a supported pre-quant Mul")
            constants = [name for name in mul.input if name in initializers]
            if len(constants) != 1 or not constants[0].endswith(".pre_quant_scale"):
                raise ValueError("missing pre-quant scale")
            name = constants[0]
            ref = external(initializers[name], 10, [k])
            item = scales[name]
            if item["external_data"] != {a.key: a.value for a in initializers[name].external_data}:
                raise ValueError("scale evidence points at a different range")
            data = base64.b64decode(item["base64"], validate=True)
            values = struct.unpack("<" + str(k) + "e", data)
            if (len(data) != ref["length"] or hashlib.sha256(data).hexdigest() != item["sha256"]
                    or not all(math.isfinite(v) and v > 0 for v in values)):
                raise ValueError("invalid pre-quant scale bytes")
            scale_bytes.append(data)
            hidden.append(next(name for name in mul.input if name not in constants))
        if len(set(hidden)) != 1 or len(set(scale_bytes)) != 1:
            raise ValueError("fusion inputs or pre-quant scale bytes differ")
        prefix = f"parksight_fusion_layer{layer}_{family}"
        names = [prefix + suffix for suffix in ("_weight", "_scale", "_output", "_plugin")]
        if any(name in existing_names for name in names):
            raise ValueError("fusion name collision")
        weight_refs = [external(initializers[n.input[1]], 3, [width // 2, k]) for n, width in zip(nodes, widths)]
        scale_refs = [external(initializers[n.input[2]], 10, [k // 128, width]) for n, width in zip(nodes, widths)]
        for name, dtype, dims, mode, refs in (
            (names[0], 3, [sum(widths) // 2, k], "concat", weight_refs),
            (names[1], 10, [k // 128, sum(widths)], "row_concat", scale_refs),
        ):
            length = sum(ref["length"] for ref in refs)
            tensor = onnx.TensorProto(name=name, data_type=dtype, dims=dims, data_location=onnx.TensorProto.EXTERNAL)
            for key, value in {"location": "fusion_weights.data", "offset": str(total_bytes), "length": str(length)}.items():
                tensor.external_data.add(key=key, value=value)
            graph.initializer.append(tensor)
            operations.append({"name": name, "mode": mode, "offset": total_bytes, "length": length,
                               "rows": k // 128 if mode == "row_concat" else 1, "sources": refs})
            total_bytes += length
        fused = onnx.helper.make_node("Int4GroupwiseGemmPlugin", [nodes[0].input[0], names[0], names[1]],
                                      [names[2]], name=names[3], domain="trt_edgellm",
                                      gemm_n=sum(widths), gemm_k=k, group_size=128)
        replacement = [fused]
        start = 0
        for index, (node, width) in enumerate(zip(nodes, widths)):
            slice_inputs = [names[2]]
            for suffix, value in (("start", start), ("end", start + width), ("axis", 2), ("step", 1)):
                name = f"{prefix}_slice{index}_{suffix}"
                if name in existing_names:
                    raise ValueError("slice initializer name collision")
                graph.initializer.append(onnx.helper.make_tensor(name, onnx.TensorProto.INT64, [1], [value]))
                slice_inputs.append(name)
            replacement.append(onnx.helper.make_node("Slice", slice_inputs, list(node.output),
                                                     name=f"{prefix}_slice{index}"))
            start += width
        removed.update(n.name for n in nodes)
        first = next(n.name for n in graph.node if n.name in {node.name for node in nodes})
        replacements[first] = replacement
        group_report.append({"layer": layer, "family": family, "k": k, "widths": widths,
                             "pre_quant_scale_sha256": hashlib.sha256(scale_bytes[0]).hexdigest()})
    if not group_report:
        raise ValueError("no supported fusion groups")
    rewritten = []
    for node in graph.node:
        if node.name in replacements:
            rewritten.extend(replacements[node.name])
        elif node.name not in removed:
            rewritten.append(node)
    del graph.node[:]
    graph.node.extend(rewritten)
    # Remove dead pre-quant Muls and their original packed tensors recursively.
    live = {value.name for value in graph.output}
    kept = []
    for node in reversed(graph.node):
        if any(name in live for name in node.output):
            kept.append(node)
            live.update(node.input)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    tensors = [t for t in graph.initializer if t.name in live]
    del graph.initializer[:]
    graph.initializer.extend(tensors)
    known = {value.name for value in graph.input} | {tensor.name for tensor in graph.initializer}
    for node in graph.node:
        if any(name and name not in known for name in node.input):
            raise ValueError(f"non-topological fusion graph at {node.name}")
        if any(name and name in known for name in node.output):
            raise ValueError(f"duplicate graph output at {node.name}")
        known.update(node.output)
    if any(value.name not in known for value in graph.output):
        raise ValueError("fusion lost a graph output")
    payload = model.SerializeToString()
    plan = {"schema_version": "parksight_edgellm_int4_fusion_plan_v1",
            "layout": "edgellm_interleave4_n_major_group128", "source_graph_sha256": evidence["graph_sha256"],
            "source_weights": evidence["external_weights"], "candidate_graph_sha256": hashlib.sha256(payload).hexdigest(),
            "fusion_weights_bytes": total_bytes, "groups": group_report, "operations": operations,
            "layout_sources": _LAYOUT_SOURCES,
            "scope": "Prepared graph only; materialization, numerical GPU verification and benchmark required"}
    output.mkdir(parents=True)
    (output / "model.onnx").write_bytes(payload)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--scale-evidence", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    plan = prepare(args.model, args.scale_evidence, args.output_directory)
    print(json.dumps({"groups": len(plan["groups"]), "fusion_weights_bytes": plan["fusion_weights_bytes"]}))


if __name__ == "__main__":
    main()
