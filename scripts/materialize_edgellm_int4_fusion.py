"""Materialize a prepared INT4 fusion graph with bounded memory and pinned source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import BinaryIO


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_operation(source: BinaryIO, target: BinaryIO, operation: dict, source_size: int) -> int:
    """Packed interleave4 weights concatenate by N; group scales concatenate by row."""
    mode = operation["mode"]
    rows = operation["rows"]
    if mode not in {"concat", "row_concat"} or type(rows) is not int or rows <= 0:
        raise ValueError("unsupported fusion operation")
    if mode == "concat" and rows != 1:
        raise ValueError("packed weight concatenation requires one row")
    if target.tell() != operation["offset"]:
        raise ValueError("fusion destination offset mismatch")
    references = operation["sources"]
    if not references or sum(ref["length"] for ref in references) != operation["length"]:
        raise ValueError("fusion operation length mismatch")
    for ref in references:
        if (type(ref["offset"]) is not int or type(ref["length"]) is not int
                or ref["offset"] < 0 or ref["length"] <= 0
                or ref["offset"] + ref["length"] > source_size or ref["length"] % rows):
            raise ValueError("invalid source tensor range")
    written = 0
    for row in range(rows):
        for ref in references:
            length = ref["length"] // rows
            source.seek(ref["offset"] + row * length)
            remaining = length
            while remaining:
                data = source.read(min(remaining, 1024 * 1024))
                if not data:
                    raise ValueError("source tensor was truncated during materialization")
                target.write(data)
                remaining -= len(data)
                written += len(data)
    return written


def materialize(prepared: Path, source_data: Path, output: Path) -> dict:
    plan = json.loads((prepared / "plan.json").read_text())
    if (plan["schema_version"] != "parksight_edgellm_int4_fusion_plan_v1"
            or plan["layout"] != "edgellm_interleave4_n_major_group128"):
        raise ValueError("unsupported fusion plan")
    if output.exists() or output.is_symlink():
        raise ValueError("output directory already exists")
    source_size = source_data.stat().st_size
    if source_size != plan["source_weights"]["size_bytes"] or file_hash(source_data) != plan["source_weights"]["sha256"]:
        raise ValueError("source weights do not match the pinned plan")
    if file_hash(prepared / "model.onnx") != plan["candidate_graph_sha256"]:
        raise ValueError("prepared graph SHA-256 mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < plan["fusion_weights_bytes"] + 16 * 1024 * 1024:
        raise ValueError("insufficient free space for fusion data")
    temporary = Path(tempfile.mkdtemp(prefix=".parksight-int4-fusion-", dir=output.parent))
    try:
        with source_data.open("rb") as source, (temporary / "fusion_weights.data").open("xb") as target:
            for operation in plan["operations"]:
                copy_operation(source, target, operation, source_size)
            if target.tell() != plan["fusion_weights_bytes"]:
                raise ValueError("final fusion data size mismatch")
        shutil.copyfile(prepared / "model.onnx", temporary / "model.onnx")
        (temporary / "model.onnx.data").symlink_to(source_data.resolve())
        shutil.copyfile(prepared / "plan.json", temporary / "plan.json")
        if file_hash(source_data) != plan["source_weights"]["sha256"]:
            raise ValueError("source data changed during materialization")
        result = {
            "schema_version": "parksight_edgellm_int4_fusion_materialization_v1",
            "status": "materialized", "source_graph_sha256": plan["source_graph_sha256"],
            "source_weights_sha256": plan["source_weights"]["sha256"],
            "candidate_graph_sha256": file_hash(temporary / "model.onnx"),
            "fusion_data_sha256": file_hash(temporary / "fusion_weights.data"),
            "fusion_data_bytes": plan["fusion_weights_bytes"], "group_count": len(plan["groups"]),
            "plan_sha256": file_hash(prepared / "plan.json"),
            "scope": "Prepared fusion data only; GPU numerical and full workload validation remain required",
        }
        (temporary / "materialization.json").write_text(json.dumps(result, indent=2) + "\n")
        if output.exists() or output.is_symlink():
            raise ValueError("output appeared during materialization")
        temporary.rename(output)
        return result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-directory", required=True, type=Path)
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(args.prepared_directory, args.source_data, args.output_directory)))


if __name__ == "__main__":
    main()
