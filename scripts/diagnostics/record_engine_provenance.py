"""记录已构建 TensorRT engine 的文件哈希与实验配置身份。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import (
    TensorRTValidationError,
    build_engine_provenance,
    sha256_file,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="由调用环境填写的设备、版本、profile 和 runtime 参数",
    )
    parser.add_argument(
        "--build-report",
        type=Path,
        help=(
            "读取 build_edgellm_vlm_engines.py 的成功 build report；"
            "自动绑定 component、builder 配置、profile 和输出 SHA-256"
        ),
    )
    parser.add_argument(
        "--reduced-vocab-dir",
        type=Path,
        help="可选 reduced vocabulary 目录，记录 vocab_map 和 metadata 的身份",
    )
    args = parser.parse_args(argv)
    if args.metadata_json is not None and args.build_report is not None:
        parser.error("use either --metadata-json or --build-report, not both")

    metadata: dict[str, Any] = {}
    build_report_identity: dict[str, Any] | None = None
    if args.metadata_json is not None:
        try:
            payload = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TensorRTValidationError(
                f"cannot read provenance metadata: {args.metadata_json}"
            ) from error
        if not isinstance(payload, dict):
            raise TensorRTValidationError("provenance metadata must be an object")
        metadata = payload
    if args.build_report is not None:
        metadata, build_report_identity = _metadata_from_build_report(
            args.build_report,
            engine_path=args.engine,
        )
    provenance = build_engine_provenance(engine_path=args.engine, metadata=metadata)
    if build_report_identity is not None:
        provenance["build_report"] = build_report_identity
    evidence_sources: dict[str, str] = {}
    if args.metadata_json is not None:
        evidence_sources["metadata_json"] = str(args.metadata_json)
    if args.build_report is not None:
        evidence_sources["build_report"] = str(args.build_report)
    if args.reduced_vocab_dir is not None:
        provenance["reduced_vocabulary"] = _reduced_vocabulary_identity(
            args.reduced_vocab_dir
        )
        evidence_sources["reduced_vocab_dir"] = str(args.reduced_vocab_dir)
    if evidence_sources:
        provenance["evidence_sources"] = evidence_sources
    write_json(provenance, args.output)
    print(args.output)
    return 0


def _metadata_from_build_report(
    path: Path,
    *,
    engine_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract configuration only from a successful report for this engine."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TensorRTValidationError(f"cannot read build report: {path}") from error
    if not isinstance(payload, dict):
        raise TensorRTValidationError("build report must be an object")
    if payload.get("schema_version") != "parksight_tensorrt_engine_build_v1":
        raise TensorRTValidationError("build report has unsupported schema_version")
    if payload.get("status") != "succeeded":
        raise TensorRTValidationError(
            f"build report must have status=succeeded, got {payload.get('status')!r}"
        )
    configuration = payload.get("configuration")
    outputs = payload.get("outputs")
    if not isinstance(configuration, dict) or not isinstance(outputs, list):
        raise TensorRTValidationError(
            "build report must contain configuration and outputs"
        )
    resolved_engine = engine_path.resolve()
    matching_outputs = [
        output
        for output in outputs
        if isinstance(output, dict)
        and isinstance(output.get("path"), str)
        and Path(output["path"]).resolve() == resolved_engine
    ]
    if len(matching_outputs) != 1:
        raise TensorRTValidationError(
            "build report does not contain exactly one matching engine output"
        )
    output = matching_outputs[0]
    actual_sha256 = sha256_file(engine_path)
    if output.get("sha256") != actual_sha256:
        raise TensorRTValidationError(
            "build report engine SHA-256 does not match the engine file"
        )
    actual_size = engine_path.stat().st_size
    if output.get("size_bytes") != actual_size:
        raise TensorRTValidationError(
            "build report engine size does not match the engine file"
        )
    metadata: dict[str, Any] = {
        **configuration,
        "builder_environment": payload.get("builder_environment", {}),
        "build_report_status": payload["status"],
    }
    return metadata, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "component": configuration.get("component"),
        "engine_output_sha256_verified": True,
    }


def _reduced_vocabulary_identity(directory: Path) -> dict[str, Any]:
    root = directory.resolve()
    if not root.is_dir():
        raise TensorRTValidationError(f"reduced vocabulary directory not found: {root}")
    map_path = root / "vocab_map.safetensors"
    metadata_path = root / "reduced_vocab.json"
    selection_report_path = root / "selection_report.json"
    missing = [
        str(path)
        for path in (map_path, metadata_path, selection_report_path)
        if not path.is_file()
    ]
    if missing:
        raise TensorRTValidationError(
            f"reduced vocabulary directory is missing required files: {missing}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TensorRTValidationError(
            f"cannot read reduced vocabulary metadata: {metadata_path}"
        ) from error
    if not isinstance(metadata, dict):
        raise TensorRTValidationError("reduced vocabulary metadata must be an object")
    try:
        selection_report = json.loads(
            selection_report_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TensorRTValidationError(
            f"cannot read reduced vocabulary selection report: {selection_report_path}"
        ) from error
    if not isinstance(selection_report, dict):
        raise TensorRTValidationError(
            "reduced vocabulary selection report must be an object"
        )
    report_map = selection_report.get("map")
    actual_map_sha256 = sha256_file(map_path)
    if not isinstance(report_map, dict) or report_map.get("sha256") != actual_map_sha256:
        raise TensorRTValidationError(
            "reduced vocabulary selection report does not match map SHA-256"
        )
    if report_map.get("count") != metadata.get("reduced_vocab_size"):
        raise TensorRTValidationError(
            "reduced vocabulary selection report count does not match metadata"
        )
    return {
        "directory": str(root),
        "map": _file_identity(map_path),
        "metadata": {
            **_file_identity(metadata_path),
            "vocab_size": metadata.get("vocab_size"),
            "reduced_vocab_size": metadata.get("reduced_vocab_size"),
        },
        "selection_report": {
            **_file_identity(selection_report_path),
            "map_sha256_verified": True,
            "map_count": report_map.get("count"),
        },
    }


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
