"""验证 TensorRT engine provenance 是否匹配固定 tuning 配置。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import (
    TensorRTTuningConfig,
    TensorRTValidationError,
    sha256_file,
    write_json,
)


def validate_provenance(
    provenance: Mapping[str, Any],
    tuning_config: TensorRTTuningConfig | None = None,
    *,
    expected_configuration: Mapping[str, Any] | None = None,
    engine_path: Path | str | None = None,
    reduced_vocab_dir: Path | str | None = None,
) -> dict[str, Any]:
    """检查 engine/reduced-vocab identity、SHA-256 和固定实验字段。"""
    reasons: list[str] = []
    if provenance.get("schema_version") != "parksight_tensorrt_engine_provenance_v1":
        reasons.append("unsupported or missing provenance schema_version")

    engine = provenance.get("engine")
    if not isinstance(engine, Mapping):
        reasons.append("engine provenance object is missing")
        engine = {}
    sha256 = engine.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        reasons.append("engine SHA-256 is missing or malformed")

    configuration = provenance.get("configuration")
    if not isinstance(configuration, Mapping):
        reasons.append("engine provenance configuration is missing")
        configuration = {}
    if tuning_config is not None and expected_configuration is not None:
        raise ValueError("provide either tuning_config or expected_configuration, not both")
    if tuning_config is not None:
        expected = tuning_config.fixed_mapping()
    elif expected_configuration is not None:
        expected = dict(expected_configuration)
    else:
        raise ValueError("one of tuning_config or expected_configuration is required")
    mismatches: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for field, expected_value in expected.items():
        if field not in configuration:
            missing.append(field)
        elif configuration[field] != expected_value:
            mismatches[field] = {
                "expected": expected_value,
                "actual": configuration[field],
            }
    if missing:
        reasons.append(f"missing fixed provenance fields: {missing}")
    if mismatches:
        reasons.append(f"fixed provenance mismatches: {mismatches}")
    engine_file: dict[str, Any] | None = None
    if engine_path is not None:
        path = Path(engine_path).resolve()
        if not path.is_file():
            reasons.append(f"engine file not found: {path}")
        else:
            actual_sha256 = sha256_file(path)
            actual_size = path.stat().st_size
            recorded_size = engine.get("size_bytes")
            engine_file = {
                "path": str(path),
                "actual_sha256": actual_sha256,
                "recorded_sha256": sha256,
                "sha256_match": sha256 == actual_sha256,
                "actual_size_bytes": actual_size,
                "recorded_size_bytes": recorded_size,
                "size_match": recorded_size == actual_size
                if isinstance(recorded_size, int) and not isinstance(recorded_size, bool)
                else None,
            }
            if sha256 != actual_sha256:
                reasons.append("recorded engine SHA-256 does not match engine file")
            if engine_file["size_match"] is False:
                reasons.append("recorded engine size does not match engine file")
    reduced_vocabulary_file: dict[str, Any] | None = None
    if reduced_vocab_dir is not None:
        root = Path(reduced_vocab_dir).resolve()
        recorded_reduced = provenance.get("reduced_vocabulary")
        if not isinstance(recorded_reduced, Mapping):
            reasons.append("reduced vocabulary provenance object is missing")
        else:
            reduced_vocabulary_file = _validate_reduced_vocabulary_files(
                recorded_reduced, root, reasons
            )
    return {
        "schema_version": "parksight_tensorrt_provenance_validation_v1",
        "valid": not reasons,
        "reasons": reasons,
        "engine_sha256": sha256 if isinstance(sha256, str) else None,
        "fixed_fields": {
            "expected": expected,
            "missing": missing,
            "mismatches": mismatches,
        },
        "engine_file": engine_file,
        "reduced_vocabulary_file": reduced_vocabulary_file,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", required=True, type=Path)
    expected_group = parser.add_mutually_exclusive_group(required=True)
    expected_group.add_argument("--tuning-config", type=Path)
    expected_group.add_argument(
        "--build-report",
        type=Path,
        help="直接使用成功 build report 的 configuration 校验 visual/LLM provenance",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        help="可选：重新计算该 engine 的 SHA-256 和大小并与 provenance 对照",
    )
    parser.add_argument(
        "--reduced-vocab-dir",
        type=Path,
        help="可选：重新计算 reduced vocabulary map/metadata 的 SHA-256 和大小",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    provenance = _read_object(args.provenance)
    if args.tuning_config is not None:
        tuning_config = TensorRTTuningConfig.load(args.tuning_config)
        result = validate_provenance(
            provenance,
            tuning_config,
            engine_path=args.engine,
            reduced_vocab_dir=args.reduced_vocab_dir,
        )
    else:
        build_report = _read_object(args.build_report)
        expected_configuration, build_report_result = _build_report_configuration(
            build_report,
            engine_path=args.engine,
        )
        result = validate_provenance(
            provenance,
            expected_configuration=expected_configuration,
            engine_path=args.engine,
            reduced_vocab_dir=args.reduced_vocab_dir,
        )
        result["build_report"] = build_report_result
        if not build_report_result["valid"]:
            result["valid"] = False
            result["reasons"].extend(build_report_result["reasons"])
    write_json(result, args.output)
    print(args.output)
    return 0 if result["valid"] else 2


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TensorRTValidationError(f"cannot read provenance: {path}") from error
    if not isinstance(payload, dict):
        raise TensorRTValidationError(f"provenance must be an object: {path}")
    return payload


def _build_report_configuration(
    payload: Mapping[str, Any],
    *,
    engine_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a build report and return its exact builder configuration."""
    reasons: list[str] = []
    if payload.get("schema_version") != "parksight_tensorrt_engine_build_v1":
        reasons.append("build report has unsupported schema_version")
    if payload.get("status") != "succeeded":
        reasons.append(f"build report must have status=succeeded, got {payload.get('status')!r}")
    configuration = payload.get("configuration")
    outputs = payload.get("outputs")
    if not isinstance(configuration, Mapping):
        reasons.append("build report configuration is missing")
        configuration = {}
    if not isinstance(outputs, list):
        reasons.append("build report outputs are missing")
        outputs = []

    output_result: dict[str, Any] | None = None
    if engine_path is not None:
        resolved_engine = engine_path.resolve()
        matching_outputs = [
            output
            for output in outputs
            if isinstance(output, Mapping)
            and isinstance(output.get("path"), str)
            and Path(output["path"]).resolve() == resolved_engine
        ]
        if len(matching_outputs) != 1:
            reasons.append("build report does not contain exactly one matching engine output")
        else:
            output = matching_outputs[0]
            if not resolved_engine.is_file():
                reasons.append(f"engine file not found: {resolved_engine}")
            else:
                actual_sha256 = sha256_file(resolved_engine)
                actual_size = resolved_engine.stat().st_size
                sha256_match = output.get("sha256") == actual_sha256
                size_match = output.get("size_bytes") == actual_size
                if not sha256_match:
                    reasons.append("build report engine SHA-256 does not match engine file")
                if not size_match:
                    reasons.append("build report engine size does not match engine file")
                output_result = {
                    "path": str(resolved_engine),
                    "recorded_sha256": output.get("sha256"),
                    "actual_sha256": actual_sha256,
                    "sha256_match": sha256_match,
                    "recorded_size_bytes": output.get("size_bytes"),
                    "actual_size_bytes": actual_size,
                    "size_match": size_match,
                }
    return dict(configuration), {
        "path": None,
        "status": payload.get("status"),
        "component": configuration.get("component"),
        "valid": not reasons,
        "reasons": reasons,
        "engine_output": output_result,
    }


def _validate_reduced_vocabulary_files(
    recorded: Mapping[str, Any], root: Path, reasons: list[str]
) -> dict[str, Any] | None:
    map_path = root / "vocab_map.safetensors"
    metadata_path = root / "reduced_vocab.json"
    selection_report_path = root / "selection_report.json"
    if (
        not map_path.is_file()
        or not metadata_path.is_file()
        or not selection_report_path.is_file()
    ):
        reasons.append(f"reduced vocabulary files are missing under: {root}")
        return None
    result = {
        "directory": str(root),
        "map": _compare_file_identity(recorded.get("map"), map_path, reasons, "map"),
        "metadata": _compare_file_identity(
            recorded.get("metadata"), metadata_path, reasons, "metadata"
        ),
        "selection_report": _compare_file_identity(
            recorded.get("selection_report"),
            selection_report_path,
            reasons,
            "selection report",
        ),
    }
    return result


def _compare_file_identity(
    recorded: Any, path: Path, reasons: list[str], label: str
) -> dict[str, Any]:
    if not isinstance(recorded, Mapping):
        reasons.append(f"reduced vocabulary {label} provenance is missing")
        recorded = {}
    actual_sha256 = sha256_file(path)
    actual_size = path.stat().st_size
    recorded_sha256 = recorded.get("sha256")
    recorded_size = recorded.get("size_bytes")
    sha256_match = recorded_sha256 == actual_sha256
    size_match = (
        recorded_size == actual_size
        if isinstance(recorded_size, int) and not isinstance(recorded_size, bool)
        else None
    )
    if not sha256_match:
        reasons.append(f"recorded reduced vocabulary {label} SHA-256 does not match file")
    if size_match is False:
        reasons.append(f"recorded reduced vocabulary {label} size does not match file")
    return {
        "path": str(path),
        "actual_sha256": actual_sha256,
        "recorded_sha256": recorded_sha256,
        "sha256_match": sha256_match,
        "actual_size_bytes": actual_size,
        "recorded_size_bytes": recorded_size,
        "size_match": size_match,
    }


if __name__ == "__main__":
    raise SystemExit(main())
