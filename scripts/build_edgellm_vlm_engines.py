"""使用固定版本的 TensorRT Edge-LLM 构建 VLM 的两个 engine。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import sha256_file


def build_commands(
    *,
    edge_root: Path,
    onnx_root: Path,
    engine_root: Path,
    max_batch_size: int,
    max_input_len: int,
    max_kv_cache_capacity: int,
    min_image_tokens: int,
    max_image_tokens: int,
    max_image_tokens_per_image: int,
    max_kv_pool_pages: int | None = None,
) -> tuple[list[str], list[str]]:
    """生成先 LLM、后视觉编码器的两条确定性构建命令。"""
    llm_command = [
        str(edge_root / "build" / "examples" / "llm" / "llm_build"),
        "--onnxDir",
        str(onnx_root / "llm"),
        "--engineDir",
        str(engine_root / "llm"),
        "--maxBatchSize",
        str(max_batch_size),
        "--maxInputLen",
        str(max_input_len),
        "--maxKVCacheCapacity",
        str(max_kv_cache_capacity),
    ]
    if max_kv_pool_pages is not None:
        llm_command.extend(("--maxKVPoolPages", str(max_kv_pool_pages)))
    visual_command = [
        str(
            edge_root
            / "build"
            / "examples"
            / "multimodal"
            / "visual_build"
        ),
        "--onnxDir",
        str(onnx_root / "visual"),
        "--engineDir",
        str(engine_root),
        "--minImageTokens",
        str(min_image_tokens),
        "--maxImageTokens",
        str(max_image_tokens),
        "--maxImageTokensPerImage",
        str(max_image_tokens_per_image),
    ]
    return llm_command, visual_command


def require_paged_kv_build_path(audit: Mapping[str, Any]) -> None:
    """拒绝只有 ABI 符号、但未接通 builder/runtime 的 paged-KV 构建。"""
    summary = audit.get("summary")
    conclusion = summary.get("conclusion") if isinstance(summary, Mapping) else None
    if conclusion != "paged_kv_path_exposed_requires_runtime_validation":
        raise RuntimeError(
            "paged-KV source audit did not expose a complete build/runtime path; "
            f"conclusion={conclusion}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-llm-root", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--onnx-root", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument(
        "--component",
        choices=("both", "llm", "visual"),
        default="both",
    )
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-kv-cache-capacity", type=int, default=2048)
    parser.add_argument(
        "--max-kv-pool-pages",
        type=int,
        help=(
            "paged KV pool 的物理 page 数；省略时保持 Edge-LLM 默认路径，"
            "仅用于独立 paged-KV candidate"
        ),
    )
    parser.add_argument(
        "--verify-paged-kv-source",
        action="store_true",
        help="构建前审计固定 Edge-LLM checkout 的 paged-KV 源码入口",
    )
    parser.add_argument("--workspace-limit-mib", type=int)
    parser.add_argument(
        "--builder-optimization-level",
        type=int,
        choices=range(0, 6),
    )
    parser.add_argument(
        "--profiling-verbosity",
        choices=("none", "layer_names_only", "detailed"),
        help="写入 TensorRT engine inspector 所需的层级/详细 profile 信息",
    )
    parser.add_argument(
        "--timing-cache",
        type=Path,
        help="TensorRT ITimingCache 文件；仅影响构建阶段 tactic 搜索，不代表推理加速",
    )
    parser.add_argument(
        "--timing-cache-binding",
        type=Path,
        help=(
            "与 timing cache 绑定的 GPU/CUDA/TensorRT/BuilderConfig JSON；"
            "使用 timing cache 时必填"
        ),
    )
    parser.add_argument(
        "--reduced-vocab-dir",
        type=Path,
        help="已校验的 reduced-vocabulary 目录；构建前核对 map 与 ONNX export 的 SHA-256",
    )
    parser.add_argument(
        "--reduced-vocab-required-token-id",
        action="append",
        type=int,
        default=[],
        help="要求出现在 reduced-vocabulary map 中的 token id；可重复传入",
    )
    parser.add_argument(
        "--reduced-vocab-coverage-report",
        type=Path,
        help="独立 holdout token coverage report；必须与 source map SHA-256 一致且 valid=true",
    )
    parser.add_argument(
        "--quantization-provenance",
        type=Path,
        help="量化阶段 provenance；用于在 engine build 前核对候选 checkpoint 的精度范围",
    )
    parser.add_argument(
        "--expected-lm-head-precision",
        choices=("fp16", "int4_awq"),
        help="与 --quantization-provenance 配套的 lm_head 精度声明",
    )
    parser.add_argument(
        "--build-report",
        type=Path,
        help="记录各组件构建耗时、命令和 builder 环境变量的 JSON",
    )
    parser.add_argument("--enable-weight-streaming", action="store_true")
    parser.add_argument("--min-image-tokens", type=int, default=8)
    parser.add_argument("--max-image-tokens", type=int, default=2048)
    parser.add_argument("--max-image-tokens-per-image", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.workspace_limit_mib is not None and args.workspace_limit_mib <= 0:
        parser.error("--workspace-limit-mib 必须大于 0")
    if args.max_kv_pool_pages is not None and args.max_kv_pool_pages <= 0:
        parser.error("--max-kv-pool-pages 必须大于 0")
    if args.max_kv_pool_pages is not None and args.component == "visual":
        parser.error("--max-kv-pool-pages requires an llm component")
    if args.verify_paged_kv_source and args.max_kv_pool_pages is None:
        parser.error("--verify-paged-kv-source requires --max-kv-pool-pages")
    if args.timing_cache is not None:
        args.timing_cache = args.timing_cache.resolve()
        if args.timing_cache.exists() and not args.timing_cache.is_file():
            parser.error(f"--timing-cache 必须指向文件：{args.timing_cache}")
        args.timing_cache.parent.mkdir(parents=True, exist_ok=True)
    if args.timing_cache is not None and args.timing_cache_binding is None:
        parser.error("--timing-cache requires --timing-cache-binding")
    if args.timing_cache_binding is not None:
        args.timing_cache_binding = args.timing_cache_binding.resolve()
        if args.timing_cache is None:
            parser.error("--timing-cache-binding requires --timing-cache")
    if args.reduced_vocab_dir is not None:
        args.reduced_vocab_dir = args.reduced_vocab_dir.resolve()
        if args.component == "visual":
            parser.error("--reduced-vocab-dir requires an llm component")
    if args.reduced_vocab_required_token_id and args.reduced_vocab_dir is None:
        parser.error(
            "--reduced-vocab-required-token-id requires --reduced-vocab-dir"
        )
    if args.reduced_vocab_coverage_report is not None and args.reduced_vocab_dir is None:
        parser.error("--reduced-vocab-coverage-report requires --reduced-vocab-dir")
    if args.reduced_vocab_dir is not None and args.reduced_vocab_coverage_report is None:
        parser.error("--reduced-vocab-dir requires --reduced-vocab-coverage-report")
    if args.reduced_vocab_coverage_report is not None:
        args.reduced_vocab_coverage_report = args.reduced_vocab_coverage_report.resolve()
    if args.expected_lm_head_precision is not None and args.quantization_provenance is None:
        parser.error("--expected-lm-head-precision requires --quantization-provenance")
    if args.quantization_provenance is not None:
        args.quantization_provenance = args.quantization_provenance.resolve()
        if args.expected_lm_head_precision is None:
            parser.error("--quantization-provenance requires --expected-lm-head-precision")

    edge_root = Path(args.edge_llm_root).resolve()
    onnx_root = Path(args.onnx_root).resolve()
    engine_root = Path(args.engine_root).resolve()
    llm_builder = edge_root / "build" / "examples" / "llm" / "llm_build"
    visual_builder = (
        edge_root
        / "build"
        / "examples"
        / "multimodal"
        / "visual_build"
    )
    plugin_path = edge_root / "build" / "libNvInfer_edgellm_plugin.so"
    required_paths = [plugin_path]
    if args.component in {"both", "llm"}:
        required_paths.extend(
            (llm_builder, onnx_root / "llm" / "model.onnx")
        )
    if args.component in {"both", "visual"}:
        required_paths.extend(
            (visual_builder, onnx_root / "visual" / "model.onnx")
        )
    reduced_vocab_report = None
    if args.reduced_vocab_dir is not None:
        reduced_vocab_report = _validate_reduced_vocab_for_build(
            args.reduced_vocab_dir,
            onnx_root / "llm",
            required_token_ids=tuple(args.reduced_vocab_required_token_id),
            lm_head_precision=args.expected_lm_head_precision,
        )
        reduced_vocab_report["coverage_report"] = _validate_reduced_vocab_coverage_report(
            args.reduced_vocab_coverage_report,
            source_map_sha256=reduced_vocab_report["map"]["sha256"],
        )
    quantization_provenance_report = None
    if args.quantization_provenance is not None:
        quantization_provenance_report = _validate_quantization_provenance(
            args.quantization_provenance,
            expected_edge_llm_revision=args.expected_revision,
            expected_lm_head_precision=args.expected_lm_head_precision,
        )
    paged_kv_source_audit = None
    if args.verify_paged_kv_source:
        from scripts.audit_edgellm_kv_cache import audit_kv_cache_source

        paged_kv_source_audit = audit_kv_cache_source(
            edge_llm_root=edge_root,
            expected_revision=args.expected_revision,
        )
        require_paged_kv_build_path(paged_kv_source_audit)
    timing_cache_binding_report = None
    if args.timing_cache_binding is not None:
        timing_cache_binding_report = _validate_timing_cache_binding(
            args.timing_cache_binding,
            expected_builder_config={
                "component": args.component,
                "max_batch_size": args.max_batch_size,
                "max_input_len": args.max_input_len,
                "max_kv_cache_capacity": args.max_kv_cache_capacity,
                "max_kv_pool_pages": args.max_kv_pool_pages,
                "workspace_limit_mib": args.workspace_limit_mib,
                "builder_optimization_level": args.builder_optimization_level,
            },
        )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"缺少 engine 构建输入：{missing_paths}")

    revision_result = subprocess.run(
        ["git", "-C", str(edge_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_revision = revision_result.stdout.strip()
    if actual_revision != args.expected_revision:
        raise RuntimeError(
            "TensorRT Edge-LLM revision 不匹配："
            f"expected={args.expected_revision}, actual={actual_revision}"
        )

    llm_engine_dir = engine_root / "llm"
    visual_engine_dir = engine_root / "visual"
    expected_outputs: list[Path] = []
    if args.component in {"both", "llm"}:
        expected_outputs.append(llm_engine_dir / "llm.engine")
    if args.component in {"both", "visual"}:
        expected_outputs.append(visual_engine_dir / "visual.engine")
    existing_outputs = [str(path) for path in expected_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"拒绝覆盖已有 engine：{existing_outputs}")
    if args.component in {"both", "llm"}:
        llm_engine_dir.mkdir(parents=True, exist_ok=True)
    if args.component in {"both", "visual"}:
        visual_engine_dir.mkdir(parents=True, exist_ok=True)

    llm_command, visual_command = build_commands(
        edge_root=edge_root,
        onnx_root=onnx_root,
        engine_root=engine_root,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
        max_image_tokens_per_image=args.max_image_tokens_per_image,
        max_kv_pool_pages=args.max_kv_pool_pages,
    )
    child_environment = os.environ.copy()
    for tuning_variable in (
        "EDGELLM_WORKSPACE_LIMIT_MIB",
        "EDGELLM_BUILDER_OPT_LEVEL",
        "EDGELLM_PROFILING_VERBOSITY",
        "EDGELLM_ENABLE_WEIGHT_STREAMING",
        "EDGELLM_TIMING_CACHE_PATH",
    ):
        child_environment.pop(tuning_variable, None)
    child_environment["EDGELLM_PLUGIN_PATH"] = str(plugin_path)
    if args.workspace_limit_mib is not None:
        child_environment["EDGELLM_WORKSPACE_LIMIT_MIB"] = str(
            args.workspace_limit_mib
        )
    if args.builder_optimization_level is not None:
        child_environment["EDGELLM_BUILDER_OPT_LEVEL"] = str(
            args.builder_optimization_level
        )
    if args.profiling_verbosity is not None:
        child_environment["EDGELLM_PROFILING_VERBOSITY"] = args.profiling_verbosity
    if args.enable_weight_streaming:
        child_environment["EDGELLM_ENABLE_WEIGHT_STREAMING"] = "1"
    if args.timing_cache is not None:
        child_environment["EDGELLM_TIMING_CACHE_PATH"] = str(args.timing_cache)
    build_report = _new_build_report(
        args=args,
        edge_root=edge_root,
        onnx_root=onnx_root,
        engine_root=engine_root,
        llm_command=llm_command,
        visual_command=visual_command,
        environment=child_environment,
        actual_revision=actual_revision,
        expected_outputs=expected_outputs,
        reduced_vocab_report=reduced_vocab_report,
        quantization_provenance_report=quantization_provenance_report,
        timing_cache_binding_report=timing_cache_binding_report,
        paged_kv_source_audit=paged_kv_source_audit,
    )
    try:
        if args.component in {"both", "llm"}:
            _run_build_step(
                component="llm",
                command=llm_command,
                environment=child_environment,
                report=build_report,
            )
        if args.component in {"both", "visual"}:
            _run_build_step(
                component="visual",
                command=visual_command,
                environment=child_environment,
                report=build_report,
            )
        build_report["status"] = "succeeded"
        return 0
    except Exception:
        build_report["status"] = "failed"
        raise
    finally:
        build_report["outputs"] = _inspect_outputs(expected_outputs)
        build_report["timing_cache"] = _inspect_optional_path(args.timing_cache)
        build_report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if args.build_report is not None:
            _write_build_report(args.build_report, build_report)


def _new_build_report(
    *,
    args: argparse.Namespace,
    edge_root: Path,
    onnx_root: Path,
    engine_root: Path,
    llm_command: list[str],
    visual_command: list[str],
    environment: dict[str, str],
    actual_revision: str,
    expected_outputs: list[Path],
    reduced_vocab_report: dict[str, Any] | None = None,
    quantization_provenance_report: dict[str, Any] | None = None,
    timing_cache_binding_report: dict[str, Any] | None = None,
    paged_kv_source_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tracked_environment = {
        key: environment[key]
        for key in (
            "EDGELLM_PLUGIN_PATH",
            "EDGELLM_WORKSPACE_LIMIT_MIB",
            "EDGELLM_BUILDER_OPT_LEVEL",
            "EDGELLM_PROFILING_VERBOSITY",
            "EDGELLM_ENABLE_WEIGHT_STREAMING",
            "EDGELLM_TIMING_CACHE_PATH",
        )
        if key in environment
    }
    return {
        "schema_version": "parksight_tensorrt_engine_build_v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "edge_llm_root": str(edge_root),
            "edge_llm_revision": actual_revision,
            "onnx_root": str(onnx_root),
            "engine_root": str(engine_root),
            "component": args.component,
            "max_batch_size": args.max_batch_size,
            "max_input_len": args.max_input_len,
            "max_kv_cache_capacity": args.max_kv_cache_capacity,
            "max_kv_pool_pages": getattr(args, "max_kv_pool_pages", None),
            "min_image_tokens": args.min_image_tokens,
            "max_image_tokens": args.max_image_tokens,
            "max_image_tokens_per_image": args.max_image_tokens_per_image,
            "workspace_limit_mib": args.workspace_limit_mib,
            "builder_optimization_level": args.builder_optimization_level,
            "profiling_verbosity": args.profiling_verbosity or "none",
            "enable_weight_streaming": args.enable_weight_streaming,
            "timing_cache": (
                str(args.timing_cache)
                if getattr(args, "timing_cache", None) is not None
                else None
            ),
            "timing_cache_binding": (
                str(args.timing_cache_binding)
                if getattr(args, "timing_cache_binding", None) is not None
                else None
            ),
            "reduced_vocab_dir": (
                str(args.reduced_vocab_dir)
                if getattr(args, "reduced_vocab_dir", None) is not None
                else None
            ),
            "reduced_vocab_required_token_ids": list(
                getattr(args, "reduced_vocab_required_token_id", [])
            ),
            "reduced_vocab_coverage_report": (
                str(args.reduced_vocab_coverage_report)
                if getattr(args, "reduced_vocab_coverage_report", None) is not None
                else None
            ),
            "quantization_provenance": (
                str(args.quantization_provenance)
                if getattr(args, "quantization_provenance", None) is not None
                else None
            ),
            "expected_lm_head_precision": getattr(
                args, "expected_lm_head_precision", None
            ),
            "verify_paged_kv_source": getattr(args, "verify_paged_kv_source", False),
        },
        "builder_environment": tracked_environment,
        "commands": {
            "llm": llm_command,
            "visual": visual_command,
        },
        "steps": [],
        "outputs": _inspect_outputs(expected_outputs),
        "timing_cache": _inspect_optional_path(getattr(args, "timing_cache", None)),
        "timing_cache_binding": timing_cache_binding_report,
        "reduced_vocabulary": reduced_vocab_report,
        "quantization_provenance": quantization_provenance_report,
        "paged_kv_source_audit": paged_kv_source_audit,
    }


def _run_build_step(
    *,
    component: str,
    command: list[str],
    environment: dict[str, str],
    report: dict[str, Any],
) -> None:
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    try:
        subprocess.run(command, check=True, env=environment)
    except subprocess.CalledProcessError as error:
        report["steps"].append(
            {
                "component": component,
                "status": "failed",
                "returncode": error.returncode,
                "started_at_utc": started_at.isoformat(),
                "duration_ms": (time.perf_counter() - started_clock) * 1000.0,
            }
        )
        raise
    report["steps"].append(
        {
            "component": component,
            "status": "succeeded",
            "returncode": 0,
            "started_at_utc": started_at.isoformat(),
            "duration_ms": (time.perf_counter() - started_clock) * 1000.0,
        }
    )


def _write_build_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _inspect_outputs(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in paths
    ]


def _inspect_optional_path(path: Path | None) -> dict[str, Any] | None:
    """记录 timing cache 的存在性和 digest，但不把它当作 engine 输出。"""
    if path is None:
        return None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _validate_timing_cache_binding(
    path: Path,
    *,
    expected_builder_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require explicit device and BuilderConfig identity for an ITimingCache.

    TensorRT performs its own compatibility checks when loading an
    ``ITimingCache``.  This sidecar is an experiment-level guard: it makes the
    GPU, CUDA/TensorRT versions and the command-line BuilderConfig auditable
    before ``llm_build`` starts, including when the cache file is being
    created for the first time.
    """
    if not path.is_file():
        raise ValueError(f"timing-cache binding must be an existing JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read timing-cache binding: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("timing-cache binding must be a JSON object")
    if payload.get("schema_version") != "parksight_tensorrt_timing_cache_binding_v1":
        raise ValueError("timing-cache binding has unsupported schema_version")
    device = payload.get("device")
    if not isinstance(device, dict):
        raise ValueError("timing-cache binding must contain a device object")
    for field in ("gpu_name", "cuda_version", "tensorrt_version"):
        value = device.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"timing-cache binding device.{field} must be non-empty")
    builder_config = payload.get("builder_config")
    if not isinstance(builder_config, dict):
        raise ValueError("timing-cache binding must contain a builder_config object")
    mismatches = {
        field: (expected, builder_config.get(field))
        for field, expected in expected_builder_config.items()
        if builder_config.get(field) != expected
    }
    if mismatches:
        raise ValueError(
            "timing-cache BuilderConfig mismatch: "
            f"{mismatches}"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "schema_version": payload["schema_version"],
        "device": dict(device),
        "builder_config": dict(builder_config),
    }


def _validate_reduced_vocab_for_build(
    source_dir: Path,
    onnx_llm_dir: Path,
    *,
    required_token_ids: tuple[int, ...] = (),
    lm_head_precision: str | None = None,
) -> dict[str, Any]:
    """Validate a reduced-vocab source and its copied ONNX artifacts."""
    map_path = source_dir / "vocab_map.safetensors"
    metadata_path = source_dir / "reduced_vocab.json"
    selection_report_path = source_dir / "selection_report.json"
    onnx_config_path = onnx_llm_dir / "config.json"
    onnx_map_path = onnx_llm_dir / "vocab_map.safetensors"
    onnx_metadata_path = onnx_llm_dir / "reduced_vocab.json"
    required_paths = (
        map_path,
        metadata_path,
        selection_report_path,
        onnx_config_path,
        onnx_map_path,
        onnx_metadata_path,
    )
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "reduced-vocabulary build inputs are incomplete: "
            f"{missing_paths}"
        )

    try:
        from scripts.validate_reduced_vocab import validate_mapping
    except ModuleNotFoundError:
        # Support direct execution when the scripts directory, rather than the
        # repository root, is the first import path.
        from validate_reduced_vocab import validate_mapping

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_vocab_size = metadata.get("vocab_size") if isinstance(metadata, dict) else None
    if (
        isinstance(original_vocab_size, bool)
        or not isinstance(original_vocab_size, int)
        or original_vocab_size <= 0
    ):
        raise ValueError("reduced-vocabulary metadata must declare a positive vocab_size")
    validation = validate_mapping(
        map_path=map_path,
        metadata_path=metadata_path,
        original_vocab_size=original_vocab_size,
        required_token_ids=required_token_ids,
        packed_int4_lm_head=lm_head_precision == "int4_awq",
    )
    if not validation["valid"]:
        raise ValueError(
            "invalid reduced-vocabulary map: "
            f"{validation['reasons']}"
        )

    source_map_sha256 = sha256_file(map_path)
    source_metadata_sha256 = sha256_file(metadata_path)
    reduced_vocab_size = validation["metadata"]["reduced_vocab_size"]
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    if not isinstance(selection_report, dict):
        raise ValueError("reduced-vocabulary selection report must be a JSON object")
    report_map = selection_report.get("map")
    if not isinstance(report_map, dict):
        raise ValueError("reduced-vocabulary selection report must contain map metadata")
    if report_map.get("sha256") != source_map_sha256:
        raise ValueError(
            "reduced-vocabulary selection report does not match source map SHA-256: "
            f"report={report_map.get('sha256')}, source={source_map_sha256}"
        )
    if report_map.get("count") != reduced_vocab_size:
        raise ValueError(
            "reduced-vocabulary selection report count does not match map metadata: "
            f"report={report_map.get('count')}, metadata={reduced_vocab_size}"
        )
    onnx_map_sha256 = sha256_file(onnx_map_path)
    onnx_metadata_sha256 = sha256_file(onnx_metadata_path)
    if source_map_sha256 != onnx_map_sha256:
        raise ValueError(
            "ONNX reduced-vocabulary map does not match source map SHA-256: "
            f"source={source_map_sha256}, onnx={onnx_map_sha256}"
        )
    if source_metadata_sha256 != onnx_metadata_sha256:
        raise ValueError(
            "ONNX reduced-vocabulary metadata does not match source metadata SHA-256: "
            f"source={source_metadata_sha256}, onnx={onnx_metadata_sha256}"
        )
    onnx_config = json.loads(onnx_config_path.read_text(encoding="utf-8"))
    if not isinstance(onnx_config, dict):
        raise ValueError("ONNX LLM config must be a JSON object")
    if onnx_config.get("reduced_vocab_size") != reduced_vocab_size:
        raise ValueError(
            "ONNX LLM config reduced_vocab_size does not match map metadata: "
            f"config={onnx_config.get('reduced_vocab_size')}, "
            f"metadata={reduced_vocab_size}"
        )
    return {
        "source_dir": str(source_dir),
        "original_vocab_size": original_vocab_size,
        "reduced_vocab_size": reduced_vocab_size,
        "required_token_ids": list(required_token_ids),
        "lm_head_precision": lm_head_precision,
        "packed_int4_group_size": 128 if lm_head_precision == "int4_awq" else None,
        "map": {
            "source_path": str(map_path),
            "onnx_path": str(onnx_map_path),
            "size_bytes": map_path.stat().st_size,
            "sha256": source_map_sha256,
        },
        "metadata": {
            "source_path": str(metadata_path),
            "onnx_path": str(onnx_metadata_path),
            "size_bytes": metadata_path.stat().st_size,
            "sha256": source_metadata_sha256,
        },
        "selection_report": {
            "path": str(selection_report_path),
            "size_bytes": selection_report_path.stat().st_size,
            "sha256": sha256_file(selection_report_path),
            "map_sha256_verified": True,
        },
        "onnx_config": {
            "path": str(onnx_config_path),
            "reduced_vocab_size": onnx_config["reduced_vocab_size"],
        },
    }


def _validate_quantization_provenance(
    path: Path,
    *,
    expected_edge_llm_revision: str,
    expected_lm_head_precision: str,
) -> dict[str, Any]:
    """Validate the quantization scope before an engine candidate is built."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read quantization provenance: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("quantization provenance must be a JSON object")
    if payload.get("status") != "succeeded":
        raise ValueError(
            "quantization provenance status must be succeeded: "
            f"{payload.get('status')!r}"
        )
    if payload.get("edge_llm_revision") != expected_edge_llm_revision:
        raise ValueError(
            "quantization provenance Edge-LLM revision mismatch: "
            f"expected={expected_edge_llm_revision}, "
            f"actual={payload.get('edge_llm_revision')}"
        )
    if payload.get("lm_head_precision") != expected_lm_head_precision:
        raise ValueError(
            "quantization provenance lm_head precision mismatch: "
            f"expected={expected_lm_head_precision}, "
            f"actual={payload.get('lm_head_precision')}"
        )
    if payload.get("quantization") != "int4_awq":
        raise ValueError(
            "quantization provenance must declare quantization=int4_awq"
        )
    workload_identity = payload.get("calibration_workload_identity")
    if not isinstance(workload_identity, str) or not workload_identity.strip():
        raise ValueError(
            "quantization provenance requires calibration_workload_identity"
        )
    calibration_rows = payload.get("calibration_rows")
    if (
        isinstance(calibration_rows, bool)
        or not isinstance(calibration_rows, int)
        or calibration_rows <= 0
    ):
        raise ValueError(
            "quantization provenance calibration_rows must be a positive integer"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": payload["status"],
        "edge_llm_revision": payload["edge_llm_revision"],
        "quantization": payload["quantization"],
        "lm_head_precision": payload["lm_head_precision"],
        "kv_cache_quantization": payload.get("kv_cache_quantization"),
        "calibration_rows": calibration_rows,
        "calibration_workload_identity": workload_identity,
    }


def _validate_reduced_vocab_coverage_report(
    path: Path,
    *,
    source_map_sha256: str,
) -> dict[str, Any]:
    """Require an independent, map-bound holdout coverage result."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read reduced-vocabulary coverage report: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("reduced-vocabulary coverage report must be a JSON object")
    if payload.get("schema_version") != "parksight_reduced_vocab_coverage_v1":
        raise ValueError("reduced-vocabulary coverage report has unsupported schema_version")
    if payload.get("valid") is not True:
        raise ValueError("reduced-vocabulary coverage report must have valid=true")
    if payload.get("samples_with_missing_tokens") != 0:
        raise ValueError("reduced-vocabulary coverage report contains missing tokens")
    report_map = payload.get("map")
    if not isinstance(report_map, dict) or report_map.get("sha256") != source_map_sha256:
        raise ValueError(
            "reduced-vocabulary coverage report does not match source map SHA-256"
        )
    token_coverage = payload.get("token_coverage_fraction")
    sample_coverage = payload.get("sample_coverage_fraction")
    if token_coverage != 1.0 or sample_coverage != 1.0:
        raise ValueError(
            "reduced-vocabulary coverage report must have complete token and sample coverage"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "valid": True,
        "reference_sample_count": payload.get("reference_sample_count"),
        "total_reference_tokens": payload.get("total_reference_tokens"),
        "token_coverage_fraction": token_coverage,
        "sample_coverage_fraction": sample_coverage,
        "map_sha256_verified": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
