"""使用固定版本的 TensorRT Edge-LLM 构建 VLM 的两个 engine。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import signal
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
    llm_builder_path: Path | None = None,
    visual_builder_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    """生成先 LLM、后视觉编码器的两条确定性构建命令。"""
    llm_command = [
        str(llm_builder_path or edge_root / "build" / "examples" / "llm" / "llm_build"),
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
            visual_builder_path or edge_root
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


def _validate_staged_paged_graph(
    path: Path, *, attention_count: int, max_batch_size: int,
    max_pages_per_sequence: int,
    max_kv_pool_pages: int,
) -> dict[str, int]:
    """Read only ONNX metadata; external INT4 weights stay on disk."""
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("paged ONNX build preflight requires the onnx package") from error
    if __package__:
        from scripts.experiments.prepare_edgellm_paged_onnx import validate_paged_graph_contract
    else:
        from experiments.prepare_edgellm_paged_onnx import validate_paged_graph_contract

    model = onnx.load(str(path), load_external_data=False)
    return validate_paged_graph_contract(
        model,
        attention_count=attention_count,
        max_batch_size=max_batch_size,
        max_pages_per_sequence=max_pages_per_sequence,
        max_kv_pool_pages=max_kv_pool_pages,
        int32_type=onnx.TensorProto.INT32,
    )


def require_paged_onnx_stage(
    onnx_root: Path, *, max_batch_size: int, max_kv_cache_capacity: int,
    max_kv_pool_pages: int,
) -> dict[str, Any]:
    """Reject dense or changed ONNX graphs before starting a paged engine build."""
    provenance = onnx_root / "paged_onnx_provenance.json"
    try:
        report = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"paged ONNX stage provenance is missing or invalid: {provenance}") from error
    expected_pages_per_sequence = (max_kv_cache_capacity + 127) // 128
    if max_kv_pool_pages < max_batch_size * expected_pages_per_sequence:
        raise ValueError("paged KV pool is smaller than the batch worst-case capacity")
    expected = {
        "schema_version": "parksight_paged_onnx_stage_v1",
        "tokens_per_page": 128,
        "max_pages_per_sequence": expected_pages_per_sequence,
        "max_kv_pool_pages": max_kv_pool_pages,
        "max_batch_size": max_batch_size,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("paged ONNX stage provenance does not match the build configuration")
    files = report.get("staged_files_sha256")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("paged ONNX stage provenance has no file hashes")
    if not {"llm/model.onnx", "llm/model.onnx.data"}.issubset(files):
        raise ValueError("paged ONNX stage provenance omits the graph or external weights")
    for name, expected_digest in files.items():
        if not isinstance(name, str) or not isinstance(expected_digest, str):
            raise ValueError("paged ONNX stage file identity is invalid")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe paged ONNX stage file path: {name}")
    staged_paths = list(onnx_root.rglob("*"))
    symlinks = [str(path) for path in staged_paths if path.is_symlink()]
    if symlinks:
        raise ValueError(f"paged ONNX stage contains symlinks: {symlinks}")
    actual_files = {
        str(path.relative_to(onnx_root)) for path in staged_paths
        if path.is_file() and path != provenance
    }
    declared_files = set(files)
    if actual_files != declared_files:
        raise ValueError(
            "paged ONNX stage file manifest differs from actual files: "
            f"unlisted={sorted(actual_files - declared_files)}, "
            f"missing={sorted(declared_files - actual_files)}"
        )
    for name, expected_digest in files.items():
        path = onnx_root / name
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise ValueError(f"paged ONNX stage file changed or is missing: {path}")
    attention_count = report.get("attention_plugin_count")
    if type(attention_count) is not int or attention_count <= 0:
        raise ValueError("paged ONNX stage has an invalid AttentionPlugin count")
    if __package__:
        from scripts.experiments.prepare_edgellm_paged_onnx import expected_attention_count
    else:
        from experiments.prepare_edgellm_paged_onnx import expected_attention_count
    layer_count = expected_attention_count(onnx_root / "llm" / "config.json")
    if attention_count != layer_count:
        raise ValueError(
            f"paged ONNX AttentionPlugin count {attention_count} differs from "
            f"num_hidden_layers {layer_count}"
        )
    graph_validation = _validate_staged_paged_graph(
        onnx_root / "llm" / "model.onnx",
        attention_count=attention_count,
        max_batch_size=max_batch_size,
        max_pages_per_sequence=expected_pages_per_sequence,
        max_kv_pool_pages=max_kv_pool_pages,
    )
    return {
        "provenance_path": str(provenance),
        "provenance_sha256": sha256_file(provenance),
        "attention_plugin_count": attention_count,
        "graph_validation": graph_validation,
        **expected,
    }


def validate_paged_engine_config(
    path: Path, *, max_batch_size: int, max_input_len: int,
    max_kv_cache_capacity: int, max_kv_pool_pages: int,
) -> dict[str, Any]:
    """Confirm the serialized engine config actually declares the paged layout."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"paged engine config is missing or invalid: {path}") from error
    builder = payload.get("builder_config") if isinstance(payload, Mapping) else None
    if not isinstance(builder, Mapping):
        raise ValueError(f"paged engine config has no builder_config: {path}")
    expected = {
        "kv_cache_layout": "paged",
        "tokens_per_page": 128,
        "max_batch_size": max_batch_size,
        "max_input_len": max_input_len,
        "max_kv_cache_capacity": max_kv_cache_capacity,
        "max_kv_pool_pages": max_kv_pool_pages,
    }
    mismatches = {
        key: {"expected": value, "actual": builder.get(key)}
        for key, value in expected.items()
        if builder.get(key) != value or type(builder.get(key)) is not type(value)
    }
    if mismatches:
        raise ValueError(f"paged engine config does not match requested layout: {mismatches}")
    return {"path": str(path), "sha256": sha256_file(path), "builder_config": expected}


def require_builder_controls(
    builder: Path, *, component: str, profiling_verbosity: str | None,
    timing_cache: Path | None,
) -> str | None:
    """Find a compiled profiling control and reject unavailable requested controls.

    Presence is only a capability hint: effective settings still need engine/cache
    inspection. The LLM builder supports a native --profilingDetailed flag even
    when the optional environment-variable patch has not been compiled.
    """
    required = [b"EDGELLM_PROFILING_VERBOSITY"] if profiling_verbosity else []
    if component == "llm" and profiling_verbosity == "detailed":
        required.append(b"--profilingDetailed")
    if timing_cache is not None:
        required.append(b"EDGELLM_TIMING_CACHE_PATH")
    if not required:
        return None
    found = set()
    tail = b""
    with builder.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = tail + chunk
            found.update(key for key in required if key in data)
            tail = data[-max(map(len, required)):]
    profiling_mode = None
    if profiling_verbosity is not None:
        if component == "llm" and profiling_verbosity == "detailed" and b"--profilingDetailed" in found:
            profiling_mode = "native_cli"
        elif b"EDGELLM_PROFILING_VERBOSITY" in found:
            profiling_mode = "environment"
        else:
            raise RuntimeError(
                f"builder does not advertise profiling control for {profiling_verbosity}: {builder}"
            )
    missing = [key.decode("ascii") for key in required if key == b"EDGELLM_TIMING_CACHE_PATH" and key not in found]
    if missing:
        raise RuntimeError(
            f"builder does not advertise requested controls {missing}: {builder}; "
            "use a builder implementing these controls, or omit the unverified options"
        )
    return profiling_mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-llm-root", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--onnx-root", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--llm-builder", type=Path, help="独立编译的 llm_build 路径")
    parser.add_argument("--visual-builder", type=Path, help="独立编译的 visual_build 路径")
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
    parser.add_argument(
        "--expected-source-fingerprint",
        help="要求 paged-KV 源码内容指纹与预检结果一致，防止固定 HEAD 下的本地补丁漂移",
    )
    parser.add_argument("--expected-llm-builder-sha256", help="预检时 LLM builder 二进制的 SHA-256")
    parser.add_argument("--expected-visual-builder-sha256", help="预检时 visual builder 二进制的 SHA-256")
    parser.add_argument("--expected-plugin-sha256", help="预检时 Edge-LLM plugin 二进制的 SHA-256")
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
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只验证源码、ONNX、builder 和 plugin；不创建 engine 目录或执行构建",
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
    if args.max_kv_pool_pages is not None and not args.verify_paged_kv_source:
        parser.error("--max-kv-pool-pages requires --verify-paged-kv-source")
    if args.verify_paged_kv_source and args.max_kv_pool_pages is None:
        parser.error("--verify-paged-kv-source requires --max-kv-pool-pages")
    if args.expected_source_fingerprint is not None:
        if not args.verify_paged_kv_source:
            parser.error("--expected-source-fingerprint requires --verify-paged-kv-source")
        if re.fullmatch(r"[0-9a-f]{64}", args.expected_source_fingerprint) is None:
            parser.error("--expected-source-fingerprint must be a lowercase SHA-256 digest")
    for name in ("expected_llm_builder_sha256", "expected_visual_builder_sha256", "expected_plugin_sha256"):
        value = getattr(args, name)
        if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA-256 digest")
    if args.expected_llm_builder_sha256 is not None and args.component == "visual":
        parser.error("--expected-llm-builder-sha256 requires an llm component")
    if args.expected_visual_builder_sha256 is not None and args.component == "llm":
        parser.error("--expected-visual-builder-sha256 requires a visual component")
    if args.timing_cache is not None:
        args.timing_cache = args.timing_cache.resolve()
        if args.timing_cache.exists() and not args.timing_cache.is_file():
            parser.error(f"--timing-cache 必须指向文件：{args.timing_cache}")
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
    llm_builder = (args.llm_builder or edge_root / "build/examples/llm/llm_build").resolve()
    visual_builder = (args.visual_builder or edge_root / "build/examples/multimodal/visual_build").resolve()
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
        if __package__:
            from scripts.diagnostics.audit_edgellm_kv_cache import audit_kv_cache_source
        else:
            from diagnostics.audit_edgellm_kv_cache import audit_kv_cache_source

        paged_kv_source_audit = audit_kv_cache_source(
            edge_llm_root=edge_root,
            expected_revision=args.expected_revision,
        )
        require_paged_kv_build_path(paged_kv_source_audit)
        if (args.expected_source_fingerprint is not None
                and paged_kv_source_audit["source_fingerprint"]["sha256"]
                != args.expected_source_fingerprint):
            raise ValueError(
                "Edge-LLM source fingerprint changed since preflight: "
                f"expected={args.expected_source_fingerprint}, "
                f"actual={paged_kv_source_audit['source_fingerprint']['sha256']}"
            )
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
    paged_onnx_stage_report = None
    if args.max_kv_pool_pages is not None:
        paged_onnx_stage_report = require_paged_onnx_stage(
            onnx_root,
            max_batch_size=args.max_batch_size,
            max_kv_cache_capacity=args.max_kv_cache_capacity,
            max_kv_pool_pages=args.max_kv_pool_pages,
        )

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
    selected_dirs = []
    if args.component in {"both", "llm"}:
        selected_dirs.append(llm_engine_dir)
    if args.component in {"both", "visual"}:
        selected_dirs.append(visual_engine_dir)
    occupied_dirs = [
        str(directory) for directory in selected_dirs
        if (directory.exists() or directory.is_symlink())
        and (not directory.is_dir() or any(directory.iterdir()))
    ]
    if occupied_dirs:
        raise FileExistsError(f"拒绝复用非空 engine 输出目录：{occupied_dirs}")
    if args.build_report is not None and args.build_report.exists():
        raise FileExistsError(f"拒绝覆盖已有 build report：{args.build_report}")
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
        llm_builder_path=llm_builder,
        visual_builder_path=visual_builder,
    )
    child_environment = os.environ.copy()
    selected_builders = []
    if args.component in {"both", "llm"}:
        selected_builders.append(("llm", llm_builder))
    if args.component in {"both", "visual"}:
        selected_builders.append(("visual", visual_builder))
    profiling_modes = {
        component: require_builder_controls(
            builder, component=component,
            profiling_verbosity=args.profiling_verbosity,
            timing_cache=args.timing_cache,
        )
        for component, builder in selected_builders
    }
    if profiling_modes.get("llm") == "native_cli":
        llm_command.append("--profilingDetailed")
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
    if "environment" in profiling_modes.values():
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
    build_report["paged_onnx_stage"] = paged_onnx_stage_report
    build_report["profiling_control_modes"] = profiling_modes
    build_report["builder_binaries"] = {
        component: {"path": str(builder), "sha256": sha256_file(builder)}
        for component, builder in selected_builders
    }
    build_report["plugin_binary"] = {
        "path": str(plugin_path),
        "sha256": sha256_file(plugin_path),
    }
    expected_binary_hashes = {
        "llm": args.expected_llm_builder_sha256,
        "visual": args.expected_visual_builder_sha256,
    }
    for component, identity in build_report["builder_binaries"].items():
        expected_hash = expected_binary_hashes[component]
        if expected_hash is not None and identity["sha256"] != expected_hash:
            raise ValueError(f"{component} builder binary changed since preflight")
    if (args.expected_plugin_sha256 is not None
            and build_report["plugin_binary"]["sha256"] != args.expected_plugin_sha256):
        raise ValueError("Edge-LLM plugin binary changed since preflight")
    build_report["paged_engine_config"] = None
    if args.check_only:
        print(json.dumps({
            "status": "ready_for_build",
            "configuration": build_report["configuration"],
            "paged_onnx_stage": paged_onnx_stage_report,
            "paged_kv_source_audit": (
                {
                    "summary": paged_kv_source_audit["summary"],
                    "source_fingerprint": paged_kv_source_audit["source_fingerprint"],
                }
                if paged_kv_source_audit is not None else None
            ),
            "profiling_control_modes": profiling_modes,
            "builder_binaries": build_report["builder_binaries"],
            "plugin_binary": build_report["plugin_binary"],
            "expected_outputs": [str(path) for path in expected_outputs],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.build_report is not None:
        args.build_report.parent.mkdir(parents=True, exist_ok=True)
        with args.build_report.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(build_report, ensure_ascii=False, indent=2) + "\n")
    try:
        if args.timing_cache is not None:
            args.timing_cache.parent.mkdir(parents=True, exist_ok=True)
        if args.component in {"both", "llm"}:
            llm_engine_dir.mkdir(parents=True, exist_ok=True)
        if args.component in {"both", "visual"}:
            visual_engine_dir.mkdir(parents=True, exist_ok=True)
        if args.component in {"both", "llm"}:
            _run_build_step(
                component="llm",
                command=llm_command,
                environment=child_environment,
                report=build_report,
            )
            if args.max_kv_pool_pages is not None:
                build_report["paged_engine_config"] = validate_paged_engine_config(
                    llm_engine_dir / "config.json",
                    max_batch_size=args.max_batch_size,
                    max_input_len=args.max_input_len,
                    max_kv_cache_capacity=args.max_kv_cache_capacity,
                    max_kv_pool_pages=args.max_kv_pool_pages,
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
            "profiling_verbosity": args.profiling_verbosity,
            "profiling_verbosity_provenance": (
                "requested; effective engine setting requires inspection"
                if args.profiling_verbosity is not None
                else "not requested; effective engine setting requires inspection"
            ),
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
            "expected_source_fingerprint": getattr(args, "expected_source_fingerprint", None),
            "expected_llm_builder_sha256": getattr(args, "expected_llm_builder_sha256", None),
            "expected_visual_builder_sha256": getattr(args, "expected_visual_builder_sha256", None),
            "expected_plugin_sha256": getattr(args, "expected_plugin_sha256", None),
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
    process_pid = None
    try:
        process = subprocess.Popen(command, env=environment)
        process_pid = process.pid
        returncode = process.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
    except (subprocess.CalledProcessError, OSError) as error:
        returncode = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
        signal_number = -returncode if returncode is not None and returncode < 0 else None
        try:
            signal_name = signal.Signals(signal_number).name if signal_number is not None else None
        except ValueError:
            signal_name = "UNKNOWN"
        report["steps"].append(
            {
                "component": component,
                "status": "failed",
                "pid": process_pid,
                "returncode": returncode,
                "failure_category": (
                    "process_signal" if signal_number is not None
                    else ("launch_error" if process_pid is None else "process_wait_error") if isinstance(error, OSError)
                    else "builder_nonzero_exit"
                ),
                "termination_signal_number": signal_number,
                "termination_signal": signal_name,
                "error_type": type(error).__name__,
                "error": str(error),
                "oom_confirmed": None,
                "started_at_utc": started_at.isoformat(),
                "duration_ms": (time.perf_counter() - started_clock) * 1000.0,
            }
        )
        raise
    report["steps"].append(
        {
            "component": component,
            "status": "succeeded",
            "pid": process_pid,
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
        from scripts.diagnostics.validate_reduced_vocab import validate_mapping
    except ModuleNotFoundError:
        # Support direct execution when the scripts directory, rather than the
        # repository root, is the first import path.
        from diagnostics.validate_reduced_vocab import validate_mapping

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
