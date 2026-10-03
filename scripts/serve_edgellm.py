"""从预构建的 LLM/visual engine 启动 Edge-LLM HTTP 服务。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import sha256_file


def _resolve_engine_directories(
    *,
    engine_root: Path | None,
    llm_engine_root: Path | None,
    visual_engine_root: Path | None,
) -> tuple[Path, Path]:
    if engine_root is not None:
        if llm_engine_root is not None or visual_engine_root is not None:
            raise ValueError(
                "use either --engine-root or both --llm-engine-root and "
                "--visual-engine-root"
            )
        return engine_root / "llm", engine_root / "visual"
    if llm_engine_root is None or visual_engine_root is None:
        raise ValueError(
            "provide --engine-root or both --llm-engine-root and "
            "--visual-engine-root"
        )
    return llm_engine_root, visual_engine_root


def serve_prebuilt_engines(
    *,
    engine_root: Path | None = None,
    llm_engine_root: Path | None = None,
    visual_engine_root: Path | None = None,
    host: str,
    port: int,
    http_max_batch_size: int = 1,
    cuda_graph: str | None = None,
    pin_optimization_profiles: bool | None = None,
    profile_switch_timing: bool | None = None,
    cache_binding_state: bool | None = None,
    cache_registered_bindings: bool | None = None,
    skip_redundant_profile_switch: bool | None = None,
    skip_fallback_binding_scan: bool | None = None,
    log_xqa_selection: bool | None = None,
    fmha_force_granular_tiling: bool | None = None,
    greedy_argmax_block_size: int | None = None,
    greedy_argmax_impl: str | None = None,
    direct_device_token_embedding: bool | None = None,
    int4_gemm_stages: int | None = None,
    int4_gemv_n_per_block: int | None = None,
    int4_gemv_block_size: int | None = None,
) -> None:
    """加载预构建 engine，避免在 8GB Jetson 上隐式执行模型导出。"""
    configure_cuda_graph(cuda_graph)
    configure_profile_contexts(pin_optimization_profiles)
    configure_profile_switch_timing(profile_switch_timing)
    configure_binding_state_cache(cache_binding_state)
    configure_registered_binding_cache(cache_registered_bindings)
    configure_redundant_profile_switch(skip_redundant_profile_switch)
    configure_fallback_binding_scan(skip_fallback_binding_scan)
    configure_xqa_selection_logging(log_xqa_selection)
    configure_fmha_force_granular_tiling(fmha_force_granular_tiling)
    configure_greedy_argmax_block_size(greedy_argmax_block_size)
    configure_greedy_argmax_impl(greedy_argmax_impl)
    configure_direct_device_token_embedding(direct_device_token_embedding)
    configure_int4_gemm_stages(int4_gemm_stages)
    configure_int4_gemv_n_per_block(int4_gemv_n_per_block)
    configure_int4_gemv_block_size(int4_gemv_block_size)
    try:
        import uvicorn  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "无法导入 uvicorn；请在当前 Jetson Python 环境安装 uvicorn，"
            "并确认其 site-packages 位于 PYTHONPATH"
        ) from error
    try:
        from experimental.server import LLM
    except ImportError as error:
        raise RuntimeError(
            "无法导入 experimental.server；请把固定 commit 的 "
            "TensorRT Edge-LLM checkout 加入 PYTHONPATH，并构建 Python bindings"
        ) from error

    llm_root, visual_root = _resolve_engine_directories(
        engine_root=engine_root,
        llm_engine_root=llm_engine_root,
        visual_engine_root=visual_engine_root,
    )
    llm = LLM(
        engine_dir=str(llm_root),
        visual_engine_dir=str(visual_root),
        max_batch_size=http_max_batch_size,
    )
    llm.serve(host=host, port=port)


def deployment_readiness(
    *,
    engine_root: Path | None = None,
    llm_engine_root: Path | None = None,
    visual_engine_root: Path | None = None,
    edge_llm_root: Path | None,
    plugin_path: Path | None,
) -> dict[str, object]:
    """Check static deployment inputs without importing or loading the runtime."""
    try:
        llm_root, visual_root = _resolve_engine_directories(
            engine_root=engine_root,
            llm_engine_root=llm_engine_root,
            visual_engine_root=visual_engine_root,
        )
    except ValueError as error:
        return {"ready": False, "configuration_error": str(error)}
    llm_root = llm_root.resolve()
    visual_root = visual_root.resolve()
    required_engines = (llm_root / "llm.engine", visual_root / "visual.engine")
    missing_engines = [
        str(path) for path in required_engines if not path.is_file()
    ]
    report: dict[str, object] = {
        "llm_engine_root": str(llm_root),
        "visual_engine_root": str(visual_root),
        "required_engines": [str(path) for path in required_engines],
        "missing_engines": missing_engines,
        "ready": not missing_engines,
    }
    try:
        configure_edge_llm_environment(edge_llm_root, plugin_path)
    except (FileNotFoundError, ValueError) as error:
        report["ready"] = False
        report["configuration_error"] = str(error)
    if edge_llm_root is not None:
        report["edge_llm_root"] = str(edge_llm_root.resolve())
    configured_plugin = os.environ.get("EDGELLM_PLUGIN_PATH")
    if configured_plugin:
        report["plugin_path"] = configured_plugin
    return report


def build_runtime_metadata(
    *,
    llm_engine_root: Path,
    visual_engine_root: Path,
    edge_llm_root: Path | None,
    plugin_path: Path | None,
    cuda_graph: str | None,
    pin_optimization_profiles: bool | None,
    profile_switch_timing: bool | None,
    cache_binding_state: bool | None,
    cache_registered_bindings: bool | None,
    skip_redundant_profile_switch: bool | None,
    skip_fallback_binding_scan: bool | None,
    runtime_patches: tuple[Path, ...] = (),
    runtime_tuning_config: Path | None = None,
    runtime_variant_id: str | None = None,
    patch_chain_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一次服务启动的 engine 与 runtime 开关身份。"""
    metadata = {
        "schema_version": "parksight_tensorrt_runtime_options_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engines": {
            "llm": _artifact_identity(llm_engine_root / "llm.engine"),
            "visual": _artifact_identity(visual_engine_root / "visual.engine"),
        },
        "edge_llm_root": str(edge_llm_root.resolve()) if edge_llm_root else None,
        "plugin_path": str(plugin_path.resolve()) if plugin_path else os.environ.get("EDGELLM_PLUGIN_PATH"),
        "options": {
            "cuda_graph": cuda_graph or "default",
            "pin_optimization_profiles": _option_state(pin_optimization_profiles),
            "profile_switch_timing": _option_state(profile_switch_timing),
            "cache_binding_state": _option_state(cache_binding_state),
            "cache_registered_bindings": _option_state(cache_registered_bindings),
            "skip_redundant_profile_switch": _option_state(skip_redundant_profile_switch),
            "skip_fallback_binding_scan": _option_state(skip_fallback_binding_scan),
            "weight_streaming_budget_bytes": os.environ.get(
                "EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES"
            ),
            "int4_gemv_n_per_block": os.environ.get(
                "EDGELLM_INT4_GEMV_N_PER_BLOCK", "default"
            ),
            "int4_gemv_block_size": os.environ.get(
                "EDGELLM_INT4_GEMV_BLOCK_SIZE", "default"
            ),
            "fmha_force_granular_tiling": os.environ.get(
                "EDGELLM_FMHA_FORCE_GRANULAR_TILING", "default"
            ),
        },
        "environment": {
            key: os.environ[key]
            for key in (
                "EDGELLM_DISABLE_CUDA_GRAPH",
                "EDGELLM_PIN_OPTIMIZATION_PROFILES",
                "EDGELLM_PROFILE_SWITCH_TIMING",
                "EDGELLM_CACHE_BINDING_STATE",
                "EDGELLM_CACHE_REGISTERED_BINDINGS",
                "EDGELLM_SKIP_REDUNDANT_PROFILE_SWITCH",
                "EDGELLM_SKIP_FALLBACK_BINDING_SCAN",
                "EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES",
                "EDGELLM_LOG_XQA_SELECTION",
                "EDGELLM_FMHA_FORCE_GRANULAR_TILING",
                "EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE",
                "EDGELLM_GREEDY_ARGMAX_IMPL",
                "EDGELLM_DIRECT_DEVICE_TOKEN_EMBED",
                "EDGELLM_INT4_GEMM_STAGES",
                "EDGELLM_INT4_GEMV_N_PER_BLOCK",
                "EDGELLM_INT4_GEMV_BLOCK_SIZE",
            )
            if key in os.environ
        },
        "runtime_patches": _runtime_patch_identities(runtime_patches),
        "runtime_tuning": {
            "config_path": (
                str(runtime_tuning_config.resolve())
                if runtime_tuning_config is not None
                else None
            ),
            "variant_id": runtime_variant_id,
        },
    }
    if patch_chain_verification is not None:
        metadata["patch_chain_verification"] = dict(patch_chain_verification)
    return metadata


def _write_runtime_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"engine file not found: {path.resolve()}")
    resolved_path = path.resolve()
    return {
        # Keep both names: the resolved path is the file actually hashed and
        # loaded, while requested_path preserves symlink-based deployment
        # provenance (for example INT4 LLM + shared FP16 visual engine).
        "path": str(resolved_path),
        "requested_path": str(path.absolute()),
        "resolved_path": str(resolved_path),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _runtime_patch_identities(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    """记录显式应用到 Edge-LLM checkout 的运行时补丁身份。"""
    identities: list[dict[str, Any]] = []
    for path in paths:
        resolved_path = path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"runtime patch not found: {resolved_path}")
        identities.append(
            {
                "path": str(resolved_path),
                "filename": resolved_path.name,
                "size_bytes": resolved_path.stat().st_size,
                "sha256": sha256_file(resolved_path),
            }
        )
    return identities


def verify_runtime_patch_chain(
    *,
    edge_llm_root: Path | None,
    runtime_patches: tuple[Path, ...],
) -> dict[str, Any]:
    """在服务启动前验证 runtime patch 能按序应用到当前源码 revision。"""
    if edge_llm_root is None:
        raise ValueError("--verify-runtime-patch-chain requires --edge-llm-root")
    if not runtime_patches:
        raise ValueError(
            "--verify-runtime-patch-chain requires at least one --runtime-patch"
        )
    try:
        from scripts.check_tensorrt_patch_chain import check_patch_chain
    except ModuleNotFoundError:
        from check_tensorrt_patch_chain import check_patch_chain
    return check_patch_chain(
        edge_llm_root=edge_llm_root,
        patches=runtime_patches,
    )


def load_runtime_variant(config_path: Path, variant_id: str):
    """读取一个 runtime tuning variant，供服务入口复用固定实验矩阵。"""
    from parksight_vlm.tensorrt import TensorRTRuntimeTuningConfig

    config = TensorRTRuntimeTuningConfig.load(config_path)
    for variant in config.variants:
        if variant.variant_id == variant_id:
            return variant
    available = ", ".join(variant.variant_id for variant in config.variants)
    raise ValueError(
        f"runtime variant not found: {variant_id}; available variants: {available}"
    )


def apply_runtime_variant(args: argparse.Namespace) -> str | None:
    """将配置中的 variant 注入 CLI 参数，并拒绝静默覆盖冲突参数。"""
    config_path = getattr(args, "runtime_tuning_config", None)
    variant_id = getattr(args, "runtime_variant", None)
    if config_path is None and variant_id is None:
        return None
    if config_path is None or variant_id is None:
        raise ValueError(
            "--runtime-tuning-config and --runtime-variant must be provided together"
        )
    variant = load_runtime_variant(config_path, variant_id)
    fields = (
        "pin_optimization_profiles",
        "cache_binding_state",
        "cache_registered_bindings",
        "skip_redundant_profile_switch",
        "skip_fallback_binding_scan",
        "fmha_force_granular_tiling",
        "int4_gemv_n_per_block",
        "int4_gemv_block_size",
    )
    for field in fields:
        configured_value = getattr(variant, field)
        if configured_value is None:
            continue
        cli_value = getattr(args, field, None)
        if cli_value is not None and cli_value != configured_value:
            raise ValueError(
                f"runtime variant {variant_id!r} conflicts with --{field.replace('_', '-')}: "
                f"CLI={cli_value}, config={configured_value}"
            )
        setattr(args, field, configured_value)
    configured_patches = tuple(Path(path) for path in variant.runtime_patches)
    if configured_patches:
        cli_patches = tuple(Path(path) for path in getattr(args, "runtime_patch", ()))
        if cli_patches:
            configured_identity = tuple(path.resolve() for path in configured_patches)
            cli_identity = tuple(path.resolve() for path in cli_patches)
            if cli_identity != configured_identity:
                raise ValueError(
                    f"runtime variant {variant_id!r} conflicts with --runtime-patch sequence"
                )
        else:
            setattr(args, "runtime_patch", list(configured_patches))
    return variant_id


def _option_state(value: bool | None) -> str:
    if value is True:
        return "enabled"
    if value is False:
        return "disabled"
    return "default"


def _configure_toggle(environment_name: str, enabled: bool | None) -> None:
    """Apply a tri-state runtime toggle without overwriting omitted settings."""
    if enabled is None:
        return
    if enabled:
        os.environ[environment_name] = "1"
    else:
        os.environ.pop(environment_name, None)


def configure_weight_streaming_budget(budget_bytes: int | None) -> None:
    """在导入 C++ runtime 前设置可选的 TensorRT 权重驻留预算。"""
    if budget_bytes is None:
        return
    if budget_bytes < 0:
        raise ValueError("weight streaming budget must not be negative")
    os.environ["EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES"] = str(budget_bytes)


def configure_cuda_graph(mode: str | None) -> None:
    """设置固定 Edge-LLM patch 使用的 CUDA Graph 开关。"""
    if mode is None:
        return
    if mode == "enabled":
        os.environ.pop("EDGELLM_DISABLE_CUDA_GRAPH", None)
        return
    if mode == "disabled":
        os.environ["EDGELLM_DISABLE_CUDA_GRAPH"] = "1"
        return
    raise ValueError("cuda graph mode must be enabled, disabled, or omitted")


def configure_profile_contexts(enabled: bool | None) -> None:
    """设置 profile-pinned execution context 实验开关。"""
    _configure_toggle("EDGELLM_PIN_OPTIMIZATION_PROFILES", enabled)


def configure_profile_switch_timing(enabled: bool | None) -> None:
    """设置 profile API 主机调用打点实验开关。"""
    _configure_toggle("EDGELLM_PROFILE_SWITCH_TIMING", enabled)


def configure_binding_state_cache(enabled: bool | None) -> None:
    """设置 prepare() 后复用 binding snapshot 的实验开关。"""
    _configure_toggle("EDGELLM_CACHE_BINDING_STATE", enabled)


def configure_registered_binding_cache(enabled: bool | None) -> None:
    """设置已注册 tensor 的 address/shape 绑定缓存实验开关。"""
    _configure_toggle("EDGELLM_CACHE_REGISTERED_BINDINGS", enabled)


def configure_redundant_profile_switch(enabled: bool | None) -> None:
    """设置同 profile 重复 setOptimizationProfileAsync 跳过开关。"""
    _configure_toggle("EDGELLM_SKIP_REDUNDANT_PROFILE_SWITCH", enabled)


def configure_fallback_binding_scan(enabled: bool | None) -> None:
    """设置已确认全量 registry binding 时跳过 fallback scan 的实验开关。"""
    _configure_toggle("EDGELLM_SKIP_FALLBACK_BINDING_SCAN", enabled)


def configure_xqa_selection_logging(enabled: bool | None) -> None:
    """设置 0028 XQA 实际 kernel 选择日志开关。"""
    _configure_toggle("EDGELLM_LOG_XQA_SELECTION", enabled)


def configure_fmha_force_granular_tiling(enabled: bool | None) -> None:
    """设置 0043 head_dim=128 tiled FMHA 候选开关。"""
    _configure_toggle("EDGELLM_FMHA_FORCE_GRANULAR_TILING", enabled)


def configure_greedy_argmax_block_size(block_size: int | None) -> None:
    """设置 0029 greedy top-1 kernel 的编译候选 block size。"""
    if block_size is None:
        return
    if block_size not in {256, 1024}:
        raise ValueError("greedy argmax block size must be 256 or 1024")
    os.environ["EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE"] = str(block_size)


def configure_greedy_argmax_impl(implementation: str | None) -> None:
    """设置 0035 greedy top-1 reduction 实现候选。"""
    if implementation is None:
        return
    if implementation not in {"cub", "warp"}:
        raise ValueError("greedy argmax implementation must be cub or warp")
    if implementation == "warp":
        os.environ["EDGELLM_GREEDY_ARGMAX_IMPL"] = "warp"
    else:
        os.environ.pop("EDGELLM_GREEDY_ARGMAX_IMPL", None)


def configure_direct_device_token_embedding(enabled: bool | None) -> None:
    """设置 0030/0031 device-side token embedding 候选开关。"""
    _configure_toggle("EDGELLM_DIRECT_DEVICE_TOKEN_EMBED", enabled)


def configure_int4_gemm_stages(stages: int | None) -> None:
    """设置 0032 INT4 W4A16 pipeline stages 候选。"""
    if stages is None:
        return
    if stages not in {2, 3, 4}:
        raise ValueError("INT4 GEMM stages must be 2, 3, or 4")
    os.environ["EDGELLM_INT4_GEMM_STAGES"] = str(stages)


def configure_int4_gemv_n_per_block(n_per_block: int | None) -> None:
    """设置 0038 INT4 GEMV 每个 block 的输出组数候选。"""
    if n_per_block is None:
        return
    if n_per_block not in {2, 4}:
        raise ValueError("INT4 GEMV N per block must be 2 or 4")
    if n_per_block == 2:
        os.environ.pop("EDGELLM_INT4_GEMV_N_PER_BLOCK", None)
    else:
        os.environ["EDGELLM_INT4_GEMV_N_PER_BLOCK"] = str(n_per_block)


def configure_int4_gemv_block_size(block_size: int | None) -> None:
    """设置 0044 INT4 GEMV threads-per-block 候选。"""
    if block_size is None:
        return
    if block_size not in {128, 256, 512}:
        raise ValueError("INT4 GEMV block size must be 128, 256, or 512")
    if block_size == 256:
        os.environ.pop("EDGELLM_INT4_GEMV_BLOCK_SIZE", None)
    else:
        os.environ["EDGELLM_INT4_GEMV_BLOCK_SIZE"] = str(block_size)


def configure_edge_llm_environment(
    edge_llm_root: Path | None,
    plugin_path: Path | None,
) -> None:
    """配置源码、pybind 和插件路径，避免依赖隐式 shell 环境。"""
    if edge_llm_root is not None:
        resolved_root = edge_llm_root.resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"TensorRT Edge-LLM root not found: {resolved_root}")
        for import_path in (resolved_root, resolved_root / "build" / "pybind"):
            if import_path.is_dir() and str(import_path) not in sys.path:
                sys.path.insert(0, str(import_path))
        os.environ.setdefault("BUILD_DIR", str(resolved_root / "build"))
        if plugin_path is None and not os.environ.get("EDGELLM_PLUGIN_PATH"):
            discovered_plugin = resolved_root / "build" / "libNvInfer_edgellm_plugin.so"
            if discovered_plugin.is_file():
                os.environ["EDGELLM_PLUGIN_PATH"] = str(discovered_plugin)

    configured_plugin = plugin_path
    if configured_plugin is None and os.environ.get("EDGELLM_PLUGIN_PATH"):
        configured_plugin = Path(os.environ["EDGELLM_PLUGIN_PATH"])
    if configured_plugin is not None:
        resolved_plugin = configured_plugin.resolve()
        if not resolved_plugin.is_file():
            raise FileNotFoundError(f"Edge-LLM plugin not found: {resolved_plugin}")
        os.environ["EDGELLM_PLUGIN_PATH"] = str(resolved_plugin)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root")
    parser.add_argument(
        "--llm-engine-root",
        type=Path,
        help="包含 llm.engine 的目录；与 --visual-engine-root 配对使用",
    )
    parser.add_argument(
        "--visual-engine-root",
        type=Path,
        help="包含 visual.engine 的目录；与 --llm-engine-root 配对使用",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--http-max-batch-size",
        type=int,
        default=1,
        help="HTTP 微批调度的最大原生 batch；必须不大于已构建 engine 的 maxBatch",
    )
    parser.add_argument(
        "--weight-streaming-budget-bytes",
        type=int,
        default=None,
        help="TensorRT LLM 权重 GPU 驻留预算；8GB Jetson 可使用 0 以最大化节省",
    )
    parser.add_argument(
        "--cuda-graph",
        choices=("enabled", "disabled"),
        default=None,
        help="CUDA Graph A/B；disabled 需要应用 0011-configurable-cuda-graph.patch",
    )
    parser.add_argument(
        "--pin-optimization-profiles",
        action="store_true",
        default=None,
        help="启用 0014 profile-pinned execution context 实验补丁",
    )
    parser.add_argument(
        "--profile-switch-timing",
        action="store_true",
        default=None,
        help="启用 0013 profile API 主机调用耗时打点",
    )
    parser.add_argument(
        "--cache-binding-state",
        action="store_true",
        default=None,
        help="启用 0015 prepare 后 binding snapshot 缓存实验补丁",
    )
    parser.add_argument(
        "--cache-registered-bindings",
        action="store_true",
        default=None,
        help="启用 0017 registered binding address/shape 缓存实验补丁",
    )
    parser.add_argument(
        "--skip-redundant-profile-switch",
        action="store_true",
        default=None,
        help="启用 0018 同 profile 重复 profile-switch 跳过实验补丁",
    )
    parser.add_argument(
        "--skip-fallback-binding-scan",
        action="store_true",
        default=None,
        help="启用 0020 全量 registry binding 时跳过 fallback scan 实验补丁",
    )
    parser.add_argument(
        "--log-xqa-selection",
        action="store_true",
        default=None,
        help="启用 0028 XQA 首次实际 kernel 选择日志",
    )
    parser.add_argument(
        "--fmha-force-granular-tiling",
        action="store_true",
        default=None,
        help="启用 0043 SM87/head_dim=128 长序列 tiled FMHA 候选",
    )
    parser.add_argument(
        "--greedy-argmax-block-size",
        type=int,
        choices=(256, 1024),
        help="0029 greedy top-1 kernel block size 候选",
    )
    parser.add_argument(
        "--greedy-argmax-impl",
        choices=("cub", "warp"),
        help="0035 greedy top-1 reduction 实现候选",
    )
    parser.add_argument(
        "--direct-device-token-embedding",
        action="store_true",
        default=None,
        help="启用 0030/0031 device-side token embedding 候选",
    )
    parser.add_argument(
        "--int4-gemm-stages",
        type=int,
        choices=(2, 3, 4),
        help="0032 INT4 W4A16 pipeline stages 候选",
    )
    parser.add_argument(
        "--int4-gemv-n-per-block",
        type=int,
        choices=(2, 4),
        help="0038 INT4 GEMV 每个 block 的输出组数候选；4 仅对 M=1 且 N 可整除 16 生效",
    )
    parser.add_argument(
        "--int4-gemv-block-size",
        type=int,
        choices=(128, 256, 512),
        help="0044 INT4 GEMV threads/block 候选；默认 256",
    )
    parser.add_argument(
        "--runtime-patch",
        action="append",
        type=Path,
        default=[],
        help="显式应用到 Edge-LLM checkout 的补丁，可重复传入并写入 provenance",
    )
    parser.add_argument(
        "--runtime-metadata-output",
        type=Path,
        help="写出 engine SHA-256 与本次 runtime 开关身份 JSON",
    )
    parser.add_argument(
        "--runtime-tuning-config",
        type=Path,
        help="runtime tuning JSON；需与 --runtime-variant 一起使用",
    )
    parser.add_argument(
        "--runtime-variant",
        help="从 --runtime-tuning-config 选择一个固定 runtime/kernel 变体",
    )
    parser.add_argument(
        "--verify-runtime-patch-chain",
        action="store_true",
        help="启动前在临时 worktree 校验 --runtime-patch 的顺序和可应用性",
    )
    parser.add_argument(
        "--edge-llm-root",
        type=Path,
        default=Path(os.environ["EDGE_LLM_ROOT"])
        if os.environ.get("EDGE_LLM_ROOT")
        else None,
        help="TensorRT Edge-LLM checkout；自动加入源码和 pybind import 路径",
    )
    parser.add_argument(
        "--plugin-path",
        type=Path,
        default=None,
        help="TensorRT Edge-LLM plugin .so；未指定时从 --edge-llm-root 自动发现",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查 engine、源码和 plugin 路径，不导入或加载 GPU runtime",
    )
    args = parser.parse_args(argv)

    if args.http_max_batch_size <= 0:
        parser.error("--http-max-batch-size must be positive")

    try:
        runtime_variant_id = apply_runtime_variant(args)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    patch_chain_verification = None
    if args.verify_runtime_patch_chain:
        try:
            patch_chain_verification = verify_runtime_patch_chain(
                edge_llm_root=args.edge_llm_root,
                runtime_patches=tuple(args.runtime_patch),
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            parser.error(str(error))

    engine_root = Path(args.engine_root).resolve() if args.engine_root else None
    if args.check_only:
        report = deployment_readiness(
            engine_root=engine_root,
            llm_engine_root=args.llm_engine_root,
            visual_engine_root=args.visual_engine_root,
            edge_llm_root=args.edge_llm_root,
            plugin_path=args.plugin_path,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 2

    try:
        llm_engine_root, visual_engine_root = _resolve_engine_directories(
            engine_root=engine_root,
            llm_engine_root=args.llm_engine_root,
            visual_engine_root=args.visual_engine_root,
        )
    except ValueError as error:
        parser.error(str(error))
    requested_llm_engine_root = llm_engine_root
    requested_visual_engine_root = visual_engine_root
    llm_engine_root = llm_engine_root.resolve()
    visual_engine_root = visual_engine_root.resolve()
    required_engines = (
        llm_engine_root / "llm.engine",
        visual_engine_root / "visual.engine",
    )
    missing_engines = [
        str(path) for path in required_engines if not path.is_file()
    ]
    if missing_engines:
        raise FileNotFoundError(f"缺少预构建 engine：{missing_engines}")

    configure_edge_llm_environment(args.edge_llm_root, args.plugin_path)
    configure_weight_streaming_budget(args.weight_streaming_budget_bytes)
    configure_cuda_graph(args.cuda_graph)
    configure_profile_contexts(args.pin_optimization_profiles)
    configure_profile_switch_timing(args.profile_switch_timing)
    configure_binding_state_cache(args.cache_binding_state)
    configure_registered_binding_cache(args.cache_registered_bindings)
    configure_redundant_profile_switch(args.skip_redundant_profile_switch)
    configure_fallback_binding_scan(args.skip_fallback_binding_scan)
    configure_xqa_selection_logging(args.log_xqa_selection)
    configure_fmha_force_granular_tiling(args.fmha_force_granular_tiling)
    configure_greedy_argmax_block_size(args.greedy_argmax_block_size)
    configure_greedy_argmax_impl(args.greedy_argmax_impl)
    configure_direct_device_token_embedding(args.direct_device_token_embedding)
    configure_int4_gemm_stages(args.int4_gemm_stages)
    configure_int4_gemv_n_per_block(args.int4_gemv_n_per_block)
    configure_int4_gemv_block_size(args.int4_gemv_block_size)
    if args.runtime_metadata_output is not None:
        _write_runtime_metadata(
            args.runtime_metadata_output,
            build_runtime_metadata(
                llm_engine_root=requested_llm_engine_root,
                visual_engine_root=requested_visual_engine_root,
                edge_llm_root=args.edge_llm_root,
                plugin_path=args.plugin_path,
                cuda_graph=args.cuda_graph,
                pin_optimization_profiles=args.pin_optimization_profiles,
                profile_switch_timing=args.profile_switch_timing,
                cache_binding_state=args.cache_binding_state,
                cache_registered_bindings=args.cache_registered_bindings,
                skip_redundant_profile_switch=args.skip_redundant_profile_switch,
                skip_fallback_binding_scan=args.skip_fallback_binding_scan,
                # The selected FMHA candidate is captured from the environment
                # in build_runtime_metadata after configuration above.
                runtime_patches=tuple(args.runtime_patch),
                runtime_tuning_config=args.runtime_tuning_config,
                runtime_variant_id=runtime_variant_id,
                patch_chain_verification=patch_chain_verification,
            ),
        )
    serve_prebuilt_engines(
        llm_engine_root=llm_engine_root,
        visual_engine_root=visual_engine_root,
        host=args.host,
        port=args.port,
        http_max_batch_size=args.http_max_batch_size,
        cuda_graph=args.cuda_graph,
        pin_optimization_profiles=args.pin_optimization_profiles,
        profile_switch_timing=args.profile_switch_timing,
        cache_binding_state=args.cache_binding_state,
        cache_registered_bindings=args.cache_registered_bindings,
        skip_redundant_profile_switch=args.skip_redundant_profile_switch,
        skip_fallback_binding_scan=args.skip_fallback_binding_scan,
        log_xqa_selection=args.log_xqa_selection,
        fmha_force_granular_tiling=args.fmha_force_granular_tiling,
        greedy_argmax_block_size=args.greedy_argmax_block_size,
        greedy_argmax_impl=args.greedy_argmax_impl,
        direct_device_token_embedding=args.direct_device_token_embedding,
        int4_gemm_stages=args.int4_gemm_stages,
        int4_gemv_n_per_block=args.int4_gemv_n_per_block,
        int4_gemv_block_size=args.int4_gemv_block_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
