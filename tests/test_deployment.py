"""TensorRT Edge-LLM 部署入口的无硬件行为测试。"""

from __future__ import annotations

import hashlib
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

from scripts.build_edgellm_vlm_engines import (
    build_commands,
    require_paged_kv_build_path,
)
from scripts.serve_edgellm import (
    build_runtime_metadata,
    apply_runtime_variant,
    configure_cuda_graph,
    configure_edge_llm_environment,
    configure_binding_state_cache,
    configure_redundant_profile_switch,
    configure_registered_binding_cache,
    configure_fallback_binding_scan,
    configure_xqa_selection_logging,
    configure_fmha_force_granular_tiling,
    configure_greedy_argmax_block_size,
    configure_greedy_argmax_impl,
    configure_direct_device_token_embedding,
    configure_int4_gemm_stages,
    configure_int4_gemv_n_per_block,
    configure_int4_gemv_block_size,
    configure_profile_contexts,
    configure_profile_switch_timing,
    configure_weight_streaming_budget,
    deployment_readiness,
    serve_prebuilt_engines,
    verify_runtime_patch_chain,
)


class DeploymentTests(unittest.TestCase):
    def test_runtime_variant_applies_kernel_values_from_matrix(self) -> None:
        args = SimpleNamespace(
            runtime_tuning_config=(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "tensorrt"
                / "jetson_orin_nano_int4_gemv_runtime_v1.json"
            ),
            runtime_variant="block128_n2",
            pin_optimization_profiles=None,
            cache_binding_state=None,
            cache_registered_bindings=None,
            skip_redundant_profile_switch=None,
            skip_fallback_binding_scan=None,
            int4_gemv_n_per_block=None,
            int4_gemv_block_size=None,
            runtime_patch=[],
        )

        selected = apply_runtime_variant(args)

        self.assertEqual(selected, "block128_n2")
        self.assertEqual(args.int4_gemv_n_per_block, 2)
        self.assertEqual(args.int4_gemv_block_size, 128)
        self.assertFalse(args.pin_optimization_profiles)
        self.assertEqual(
            [path.name for path in args.runtime_patch],
            [
                "0038-int4-gemv-nperblock4.patch",
                "0049-int4-gemv-block-size-candidate-v2.patch",
            ],
        )

    def test_runtime_variant_applies_fmha_candidate_from_matrix(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "tensorrt"
            / "jetson_orin_nano_int4_fmha_runtime_v1.json"
        )
        args = SimpleNamespace(
            runtime_tuning_config=config_path,
            runtime_variant="candidate_head128_force_granular_tiling",
            pin_optimization_profiles=None,
            cache_binding_state=None,
            cache_registered_bindings=None,
            skip_redundant_profile_switch=None,
            skip_fallback_binding_scan=None,
            fmha_force_granular_tiling=None,
            int4_gemv_n_per_block=None,
            int4_gemv_block_size=None,
            runtime_patch=[],
        )

        selected = apply_runtime_variant(args)

        self.assertEqual(selected, "candidate_head128_force_granular_tiling")
        self.assertTrue(args.fmha_force_granular_tiling)
        self.assertEqual(
            [path.name for path in args.runtime_patch],
            ["0043-fmha-head128-tiled-candidate.patch"],
        )

    def test_runtime_variant_rejects_conflicting_cli_value(self) -> None:
        args = SimpleNamespace(
            runtime_tuning_config=(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "tensorrt"
                / "jetson_orin_nano_int4_gemv_runtime_v1.json"
            ),
            runtime_variant="block128_n2",
            pin_optimization_profiles=None,
            cache_binding_state=None,
            cache_registered_bindings=None,
            skip_redundant_profile_switch=None,
            skip_fallback_binding_scan=None,
            int4_gemv_n_per_block=2,
            int4_gemv_block_size=512,
            runtime_patch=[],
        )

        with self.assertRaisesRegex(ValueError, "conflicts"):
            apply_runtime_variant(args)

    def test_runtime_variant_rejects_different_patch_sequence(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "tensorrt"
            / "jetson_orin_nano_int4_gemv_runtime_v1.json"
        )
        args = SimpleNamespace(
            runtime_tuning_config=config_path,
            runtime_variant="block128_n2",
            pin_optimization_profiles=None,
            cache_binding_state=None,
            cache_registered_bindings=None,
            skip_redundant_profile_switch=None,
            skip_fallback_binding_scan=None,
            int4_gemv_n_per_block=None,
            int4_gemv_block_size=None,
            runtime_patch=[Path("patches/tensorrt-edge-llm/0038-int4-gemv-nperblock4.patch")],
        )

        with self.assertRaisesRegex(ValueError, "patch sequence"):
            apply_runtime_variant(args)

    def test_runtime_patch_chain_verification_is_recordable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            patch_file = root / "candidate.patch"
            patch_file.write_text("patch", encoding="utf-8")
            with patch(
                "scripts.check_tensorrt_patch_chain.check_patch_chain",
                return_value={
                    "schema_version": "parksight_tensorrt_patch_chain_v1",
                    "status": "succeeded",
                    "edge_llm_revision": "edge-revision",
                },
            ) as checker:
                verification = verify_runtime_patch_chain(
                    edge_llm_root=root,
                    runtime_patches=(patch_file,),
                )

            checker.assert_called_once_with(
                edge_llm_root=root,
                patches=(patch_file,),
            )
            self.assertEqual(verification["status"], "succeeded")

            llm_root = root / "llm"
            visual_root = root / "visual"
            llm_root.mkdir()
            visual_root.mkdir()
            (llm_root / "llm.engine").write_bytes(b"llm-engine")
            (visual_root / "visual.engine").write_bytes(b"visual-engine")
            metadata = build_runtime_metadata(
                llm_engine_root=llm_root,
                visual_engine_root=visual_root,
                edge_llm_root=root,
                plugin_path=None,
                cuda_graph=None,
                pin_optimization_profiles=None,
                profile_switch_timing=None,
                cache_binding_state=None,
                cache_registered_bindings=None,
                skip_redundant_profile_switch=None,
                skip_fallback_binding_scan=None,
                runtime_patches=(patch_file,),
                patch_chain_verification=verification,
            )
            self.assertEqual(
                metadata["patch_chain_verification"]["edge_llm_revision"],
                "edge-revision",
            )

    def test_runtime_metadata_binds_engine_hashes_and_runtime_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            llm_root = root / "llm"
            visual_root = root / "visual"
            llm_root.mkdir()
            visual_root.mkdir()
            (llm_root / "llm.engine").write_bytes(b"llm-engine")
            (visual_root / "visual.engine").write_bytes(b"visual-engine")
            with patch.dict(
                "os.environ",
                {
                    "EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES": "0",
                    "EDGELLM_LOG_XQA_SELECTION": "1",
                    "EDGELLM_FMHA_FORCE_GRANULAR_TILING": "1",
                    "EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE": "1024",
                    "EDGELLM_GREEDY_ARGMAX_IMPL": "warp",
                    "EDGELLM_DIRECT_DEVICE_TOKEN_EMBED": "1",
                    "EDGELLM_INT4_GEMM_STAGES": "2",
                    "EDGELLM_INT4_GEMV_N_PER_BLOCK": "4",
                    "EDGELLM_INT4_GEMV_BLOCK_SIZE": "512",
                },
                clear=True,
            ):
                metadata = build_runtime_metadata(
                    llm_engine_root=llm_root,
                    visual_engine_root=visual_root,
                    edge_llm_root=None,
                    plugin_path=None,
                    cuda_graph="enabled",
                    pin_optimization_profiles=True,
                    profile_switch_timing=None,
                    cache_binding_state=True,
                    cache_registered_bindings=True,
                    skip_redundant_profile_switch=False,
                    skip_fallback_binding_scan=None,
                    runtime_tuning_config=(
                        Path(__file__).resolve().parents[1]
                        / "configs"
                        / "tensorrt"
                        / "jetson_orin_nano_int4_gemv_runtime_v1.json"
                    ),
                    runtime_variant_id="block512_n2",
                    stream_usage=True,
                )

        self.assertEqual(metadata["engines"]["llm"]["size_bytes"], len(b"llm-engine"))
        self.assertEqual(metadata["engines"]["visual"]["size_bytes"], len(b"visual-engine"))
        self.assertEqual(
            metadata["engines"]["llm"]["resolved_path"],
            metadata["engines"]["llm"]["path"],
        )
        self.assertTrue(metadata["engines"]["visual"]["requested_path"].endswith("visual.engine"))
        self.assertEqual(metadata["options"]["pin_optimization_profiles"], "enabled")
        self.assertEqual(metadata["options"]["profile_switch_timing"], "default")
        self.assertEqual(metadata["options"]["skip_redundant_profile_switch"], "disabled")
        self.assertEqual(metadata["options"]["weight_streaming_budget_bytes"], "0")
        self.assertEqual(metadata["environment"]["EDGELLM_LOG_XQA_SELECTION"], "1")
        self.assertEqual(metadata["environment"]["EDGELLM_FMHA_FORCE_GRANULAR_TILING"], "1")
        self.assertEqual(metadata["environment"]["EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE"], "1024")
        self.assertEqual(metadata["environment"]["EDGELLM_GREEDY_ARGMAX_IMPL"], "warp")
        self.assertEqual(metadata["environment"]["EDGELLM_DIRECT_DEVICE_TOKEN_EMBED"], "1")
        self.assertEqual(metadata["environment"]["EDGELLM_INT4_GEMM_STAGES"], "2")
        self.assertEqual(metadata["environment"]["EDGELLM_INT4_GEMV_N_PER_BLOCK"], "4")
        self.assertEqual(metadata["options"]["int4_gemv_block_size"], "512")
        self.assertEqual(metadata["environment"]["EDGELLM_INT4_GEMV_BLOCK_SIZE"], "512")
        self.assertEqual(metadata["runtime_tuning"]["variant_id"], "block512_n2")
        self.assertTrue(metadata["options"]["stream_usage"])
        extension_path = Path(__file__).resolve().parents[1] / "src/parksight_vlm/inference/edge_server_metrics.py"
        self.assertEqual(metadata["stream_usage_extension"]["source"]["sha256"],
                         hashlib.sha256(extension_path.read_bytes()).hexdigest())
        self.assertIn("decode_ms", metadata["stream_usage_extension"]["unknown_fields"])
        self.assertTrue(metadata["runtime_tuning"]["config_path"].endswith("jetson_orin_nano_int4_gemv_runtime_v1.json"))

    def test_runtime_metadata_binds_patch_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            llm_root = root / "llm"
            visual_root = root / "visual"
            llm_root.mkdir()
            visual_root.mkdir()
            (llm_root / "llm.engine").write_bytes(b"llm-engine")
            (visual_root / "visual.engine").write_bytes(b"visual-engine")
            patch = root / "0024-greedy.patch"
            patch.write_bytes(b"patch-content")

            metadata = build_runtime_metadata(
                llm_engine_root=llm_root,
                visual_engine_root=visual_root,
                edge_llm_root=None,
                plugin_path=None,
                cuda_graph=None,
                pin_optimization_profiles=None,
                profile_switch_timing=None,
                cache_binding_state=None,
                cache_registered_bindings=None,
                skip_redundant_profile_switch=None,
                skip_fallback_binding_scan=None,
                runtime_patches=(patch,),
            )

        self.assertEqual(metadata["runtime_patches"][0]["filename"], patch.name)
        self.assertEqual(metadata["runtime_patches"][0]["size_bytes"], len(b"patch-content"))
        self.assertEqual(
            metadata["runtime_patches"][0]["sha256"],
            hashlib.sha256(b"patch-content").hexdigest(),
        )

    def test_runtime_metadata_rejects_missing_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            llm_root = root / "llm"
            visual_root = root / "visual"
            llm_root.mkdir()
            visual_root.mkdir()
            (llm_root / "llm.engine").write_bytes(b"llm")
            (visual_root / "visual.engine").write_bytes(b"visual")
            with self.assertRaisesRegex(FileNotFoundError, "runtime patch not found"):
                build_runtime_metadata(
                    llm_engine_root=llm_root,
                    visual_engine_root=visual_root,
                    edge_llm_root=None,
                    plugin_path=None,
                    cuda_graph=None,
                    pin_optimization_profiles=None,
                    profile_switch_timing=None,
                    cache_binding_state=None,
                    cache_registered_bindings=None,
                    skip_redundant_profile_switch=None,
                    skip_fallback_binding_scan=None,
                    runtime_patches=(root / "missing.patch",),
                )

    def test_edge_llm_environment_discovers_import_and_plugin_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pybind = root / "build" / "pybind"
            pybind.mkdir(parents=True)
            plugin = root / "build" / "libNvInfer_edgellm_plugin.so"
            plugin.write_bytes(b"plugin")
            with patch.dict("os.environ", {}, clear=True), patch.object(
                sys, "path", []
            ):
                configure_edge_llm_environment(root, None)

                self.assertIn(str(root), sys.path)
                self.assertIn(str(pybind), sys.path)
                self.assertEqual(os.environ["BUILD_DIR"], str(root / "build"))
                self.assertEqual(os.environ["EDGELLM_PLUGIN_PATH"], str(plugin))

    def test_edge_llm_environment_rejects_missing_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "plugin not found"):
                configure_edge_llm_environment(
                    Path(temporary_directory),
                    Path(temporary_directory) / "missing.so",
                )

    def test_edge_llm_environment_rejects_missing_plugin_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(
                "os.environ",
                {
                    "EDGELLM_PLUGIN_PATH": str(
                        Path(temporary_directory) / "missing.so"
                    )
                },
                clear=True,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "plugin not found"):
                    configure_edge_llm_environment(None, None)

    def test_weight_streaming_budget_is_exported_for_runtime(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_weight_streaming_budget(0)
            self.assertEqual(
                os.environ["EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES"], "0"
            )

    def test_weight_streaming_budget_rejects_negative_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            configure_weight_streaming_budget(-1)

    def test_cuda_graph_mode_is_explicit_and_reversible(self) -> None:
        with patch.dict(
            "os.environ",
            {"EDGELLM_DISABLE_CUDA_GRAPH": "1"},
            clear=True,
        ):
            configure_cuda_graph("enabled")
            self.assertNotIn("EDGELLM_DISABLE_CUDA_GRAPH", os.environ)
            configure_cuda_graph("disabled")
            self.assertEqual(os.environ["EDGELLM_DISABLE_CUDA_GRAPH"], "1")

    def test_cuda_graph_mode_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "cuda graph mode"):
            configure_cuda_graph("auto")

    def test_profile_context_mode_is_explicit_and_reversible(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_profile_contexts(True)
            self.assertEqual(os.environ["EDGELLM_PIN_OPTIMIZATION_PROFILES"], "1")
            configure_profile_contexts(False)
            self.assertNotIn("EDGELLM_PIN_OPTIMIZATION_PROFILES", os.environ)

    def test_profile_context_mode_omitted_preserves_environment(self) -> None:
        with patch.dict(
            "os.environ", {"EDGELLM_PIN_OPTIMIZATION_PROFILES": "1"}, clear=True
        ):
            configure_profile_contexts(None)
            self.assertEqual(os.environ["EDGELLM_PIN_OPTIMIZATION_PROFILES"], "1")

    def test_profile_switch_timing_mode_is_explicit_and_reversible(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_profile_switch_timing(True)
            self.assertEqual(os.environ["EDGELLM_PROFILE_SWITCH_TIMING"], "1")
            configure_profile_switch_timing(False)
            self.assertNotIn("EDGELLM_PROFILE_SWITCH_TIMING", os.environ)

    def test_binding_state_cache_mode_is_explicit_and_reversible(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_binding_state_cache(True)
            self.assertEqual(os.environ["EDGELLM_CACHE_BINDING_STATE"], "1")
            configure_binding_state_cache(False)
            self.assertNotIn("EDGELLM_CACHE_BINDING_STATE", os.environ)

    def test_registered_binding_cache_mode_is_explicit_and_reversible(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_registered_binding_cache(True)
            self.assertEqual(
                os.environ["EDGELLM_CACHE_REGISTERED_BINDINGS"], "1"
            )
            configure_registered_binding_cache(False)
            self.assertNotIn("EDGELLM_CACHE_REGISTERED_BINDINGS", os.environ)

    def test_redundant_profile_switch_mode_is_explicit_and_reversible(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_redundant_profile_switch(True)
            self.assertEqual(
                os.environ["EDGELLM_SKIP_REDUNDANT_PROFILE_SWITCH"], "1"
            )
            configure_redundant_profile_switch(False)
            self.assertNotIn("EDGELLM_SKIP_REDUNDANT_PROFILE_SWITCH", os.environ)

    def test_fallback_binding_scan_mode_is_explicit_and_reversible(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_fallback_binding_scan(True)
            self.assertEqual(
                os.environ["EDGELLM_SKIP_FALLBACK_BINDING_SCAN"], "1"
            )
            configure_fallback_binding_scan(False)
            self.assertNotIn("EDGELLM_SKIP_FALLBACK_BINDING_SCAN", os.environ)

    def test_candidate_runtime_options_are_explicit_and_reversible(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            configure_xqa_selection_logging(True)
            configure_fmha_force_granular_tiling(True)
            configure_greedy_argmax_block_size(1024)
            configure_greedy_argmax_impl("warp")
            configure_direct_device_token_embedding(True)
            configure_int4_gemm_stages(2)
            configure_int4_gemv_n_per_block(4)
            configure_int4_gemv_block_size(512)
            self.assertEqual(os.environ["EDGELLM_LOG_XQA_SELECTION"], "1")
            self.assertEqual(os.environ["EDGELLM_FMHA_FORCE_GRANULAR_TILING"], "1")
            self.assertEqual(os.environ["EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE"], "1024")
            self.assertEqual(os.environ["EDGELLM_GREEDY_ARGMAX_IMPL"], "warp")
            self.assertEqual(os.environ["EDGELLM_DIRECT_DEVICE_TOKEN_EMBED"], "1")
            self.assertEqual(os.environ["EDGELLM_INT4_GEMM_STAGES"], "2")
            self.assertEqual(os.environ["EDGELLM_INT4_GEMV_N_PER_BLOCK"], "4")
            self.assertEqual(os.environ["EDGELLM_INT4_GEMV_BLOCK_SIZE"], "512")

            configure_xqa_selection_logging(False)
            configure_fmha_force_granular_tiling(False)
            configure_greedy_argmax_block_size(256)
            configure_greedy_argmax_impl("cub")
            configure_direct_device_token_embedding(False)
            configure_int4_gemm_stages(4)
            configure_int4_gemv_n_per_block(2)
            configure_int4_gemv_block_size(256)
            self.assertNotIn("EDGELLM_LOG_XQA_SELECTION", os.environ)
            self.assertNotIn("EDGELLM_FMHA_FORCE_GRANULAR_TILING", os.environ)
            self.assertEqual(os.environ["EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE"], "256")
            self.assertNotIn("EDGELLM_GREEDY_ARGMAX_IMPL", os.environ)
            self.assertNotIn("EDGELLM_DIRECT_DEVICE_TOKEN_EMBED", os.environ)
            self.assertEqual(os.environ["EDGELLM_INT4_GEMM_STAGES"], "4")
            self.assertNotIn("EDGELLM_INT4_GEMV_N_PER_BLOCK", os.environ)
            self.assertNotIn("EDGELLM_INT4_GEMV_BLOCK_SIZE", os.environ)

    def test_candidate_runtime_options_reject_unsupported_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "block size"):
            configure_greedy_argmax_block_size(512)
        with self.assertRaisesRegex(ValueError, "implementation"):
            configure_greedy_argmax_impl("invalid")
        with self.assertRaisesRegex(ValueError, "stages"):
            configure_int4_gemm_stages(1)
        with self.assertRaisesRegex(ValueError, "N per block"):
            configure_int4_gemv_n_per_block(8)
        with self.assertRaisesRegex(ValueError, "GEMV block size"):
            configure_int4_gemv_block_size(1024)

    def test_deployment_readiness_checks_both_engines_without_loading_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "llm").mkdir()
            (root / "visual").mkdir()
            (root / "llm" / "llm.engine").write_bytes(b"llm")
            (root / "visual" / "visual.engine").write_bytes(b"visual")
            edge_root = root / "edge-llm"
            (edge_root / "build").mkdir(parents=True)
            plugin = edge_root / "build" / "libNvInfer_edgellm_plugin.so"
            plugin.write_bytes(b"plugin")

            with patch.dict("os.environ", {}, clear=True):
                report = deployment_readiness(
                    engine_root=root,
                    edge_llm_root=edge_root,
                    plugin_path=None,
                )

        self.assertTrue(report["ready"])
        self.assertEqual(report["missing_engines"], [])
        self.assertEqual(report["plugin_path"], str(plugin))

    def test_deployment_readiness_reports_missing_visual_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "llm").mkdir()
            (root / "visual").mkdir()
            (root / "llm" / "llm.engine").write_bytes(b"llm")
            with patch.dict("os.environ", {}, clear=True):
                report = deployment_readiness(
                    engine_root=root,
                    edge_llm_root=None,
                    plugin_path=None,
                )

        self.assertFalse(report["ready"])
        self.assertEqual(len(report["missing_engines"]), 1)

    def test_deployment_readiness_supports_separate_llm_and_visual_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            llm_root = root / "int4" / "llm"
            visual_root = root / "fp16" / "visual"
            llm_root.mkdir(parents=True)
            visual_root.mkdir(parents=True)
            (llm_root / "llm.engine").write_bytes(b"llm")
            (visual_root / "visual.engine").write_bytes(b"visual")

            with patch.dict("os.environ", {}, clear=True):
                report = deployment_readiness(
                    llm_engine_root=llm_root,
                    visual_engine_root=visual_root,
                    edge_llm_root=None,
                    plugin_path=None,
                )

        self.assertTrue(report["ready"])
        self.assertEqual(report["missing_engines"], [])
        self.assertEqual(report["llm_engine_root"], str(llm_root.resolve()))
        self.assertEqual(report["visual_engine_root"], str(visual_root.resolve()))

    def test_vlm_engine_builder_prepares_llm_then_visual_build(self) -> None:
        edge_root = Path("/opt/TensorRT-Edge-LLM")
        onnx_root = Path("/work/onnx")
        engine_root = Path("/work/engines")

        llm_command, visual_command = build_commands(
            edge_root=edge_root,
            onnx_root=onnx_root,
            engine_root=engine_root,
            max_batch_size=1,
            max_input_len=1024,
            max_kv_cache_capacity=2048,
            min_image_tokens=8,
            max_image_tokens=2048,
            max_image_tokens_per_image=2048,
        )

        self.assertEqual(
            Path(llm_command[0]),
            edge_root / "build" / "examples" / "llm" / "llm_build",
        )
        self.assertIn(str(onnx_root / "llm"), llm_command)
        self.assertIn(str(engine_root / "llm"), llm_command)
        self.assertNotIn("--maxKVPoolPages", llm_command)
        self.assertEqual(
            Path(visual_command[0]),
            edge_root
            / "build"
            / "examples"
            / "multimodal"
            / "visual_build",
        )
        self.assertIn(str(onnx_root / "visual"), visual_command)
        visual_engine_dir_index = visual_command.index("--engineDir") + 1
        self.assertEqual(visual_command[visual_engine_dir_index], str(engine_root))

        paged_llm_command, _ = build_commands(
            edge_root=edge_root,
            onnx_root=onnx_root,
            engine_root=engine_root,
            max_batch_size=1,
            max_input_len=1024,
            max_kv_cache_capacity=1024,
            min_image_tokens=8,
            max_image_tokens=2048,
            max_image_tokens_per_image=2048,
            max_kv_pool_pages=16,
        )
        pool_index = paged_llm_command.index("--maxKVPoolPages") + 1
        self.assertEqual(paged_llm_command[pool_index], "16")

    def test_paged_kv_build_gate_requires_complete_source_path(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "complete build/runtime path"):
            require_paged_kv_build_path(
                {"summary": {"conclusion": "paged_kv_not_wired_in_checkout"}}
            )

        require_paged_kv_build_path(
            {"summary": {"conclusion": "paged_kv_path_exposed_requires_runtime_validation"}}
        )

    def test_server_loads_prebuilt_llm_and_visual_engines(self) -> None:
        calls: dict[str, object] = {}

        class FakeLlm:
            def __init__(self, **kwargs: object) -> None:
                calls["init"] = kwargs

            def serve(self, *, host: str, port: int) -> None:
                calls["serve"] = {"host": host, "port": port}

        experimental_module = ModuleType("experimental")
        server_module = ModuleType("experimental.server")
        server_module.LLM = FakeLlm  # type: ignore[attr-defined]
        experimental_module.server = server_module  # type: ignore[attr-defined]

        with patch.dict(
            "sys.modules",
            {
                "experimental": experimental_module,
                "experimental.server": server_module,
                "uvicorn": ModuleType("uvicorn"),
            },
        ):
            serve_prebuilt_engines(
                engine_root=Path("/work/engines"),
                host="127.0.0.1",
                port=8000,
            )

        self.assertEqual(
            calls["init"],
            {
                "engine_dir": str(Path("/work/engines/llm")),
                "visual_engine_dir": str(Path("/work/engines/visual")),
                "max_batch_size": 1,
            },
        )
        self.assertEqual(calls["serve"], {"host": "127.0.0.1", "port": 8000})

    def test_server_installs_usage_extension_only_when_requested(self) -> None:
        api = SimpleNamespace(_generate_stream_sse=lambda llm_instance, response_id: iter(()))
        original = api._generate_stream_sse
        calls = []
        class FakeLlm:
            def __init__(self, **kwargs):
                calls.append(bool(getattr(api._generate_stream_sse, "_parksight_stream_usage", False)))
            def serve(self, **kwargs):
                pass
        server = ModuleType("experimental.server")
        server.LLM = FakeLlm
        server.api_server = api
        with patch.dict("sys.modules", {
            "experimental": ModuleType("experimental"), "experimental.server": server,
            "uvicorn": ModuleType("uvicorn"),
        }):
            serve_prebuilt_engines(engine_root=Path("/work/engines"), host="127.0.0.1", port=8000)
            self.assertIs(api._generate_stream_sse, original)
            serve_prebuilt_engines(engine_root=Path("/work/engines"), host="127.0.0.1", port=8000,
                                  stream_usage=True)
        self.assertEqual(calls, [False, True])

    def test_server_selects_candidate_before_constructor_and_restores_upstream_loader(self):
        original = lambda: "old binding"
        runtime = object()
        calls = []
        engine = SimpleNamespace(_import_runtime=original)
        class FakeLlm:
            def __init__(self, **kwargs):
                calls.append(engine._import_runtime())
            def serve(self, **kwargs):
                self_outer.assertIs(engine._import_runtime, original)
        self_outer = self
        engine.LLM = FakeLlm
        server = ModuleType("experimental.server")
        server.LLM = FakeLlm
        server.engine = engine
        with patch.dict("sys.modules", {
            "experimental": ModuleType("experimental"), "experimental.server": server,
            "uvicorn": ModuleType("uvicorn"),
        }), patch("parksight_vlm.inference.edge_native.load_native_binding", return_value=runtime):
            serve_prebuilt_engines(engine_root=Path("/work/engines"), host="127.0.0.1", port=8000,
                                  native_binding_directory=Path("/candidate"), native_binding_sha256="0" * 64)
        self.assertEqual(calls, [runtime])

    def test_check_only_validates_candidate_identity_without_native_import(self):
        from scripts.serve_edgellm import main
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("llm/llm.engine", "visual/visual.engine"):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"engine")
            binding = root / "binding"
            binding.mkdir()
            module = binding / "_edgellm_runtime.so"
            module.write_bytes(b"candidate")
            sha = hashlib.sha256(b"candidate").hexdigest()
            stdout = io.StringIO()
            with patch.dict("os.environ", {}, clear=True), contextlib.redirect_stdout(stdout), \
                 patch("parksight_vlm.inference.edge_native.load_native_binding", side_effect=AssertionError("GPU import")):
                rc = main(["--engine-root", str(root), "--check-only",
                           "--native-binding-directory", str(binding), "--native-binding-sha256", sha])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(stdout.getvalue())["native_binding_override"],
                             {"path": str(module), "sha256": sha})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--engine-root", str(root), "--check-only", "--native-binding-directory", str(binding)])

    def test_serial_mode_disables_profiling_during_construction_and_after_server_failure(self):
        from tests.inference.test_edge_stage_metrics import FakeNative
        runtime = FakeNative()
        instances = []
        engine = SimpleNamespace(_import_runtime=lambda: None)
        class FakeLlm:
            def __init__(self, **kwargs):
                self_outer.assertFalse(runtime.get_profiling_enabled())
                self.has_draft_model = False
                self._batch_scheduler = None
                self._rt = engine._import_runtime()
                self._runtime = runtime
                instances.append(self)
            def serve(self, **kwargs):
                self_outer.assertTrue(runtime.get_profiling_enabled())
                raise RuntimeError('server failed')
        self_outer = self
        engine.LLM = FakeLlm
        server = ModuleType('experimental.server')
        server.LLM = FakeLlm
        server.engine = engine
        server.api_server = SimpleNamespace(_generate_stream_sse=lambda llm_instance, response_id: iter(()),
                                            _create_app=lambda llm_instance: None)
        with patch.dict('sys.modules', {
            'experimental': ModuleType('experimental'), 'experimental.server': server,
            'uvicorn': ModuleType('uvicorn'),
        }), patch('parksight_vlm.inference.edge_native.load_native_binding', return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, 'server failed'):
                serve_prebuilt_engines(engine_root=Path('/engines'), host='127.0.0.1', port=8000,
                                       native_binding_directory=Path('/candidate'), native_binding_sha256='0'*64,
                                       serial_stage_metrics=True)
        self.assertFalse(runtime.get_profiling_enabled())
        self.assertIs(instances[0]._runtime, runtime)
        with self.assertRaises(ValueError):
            serve_prebuilt_engines(engine_root=Path('/engines'), host='127.0.0.1', port=8000,
                                   http_max_batch_size=2, serial_stage_metrics=True)


if __name__ == "__main__":
    unittest.main()
