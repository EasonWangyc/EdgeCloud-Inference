"""应用配置、Runtime 构建与无标注分析测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from parksight_vlm.app import AppConfigError, AppStudyConfig, RuntimeConfig, build_runtime
from parksight_vlm.app.analyze_image import analyze_image, main
from parksight_vlm.inference import RuntimeGeneration, TransformersRuntime
from parksight_vlm.workload import FrozenWorkload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = FrozenWorkload.load(
    PROJECT_ROOT / "configs" / "workloads" / "parking_risk_v1.json"
)
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "inference" / "scene.jpg"


class StaticBackend:
    def generate(self, *, image_path: Path, workload: FrozenWorkload) -> RuntimeGeneration:
        return RuntimeGeneration(
            raw_output=json.dumps(
                {
                    "schema_version": "parking_risk_v1",
                    "risk_level": "low",
                    "events": [],
                    "evidence": ["The visible maneuver path is clear."],
                    "driver_advice": ["maintain_observation"],
                }
            )
        )


class AppTests(unittest.TestCase):
    def test_loads_all_repository_study_configs(self) -> None:
        transformers_config = AppStudyConfig.load(
            PROJECT_ROOT / "configs" / "studies" / "server" / "transformers_base.json"
        )
        jetson_transformers_config = AppStudyConfig.load(
            PROJECT_ROOT
            / "configs"
            / "studies"
            / "jetson" / "jetson_transformers_fp16.json"
        )
        edge_config = AppStudyConfig.load(
            PROJECT_ROOT / "configs" / "studies" / "jetson" / "edgellm_fp16.json"
        )
        edge_ps20_config = AppStudyConfig.load(
            PROJECT_ROOT
            / "configs"
            / "studies"
            / "jetson" / "jetson_edgellm_fp16_ps20_pilot.json"
        )
        candidate_lora_config = AppStudyConfig.load(
            PROJECT_ROOT
            / "configs"
            / "studies"
            / "server" / "server_transformers_lora_ps64_codex_candidate_v1_ps20_pilot.json"
        )
        strict_json_edge_config = AppStudyConfig.load(
            PROJECT_ROOT
            / "configs"
            / "studies"
            / "jetson" / "jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_strict_json.json"
        )
        reviewed_edge_config = AppStudyConfig.load(
            PROJECT_ROOT
            / "configs"
            / "studies"
            / "jetson" / "jetson_edgellm_lora_ps64_reviewed_v1_ps20_pilot.json"
        )
        reviewed_int4_edge_config = AppStudyConfig.load(
            PROJECT_ROOT
            / "configs"
            / "studies"
            / "jetson" / "jetson_edgellm_int4_awq_ps64_reviewed_v1_ps20_pilot_strict_json.json"
        )

        self.assertEqual(transformers_config.runtime.backend, "transformers")
        self.assertEqual(
            jetson_transformers_config.runtime.backend, "transformers"
        )
        self.assertEqual(jetson_transformers_config.runtime.precision, "fp16")
        self.assertEqual(
            jetson_transformers_config.runtime.backend_revision,
            "transformers==4.57.6",
        )
        self.assertEqual(
            jetson_transformers_config.study.power_mode, "15W_MODE_0"
        )
        self.assertEqual(edge_config.runtime.backend, "tensorrt_edge_llm_http")
        self.assertEqual(edge_config.study.workload.identity, WORKLOAD.identity)
        self.assertEqual(
            edge_config.runtime.backend_revision,
            "7f061f21f0a581ba234a1e233c9315b89d8e47d6",
        )
        self.assertEqual(edge_config.runtime.adapter_revision, "edge-http-v2")
        self.assertEqual(edge_config.study.power_mode, "15W_MODE_0")
        self.assertEqual(
            edge_ps20_config.runtime.backend_revision,
            "7f061f21f0a581ba234a1e233c9315b89d8e47d6",
        )
        self.assertEqual(edge_ps20_config.runtime.precision, "fp16")
        self.assertEqual(
            candidate_lora_config.runtime.adapter_revision,
            "ps64-codex-candidate-v1-r16",
        )
        self.assertEqual(
            candidate_lora_config.runtime.precision,
            "bf16+lora-candidate",
        )
        self.assertEqual(
            candidate_lora_config.runtime.options["adapter_path"],
            "artifacts/adapters/qwen3_vl_2b_parking_lora_ps64_codex_candidate_v1",
        )
        self.assertEqual(
            strict_json_edge_config.study.workload.workload_id,
            "parking_risk_v2_strict_json",
        )
        self.assertEqual(strict_json_edge_config.runtime.precision, "int4_awq")
        self.assertEqual(
            reviewed_edge_config.runtime.adapter_revision,
            "ps64-reviewed-v1-r16-merged",
        )
        self.assertEqual(reviewed_edge_config.runtime.precision, "fp16+lora-merged")
        self.assertEqual(
            reviewed_int4_edge_config.study.workload.workload_id,
            "parking_risk_v2_strict_json",
        )
        self.assertEqual(reviewed_int4_edge_config.runtime.precision, "int4_awq")
        self.assertEqual(
            reviewed_int4_edge_config.runtime.adapter_revision,
            "ps64-reviewed-v1-int4-awq",
        )
        self.assertEqual(edge_ps20_config.study.power_mode, "15W_MODE_0")
        self.assertEqual(
            edge_ps20_config.manifest_path.name,
            "ps20_pilot_v1.jsonl",
        )
        self.assertEqual(
            edge_ps20_config.annotations_path.name,
            "ps20_pilot_v1.jsonl",
        )

    def test_loads_candidate_80_server_study_configs(self) -> None:
        train_config = AppStudyConfig.load(
            PROJECT_ROOT
            / "configs"
            / "studies"
            / "server" / "server_transformers_lora_ps64_codex_candidate_v1_ps80_train_strict_json.json"
        )
        validation_config = AppStudyConfig.load(
            PROJECT_ROOT
            / "configs"
            / "studies"
            / "server" / "server_transformers_lora_ps64_codex_candidate_v1_ps80_validation_strict_json.json"
        )

        self.assertEqual(train_config.study.split.value, "train")
        self.assertEqual(validation_config.study.split.value, "validation")
        self.assertEqual(
            train_config.study.workload.workload_id,
            "parking_risk_v2_strict_json",
        )
        self.assertEqual(
            validation_config.study.workload.identity,
            train_config.study.workload.identity,
        )
        self.assertEqual(
            train_config.runtime.options["adapter_path"],
            "artifacts/adapters/qwen3_vl_2b_parking_lora_ps64_codex_candidate_v1",
        )
        self.assertEqual(
            validation_config.runtime.options["adapter_path"],
            train_config.runtime.options["adapter_path"],
        )

    def test_build_runtime_rejects_unknown_options(self) -> None:
        config = RuntimeConfig(
            backend="transformers",
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="revision",
            adapter_revision="none",
            precision="bf16",
            options={"unknown": True},
        )
        with self.assertRaisesRegex(AppConfigError, "unsupported transformers options"):
            build_runtime(config, data_root=FIXTURE_IMAGE.parent)

    @patch("parksight_vlm.app.runtime_factory.HuggingFaceQwen3VlBackend")
    def test_build_runtime_passes_optional_lora_adapter_path(
        self, backend_type: Mock
    ) -> None:
        config = RuntimeConfig(
            backend="transformers",
            backend_revision="5.9.0",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="immutable-model-commit",
            adapter_revision="adapter-v1",
            precision="bf16+lora",
            options={"adapter_path": "artifacts/adapters/adapter-v1"},
        )

        build_runtime(config, data_root=FIXTURE_IMAGE.parent)

        self.assertEqual(
            backend_type.call_args.kwargs["adapter_path"],
            "artifacts/adapters/adapter-v1",
        )

    def test_build_runtime_rejects_mutable_or_placeholder_revision(self) -> None:
        for revision in ("main", "replace-with-immutable-huggingface-commit"):
            config = RuntimeConfig(
                backend="transformers",
                backend_revision="test",
                model_id="Qwen/Qwen3-VL-2B-Instruct",
                model_revision=revision,
                adapter_revision="none",
                precision="bf16",
                options={},
            )
            with self.subTest(revision=revision), self.assertRaises(AppConfigError):
                build_runtime(config, data_root=FIXTURE_IMAGE.parent)

        backend_config = RuntimeConfig(
            backend="transformers",
            backend_revision="replace-with-installed-transformers-version",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="immutable-model-commit",
            adapter_revision="none",
            precision="bf16",
            options={},
        )
        with self.assertRaisesRegex(AppConfigError, "backend_revision"):
            build_runtime(backend_config, data_root=FIXTURE_IMAGE.parent)

    def test_single_image_analysis_does_not_require_reference_annotation(self) -> None:
        runtime = TransformersRuntime(
            data_root=FIXTURE_IMAGE.parent,
            backend=StaticBackend(),
            backend_revision="test",
            model_id="Qwen/Qwen3-VL-2B-Instruct",
            model_revision="test",
        )

        record = analyze_image(
            image_path=FIXTURE_IMAGE,
            runtime=runtime,
            workload=WORKLOAD,
        )

        self.assertTrue(record.succeeded)
        self.assertEqual(record.case_id, "scene")

    def test_single_image_cli_passes_transformers_loading_options(self) -> None:
        record = Mock(succeeded=True)
        record.to_mapping.return_value = {"succeeded": True}

        with (
            patch(
                "parksight_vlm.app.analyze_image.build_runtime",
                return_value=Mock(),
            ) as build_runtime_mock,
            patch(
                "parksight_vlm.app.analyze_image.analyze_image",
                return_value=record,
            ),
            patch("builtins.print"),
        ):
            exit_code = main(
                [
                    "--image",
                    str(FIXTURE_IMAGE),
                    "--runtime",
                    "transformers",
                    "--backend-revision",
                    "transformers==4.57.6",
                    "--model-revision",
                    "immutable-model-commit",
                    "--precision",
                    "fp16",
                    "--device-map",
                    "auto",
                    "--dtype",
                    "float16",
                    "--attn-implementation",
                    "sdpa",
                ]
            )

        runtime_config = build_runtime_mock.call_args.args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            runtime_config.options,
            {
                "device_map": "auto",
                "dtype": "float16",
                "attn_implementation": "sdpa",
            },
        )


if __name__ == "__main__":
    unittest.main()
