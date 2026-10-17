from __future__ import annotations

import unittest
from pathlib import Path

from parksight_vlm.flows import ExternalFlowPlan, FlowValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_flow(path: Path) -> ExternalFlowPlan:
    """Resolve a named flow from its stage directory or the historical archive."""
    if path.parent == REPOSITORY_ROOT / "configs" / "flows":
        matches = sorted(path.parent.glob(f"*/{path.name}"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one grouped flow for {path}: {matches}")
        path = matches[0]
    return ExternalFlowPlan.load(path)


class ExternalFlowPlanTests(unittest.TestCase):
    def test_loads_every_stage_template_with_stable_identity(self) -> None:
        expected_stages = {
            "train_lora.example.json": "train_lora",
            "merge_lora.example.json": "merge_lora",
            "export_model.example.json": "export_model",
            "build_engine.example.json": "build_engine",
        }
        for filename, expected_stage in expected_stages.items():
            with self.subTest(filename=filename):
                path = REPOSITORY_ROOT / "configs" / "flows" / filename
                first = _load_flow(path)
                second = _load_flow(path)
                self.assertEqual(first.stage, expected_stage)
                self.assertEqual(first.identity, second.identity)
                self.assertEqual(len(first.identity), 64)
                self.assertFalse(first.readiness_mapping()["ready"])

    def test_loads_pinned_qwen3_vl_int4_awq_flows(self) -> None:
        expected = {
            "quantize_qwen3_vl_2b_int4_awq.json": "quantize_model",
            "export_qwen3_vl_2b_int4_awq.json": "export_model",
            "build_qwen3_vl_2b_int4_awq_llm_engine_i768_k1024.json": (
                "build_engine"
            ),
        }

        plans = {
            filename: _load_flow(
                REPOSITORY_ROOT / "configs" / "flows" / filename
            )
            for filename in expected
        }
        for filename, stage in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual(plans[filename].stage, stage)
                self.assertNotIn("replace-with-", " ".join(plans[filename].command))

        quantize_command = plans[
            "quantize_qwen3_vl_2b_int4_awq.json"
        ].command
        self.assertIn("int4_awq", quantize_command)
        self.assertIn("128", quantize_command)
        self.assertNotIn("--visual_quantization", quantize_command)
        self.assertNotIn("--lm_head_quantization", quantize_command)

        build_command = plans[
            "build_qwen3_vl_2b_int4_awq_llm_engine_i768_k1024.json"
        ].command
        self.assertIn("768", build_command)
        self.assertIn("1024", build_command)
        self.assertNotIn("--enable-weight-streaming", build_command)

    def test_loads_explicit_lm_head_quantization_candidate_flow(self) -> None:
        plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "quantize_qwen3_vl_2b_int4_awq_lm_head_candidate.json"
        )
        self.assertEqual(plan.stage, "quantize_model")
        self.assertIn("--lm-head-quantization", plan.command)
        self.assertIn("int4_awq", plan.command)
        self.assertIn("lm_head_candidate", " ".join(plan.command))

    def test_loads_lm_head_candidate_export_and_build_flows(self) -> None:
        export_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "export_qwen3_vl_2b_int4_awq_lm_head_candidate.json"
        )
        build_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_int4_awq_lm_head_candidate_llm_i768_k1024.json"
        )
        self.assertEqual(export_plan.stage, "export_model")
        self.assertNotIn("--reduced-vocab-dir", export_plan.command)
        self.assertTrue(
            any(path.name == "calibration_provenance.json" for path in export_plan.required_inputs)
        )
        self.assertEqual(build_plan.stage, "build_engine")
        self.assertIn("--builder-optimization-level", build_plan.command)
        self.assertIn("1", build_plan.command)
        self.assertIn("1024", build_plan.command)
        self.assertIn("--quantization-provenance", build_plan.command)
        self.assertIn("--expected-lm-head-precision", build_plan.command)
        self.assertIn("int4_awq", build_plan.command)
        self.assertTrue(
            any(path.name == "calibration_provenance.json" for path in build_plan.required_inputs)
        )

    def test_loads_lm_head_and_reduced_vocabulary_compound_candidate(self) -> None:
        for size in ("32768", "65536"):
            with self.subTest(size=size):
                export_plan = _load_flow(
                    REPOSITORY_ROOT
                    / "configs"
                    / "flows"
                    / "archive"
                    / f"export_qwen3_vl_2b_int4_awq_lm_head_reduced_vocab{size}_history.json"
                )
                build_plan = _load_flow(
                    REPOSITORY_ROOT
                    / "configs"
                    / "flows"
                    / "archive"
                    / f"build_qwen3_vl_2b_int4_awq_lm_head_reduced_vocab{size}_history_llm_i768_k1024_opt1.json"
                )

                export_command = " ".join(export_plan.command)
                build_command = " ".join(build_plan.command)
                self.assertEqual(export_plan.stage, "export_model")
                self.assertIn("lm_head_candidate", export_command)
                self.assertIn("--reduced-vocab-dir", export_command)
                self.assertIn(f"reduced_vocab{size}", export_command)
                self.assertIn("plus_history", export_command)
                self.assertTrue(
                    any(path.name == "calibration_provenance.json" for path in export_plan.required_inputs)
                )
                self.assertTrue(
                    any(path.name == "selection_report.json" for path in export_plan.required_inputs)
                )

                self.assertEqual(build_plan.stage, "build_engine")
                self.assertIn("--reduced-vocab-coverage-report", build_command)
                self.assertIn("--quantization-provenance", build_command)
                self.assertIn("--expected-lm-head-precision int4_awq", build_command)
                self.assertIn("--builder-optimization-level 1", build_command)
                self.assertTrue(
                    any(path.name == "calibration_provenance.json" for path in build_plan.required_inputs)
                )
                self.assertTrue(
                    any(path.name == "selection_report.json" for path in build_plan.required_inputs)
                )
                self.assertTrue(
                    any(
                        path.name == f"reduced_vocab_{size}_ps20_character_coverage_plus_history.json"
                        for path in build_plan.required_inputs
                    )
                )

    def test_loads_reduced_vocabulary_export_and_build_flows(self) -> None:
        export_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "export_qwen3_vl_2b_int4_awq_reduced_vocab65536.json"
        )
        build_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_int4_awq_reduced_vocab65536_llm_i768_k1024_opt1.json"
        )
        self.assertEqual(export_plan.stage, "export_model")
        self.assertIn("--reduced-vocab-dir", export_plan.command)
        self.assertTrue(
            any(path.name == "vocab_map.safetensors" for path in export_plan.required_inputs)
        )
        self.assertTrue(
            any(path.name == "selection_report.json" for path in export_plan.required_inputs)
        )
        self.assertEqual(build_plan.stage, "build_engine")
        self.assertIn("--builder-optimization-level", build_plan.command)
        self.assertIn("1", build_plan.command)
        self.assertTrue(
            any(path.name == "vocab_map.safetensors" for path in build_plan.required_inputs)
        )
        self.assertTrue(
            any(path.name == "selection_report.json" for path in build_plan.required_inputs)
        )
        self.assertFalse(
            any(
                "/onnx/" in path.as_posix() and path.name == "selection_report.json"
                for path in build_plan.required_inputs
            )
        )
        self.assertIn("--reduced-vocab-coverage-report", build_plan.command)
        self.assertTrue(
            any(path.name == "reduced_vocab_65536_ps20_character_coverage.json" for path in build_plan.required_inputs)
        )

        smaller_export_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "export_qwen3_vl_2b_int4_awq_reduced_vocab32768.json"
        )
        smaller_build_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_int4_awq_reduced_vocab32768_llm_i768_k1024_opt1.json"
        )
        self.assertIn("--reduced-vocab-dir", smaller_export_plan.command)
        self.assertIn("32768", " ".join(smaller_export_plan.command))
        self.assertIn("32768", " ".join(smaller_build_plan.command))
        self.assertIn("--reduced-vocab-coverage-report", smaller_build_plan.command)

        for size in ("32768", "65536"):
            with self.subTest(size=size):
                history_export = _load_flow(
                    REPOSITORY_ROOT
                    / "configs"
                    / "flows"
                    / "archive"
                    / f"export_qwen3_vl_2b_int4_awq_reduced_vocab{size}_history.json"
                )
                history_build = _load_flow(
                    REPOSITORY_ROOT
                    / "configs"
                    / "flows"
                    / "archive"
                    / f"build_qwen3_vl_2b_int4_awq_reduced_vocab{size}_history_llm_i768_k1024_opt1.json"
                )
                self.assertEqual(history_export.stage, "export_model")
                self.assertEqual(history_build.stage, "build_engine")
                self.assertIn("plus_history", " ".join(history_export.command))
                self.assertIn("plus_history", " ".join(history_build.command))
                self.assertIn(
                    f"reduced_vocab_{size}_ps20_character_coverage_plus_history.json",
                    " ".join(history_build.command),
                )
                self.assertIn(
                    f"reduced_vocab_{size}_ps20_character_coverage_plus_history.json",
                    " ".join(str(path) for path in history_build.required_inputs),
                )

    def test_loads_executable_qwen3_vl_lora_flows(self) -> None:
        expected = {
            "train_qwen3_vl_2b_lora_ps80_v1.json": "train_lora",
            "merge_qwen3_vl_2b_lora_ps80_v1.json": "merge_lora",
            "export_qwen3_vl_2b_lora_ps80_v1.json": "export_model",
            "build_qwen3_vl_2b_lora_ps80_v1_llm_engine_i768_k1024.json": (
                "build_engine"
            ),
        }

        for filename, stage in expected.items():
            with self.subTest(filename=filename):
                plan = _load_flow(
                    REPOSITORY_ROOT / "configs" / "flows" / filename
                )
                self.assertEqual(plan.stage, stage)
                self.assertNotIn("replace-with-", " ".join(plan.command))

    def test_loads_dynamic_batch4_int4_engine_flow(self) -> None:
        plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_dynamic_batch4.json"
        )
        self.assertEqual(plan.stage, "build_engine")
        command = " ".join(plan.command)
        self.assertIn("--max-batch-size 4", command)
        self.assertIn("--builder-optimization-level 1", command)
        self.assertIn("--max-input-len 768", command)
        self.assertIn("--max-kv-cache-capacity 1024", command)
        self.assertTrue(any("dynamic_batch4" in path.as_posix() for path in plan.expected_outputs))

    def test_loads_paged_kv_candidate_flow(self) -> None:
        plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_paged_kv.json"
        )
        self.assertEqual(plan.stage, "build_engine")
        command = " ".join(plan.command)
        self.assertIn("--max-kv-pool-pages 16", command)
        self.assertIn("--verify-paged-kv-source", command)
        self.assertIn("--max-batch-size 1", command)
        self.assertIn("--builder-optimization-level 1", command)
        self.assertTrue(any("paged_kv" in path.as_posix() for path in plan.expected_outputs))

    def test_loads_dynamic_batch_paged_kv_candidate_flow(self) -> None:
        plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_dynamic_batch4_paged_kv.json"
        )
        command = " ".join(plan.command)
        self.assertIn("--max-batch-size 4", command)
        self.assertIn("--max-kv-pool-pages 32", command)
        self.assertIn("--verify-paged-kv-source", command)
        self.assertIn("--builder-optimization-level 1", command)
        self.assertTrue(any("dynamic_batch4_paged_kv" in path.as_posix() for path in plan.expected_outputs))

    def test_paged_builds_consume_matching_staged_onnx(self) -> None:
        for batch_size, pool_pages, suffix, build_name in (
            (1, 16, "pagedkv16", "build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_paged_kv"),
            (4, 32, "pagedkv32", "build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_dynamic_batch4_paged_kv"),
        ):
            with self.subTest(batch_size=batch_size):
                prepare = _load_flow(
                    REPOSITORY_ROOT / "configs" / "flows"
                    / f"prepare_qwen3_vl_2b_int4_awq_n128_{suffix}.json"
                )
                build = _load_flow(
                    REPOSITORY_ROOT / "configs" / "flows" / f"{build_name}.json"
                )
                self.assertEqual(prepare.stage, "export_model")
                self.assertIn(f"--max-batch-size {batch_size}", " ".join(prepare.command))
                self.assertIn(f"--max-kv-pool-pages {pool_pages}", " ".join(prepare.command))
                self.assertIn(f"--max-batch-size {batch_size}", " ".join(build.command))
                self.assertIn(f"--max-kv-pool-pages {pool_pages}", " ".join(build.command))
                staged_root = f"artifacts/onnx/qwen3_vl_2b_int4_awq_n128_{suffix}"
                self.assertIn(f"--destination-root {staged_root}", " ".join(prepare.command))
                self.assertIn(f"--onnx-root {staged_root}", " ".join(build.command))
                self.assertTrue(any(path.name == "paged_onnx_provenance.json" for path in build.required_inputs))
                self.assertTrue(any(path.name == "audit_edgellm_kv_cache.py" for path in build.required_inputs))
                self.assertIn(
                    REPOSITORY_ROOT / "scripts" / "experiments" / "prepare_edgellm_paged_onnx.py",
                    build.required_inputs,
                )
                self.assertTrue(any(path.name == "libNvInfer_edgellm_plugin.so" for path in build.required_inputs))
                staged_inputs = {
                    path for path in build.required_inputs if staged_root in path.as_posix()
                }
                self.assertEqual(staged_inputs, set(prepare.expected_outputs))

    def test_detailed_paged_candidate_is_isolated_from_primary_build(self) -> None:
        directory = REPOSITORY_ROOT / "configs" / "flows"
        primary = _load_flow(
            directory / "build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_dynamic_batch4_paged_kv.json"
        )
        detailed = _load_flow(
            directory / "build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_dynamic_batch4_paged_kv_detailed.json"
        )
        self.assertIn("--profiling-verbosity detailed", " ".join(detailed.command))
        self.assertNotIn("--profiling-verbosity", primary.command)
        self.assertNotEqual(primary.expected_outputs, detailed.expected_outputs)
        self.assertEqual(primary.required_inputs, detailed.required_inputs)
        self.assertEqual(
            primary.command[primary.command.index("--expected-source-fingerprint") + 1],
            detailed.command[detailed.command.index("--expected-source-fingerprint") + 1],
        )
        for flag in ("--expected-llm-builder-sha256", "--expected-plugin-sha256"):
            self.assertEqual(
                primary.command[primary.command.index(flag) + 1],
                detailed.command[detailed.command.index(flag) + 1],
            )

    def test_loads_visual_builder_level_candidates(self) -> None:
        for level in ("2", "3"):
            with self.subTest(level=level):
                plan = _load_flow(
                    REPOSITORY_ROOT
                    / "configs"
                    / "flows"
                    / f"build_qwen3_vl_2b_fp16_visual_engine_opt{level}.json"
                )
                command = " ".join(plan.command)
                self.assertEqual(plan.stage, "build_engine")
                self.assertIn("--component visual", command)
                self.assertIn(f"--builder-optimization-level {level}", command)
                self.assertIn("--workspace-limit-mib 1024", command)
                self.assertTrue(any("visual_opt" + level in path.as_posix() for path in plan.expected_outputs))

    def test_loads_reduced_vocab_dynamic_batch4_compound_flow(self) -> None:
        plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "archive"
            / "build_qwen3_vl_2b_int4_awq_reduced_vocab65536_history_llm_i768_k1024_opt1_dynamic_batch4.json"
        )
        self.assertEqual(plan.stage, "build_engine")
        command = " ".join(plan.command)
        self.assertIn("--max-batch-size 4", command)
        self.assertIn("--reduced-vocab-dir", command)
        self.assertIn("--reduced-vocab-required-token-id 151645", command)
        self.assertIn("--reduced-vocab-required-token-id 151643", command)
        self.assertIn("plus_history", command)
        self.assertTrue(any("dynamic_batch4" in path.as_posix() for path in plan.expected_outputs))

    def test_loads_explicit_codex_candidate_training_flow(self) -> None:
        plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "train_qwen3_vl_2b_lora_ps64_codex_candidate_v1.json"
        )
        self.assertEqual(plan.stage, "train_lora")
        self.assertIn("codex_candidate", " ".join(plan.command))
        self.assertTrue(
            any(path.name == "ps64_codex_candidate_v1.jsonl" for path in plan.required_inputs)
        )
        self.assertTrue(
            any("codex_candidate_v1" in path.as_posix() for path in plan.expected_outputs)
        )

    def test_loads_formal_post_review_ps64_flows(self) -> None:
        expected = {
            "export_qwen3_vl_2b_lora_ps64_reviewed_v1.json": "export_model",
            "quantize_qwen3_vl_2b_lora_ps64_reviewed_v1.json": "quantize_model",
            "export_qwen3_vl_2b_lora_ps64_reviewed_v1_int4_awq.json": "export_model",
            "build_qwen3_vl_2b_lora_ps64_reviewed_v1_fp16_llm_engine_i768_k1024.json": "build_engine",
            "build_qwen3_vl_2b_lora_ps64_reviewed_v1_int4_awq_llm_engine_i768_k1024.json": "build_engine",
        }
        plans = {
            filename: _load_flow(
                REPOSITORY_ROOT / "configs" / "flows" / filename
            )
            for filename in expected
        }

        for filename, stage in expected.items():
            with self.subTest(filename=filename):
                plan = plans[filename]
                self.assertEqual(plan.stage, stage)
                self.assertNotIn("replace-with-", " ".join(plan.command))
                self.assertFalse(plan.readiness_mapping()["ready"])

        quantize_plan = plans["quantize_qwen3_vl_2b_lora_ps64_reviewed_v1.json"]
        self.assertIn("ps16_human_confirmed_v1.jsonl", " ".join(quantize_plan.command))
        self.assertIn(
            "int4_awq",
            " ".join(str(path) for path in quantize_plan.expected_outputs),
        )

        fp16_build = plans[
            "build_qwen3_vl_2b_lora_ps64_reviewed_v1_fp16_llm_engine_i768_k1024.json"
        ]
        int4_build = plans[
            "build_qwen3_vl_2b_lora_ps64_reviewed_v1_int4_awq_llm_engine_i768_k1024.json"
        ]
        self.assertIn("--enable-weight-streaming", fp16_build.command)
        self.assertNotIn("--enable-weight-streaming", int4_build.command)
        self.assertIn("768", fp16_build.command)
        self.assertIn("1024", int4_build.command)

    def test_loads_pinned_qwen3_vl_fp16_deployment_flows(self) -> None:
        expected = {
            "export_qwen3_vl_2b_fp16.json": "export_model",
            "build_qwen3_vl_2b_fp16_engines.json": "build_engine",
            "build_qwen3_vl_2b_fp16_llm_engine.json": "build_engine",
            "build_qwen3_vl_2b_fp16_visual_engine.json": "build_engine",
        }

        for filename, stage in expected.items():
            with self.subTest(filename=filename):
                plan = _load_flow(
                    REPOSITORY_ROOT / "configs" / "flows" / filename
                )
                serialized_command = " ".join(plan.command)
                self.assertEqual(plan.stage, stage)
                self.assertNotIn("replace-with-", serialized_command)
                self.assertIn("qwen3_vl_2b_fp16", serialized_command.lower())

        build_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_fp16_engines.json"
        )
        self.assertIn(
            "7f061f21f0a581ba234a1e233c9315b89d8e47d6",
            build_plan.command,
        )
        export_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "export_qwen3_vl_2b_fp16.json"
        )
        export_outputs = {path.as_posix() for path in export_plan.expected_outputs}
        export_inputs = {path.as_posix() for path in export_plan.required_inputs}
        build_outputs = {path.as_posix() for path in build_plan.expected_outputs}
        self.assertTrue(
            any(path.endswith("/model.safetensors") for path in export_inputs)
        )
        self.assertFalse(
            any(path.endswith("/model.safetensors.index.json") for path in export_inputs)
        )
        self.assertTrue(
            any(path.endswith("/llm/tokenizer.json") for path in export_outputs)
        )
        self.assertTrue(
            any(path.endswith("/visual/config.json") for path in export_outputs)
        )
        self.assertTrue(
            any(path.endswith("/llm/tokenizer.json") for path in build_outputs)
        )
        self.assertTrue(
            any(path.endswith("/visual/config.json") for path in build_outputs)
        )

        llm_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_fp16_llm_engine.json"
        )
        visual_plan = _load_flow(
            REPOSITORY_ROOT
            / "configs"
            / "flows"
            / "build_qwen3_vl_2b_fp16_visual_engine.json"
        )
        self.assertIn("llm", llm_plan.command)
        self.assertIn("visual", visual_plan.command)
        self.assertTrue(
            all("/visual/" not in path.as_posix() for path in llm_plan.expected_outputs)
        )
        self.assertTrue(
            all("/llm/" not in path.as_posix() for path in visual_plan.expected_outputs)
        )

    def test_rejects_unknown_fields_and_unsafe_path_overlap(self) -> None:
        valid = {
            "flow_id": "flow",
            "stage": "train_lora",
            "working_directory": ".",
            "command": ["trainer"],
            "required_inputs": ["input.json"],
            "expected_outputs": ["output.json"],
            "record_path": "record.json",
            "log_path": "flow.log",
        }
        with self.assertRaises(FlowValidationError):
            ExternalFlowPlan.from_mapping(
                {**valid, "unexpected": True}, config_root=REPOSITORY_ROOT
            )
        with self.assertRaises(FlowValidationError):
            ExternalFlowPlan.from_mapping(
                {**valid, "record_path": "output.json"},
                config_root=REPOSITORY_ROOT,
            )


if __name__ == "__main__":
    unittest.main()
