"""Hardware-free checks for the paged Edge-LLM build preflight."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import build_edgellm_vlm_engines as build_module
from scripts.build_edgellm_vlm_engines import (
    _validate_staged_paged_graph, main, validate_paged_engine_config,
)


class EdgeLLMBuildPreflightTests(unittest.TestCase):
    def test_paged_graph_validation_loads_without_external_weights(self) -> None:
        fake_model = object()
        loader = Mock(return_value=fake_model)
        fake_onnx = SimpleNamespace(
            load=loader,
            TensorProto=SimpleNamespace(INT32=6),
        )
        with patch.dict("sys.modules", {"onnx": fake_onnx}), \
                patch("scripts.experiments.prepare_edgellm_paged_onnx.validate_paged_graph_contract",
                      return_value={"attention_plugin_count": 1}) as validator:
            result = _validate_staged_paged_graph(
                Path("model.onnx"), attention_count=1, max_batch_size=4,
                max_pages_per_sequence=8, max_kv_pool_pages=32,
            )
        self.assertEqual(result["attention_plugin_count"], 1)
        loader.assert_called_once_with("model.onnx", load_external_data=False)
        validator.assert_called_once_with(
            fake_model, attention_count=1, max_batch_size=4,
            max_pages_per_sequence=8,
            max_kv_pool_pages=32, int32_type=6,
        )

        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        with patch.object(build_module, "__package__", None), \
                patch.object(sys, "path", [str(scripts_dir), *sys.path]), \
                patch.dict("sys.modules", {"onnx": fake_onnx}):
            with patch("experiments.prepare_edgellm_paged_onnx.validate_paged_graph_contract",
                       return_value={"attention_plugin_count": 1}) as direct_validator:
                self.assertEqual(_validate_staged_paged_graph(
                    Path("model.onnx"), attention_count=1, max_batch_size=4,
                    max_pages_per_sequence=8, max_kv_pool_pages=32,
                )["attention_plugin_count"], 1)
            direct_validator.assert_called_once()

    def test_serialized_paged_config_must_match_requested_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            builder = {
                "kv_cache_layout": "paged",
                "tokens_per_page": 128,
                "max_batch_size": 4,
                "max_input_len": 768,
                "max_kv_cache_capacity": 1024,
                "max_kv_pool_pages": 32,
            }
            path.write_text(json.dumps({"builder_config": builder}), encoding="utf-8")
            expected = {
                "max_batch_size": 4,
                "max_input_len": 768,
                "max_kv_cache_capacity": 1024,
                "max_kv_pool_pages": 32,
            }
            result = validate_paged_engine_config(path, **expected)
            self.assertEqual(result["builder_config"]["kv_cache_layout"], "paged")
            self.assertEqual(result["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

            for key, wrong_value in (
                ("kv_cache_layout", "contiguous"),
                ("tokens_per_page", 64),
                ("max_kv_pool_pages", 16),
            ):
                with self.subTest(field=key):
                    invalid = {**builder, key: wrong_value}
                    path.write_text(json.dumps({"builder_config": invalid}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, key):
                        validate_paged_engine_config(path, **expected)

    def test_paged_preflight_checks_inputs_without_creating_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            edge = root / "edge"
            edge.mkdir()
            subprocess.run(["git", "-C", str(edge), "init", "-q"], check=True)
            subprocess.run(
                ["git", "-C", str(edge), "-c", "user.name=Test", "-c",
                 "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "base"],
                check=True,
            )
            revision = subprocess.check_output(
                ["git", "-C", str(edge), "rev-parse", "HEAD"], text=True
            ).strip()
            builder = edge / "build/examples/llm/llm_build"
            plugin = edge / "build/libNvInfer_edgellm_plugin.so"
            builder.parent.mkdir(parents=True)
            builder.write_bytes(b"builder")
            plugin.write_bytes(b"plugin")

            stage = root / "paged-onnx"
            llm = stage / "llm"
            llm.mkdir(parents=True)
            graph = llm / "model.onnx"
            weights = llm / "model.onnx.data"
            config = llm / "config.json"
            graph.write_bytes(b"AttentionPlugin enable_paged_kv kv_page_table kv_page_table")
            weights.write_bytes(b"weights")
            config.write_text('{"num_hidden_layers": 1}', encoding="utf-8")
            stage_report = {
                "schema_version": "parksight_paged_onnx_stage_v1",
                "tokens_per_page": 128,
                "max_pages_per_sequence": 1,
                "max_kv_pool_pages": 1,
                "max_batch_size": 1,
                "attention_plugin_count": 1,
                "staged_files_sha256": {
                    "llm/model.onnx": hashlib.sha256(graph.read_bytes()).hexdigest(),
                    "llm/model.onnx.data": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "llm/config.json": hashlib.sha256(config.read_bytes()).hexdigest(),
                },
            }
            (stage / "paged_onnx_provenance.json").write_text(
                json.dumps(stage_report), encoding="utf-8"
            )
            engine = root / "engine"
            build_report = root / "build-report.json"
            arguments = [
                "--edge-llm-root", str(edge), "--expected-revision", revision,
                "--onnx-root", str(stage), "--engine-root", str(engine),
                "--component", "llm", "--max-batch-size", "1",
                "--max-kv-cache-capacity", "128", "--max-kv-pool-pages", "1",
                "--verify-paged-kv-source", "--expected-source-fingerprint", "a" * 64,
                "--expected-llm-builder-sha256", hashlib.sha256(b"builder").hexdigest(),
                "--expected-plugin-sha256", hashlib.sha256(b"plugin").hexdigest(),
                "--build-report", str(build_report),
                "--check-only",
            ]
            audit = {
                "summary": {"conclusion": "paged_kv_path_exposed_requires_runtime_validation"},
                "source_fingerprint": {"sha256": "a" * 64, "file_count": 1},
            }
            validator_patcher = patch(
                "scripts.build_edgellm_vlm_engines._validate_staged_paged_graph",
                return_value={"attention_plugin_count": 1, "kv_cache_input_count": 1,
                              "kv_cache_output_count": 1},
            )
            graph_validator = validator_patcher.start()
            self.addCleanup(validator_patcher.stop)
            output = io.StringIO()
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(arguments), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["status"], "ready_for_build")
            self.assertEqual(result["paged_onnx_stage"]["attention_plugin_count"], 1)
            self.assertEqual(result["paged_onnx_stage"]["graph_validation"]["kv_cache_input_count"], 1)
            self.assertEqual(result["paged_kv_source_audit"]["source_fingerprint"]["sha256"], "a" * 64)
            self.assertEqual(result["plugin_binary"]["sha256"], hashlib.sha256(b"plugin").hexdigest())
            self.assertFalse(engine.exists())
            self.assertFalse(build_report.exists())

            direct_audit = SimpleNamespace(audit_kv_cache_source=Mock(return_value=audit))
            scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
            with patch.object(build_module, "__package__", None), \
                    patch.object(sys, "path", [str(scripts_dir), *sys.path]), \
                    patch.dict("sys.modules", {"diagnostics.audit_edgellm_kv_cache": direct_audit}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            direct_audit.audit_kv_cache_source.assert_called_once()
            self.assertFalse(engine.exists())

            config.write_text('{"num_hidden_layers": 2}', encoding="utf-8")
            stage_report["staged_files_sha256"]["llm/config.json"] = hashlib.sha256(
                config.read_bytes()
            ).hexdigest()
            (stage / "paged_onnx_provenance.json").write_text(
                json.dumps(stage_report), encoding="utf-8"
            )
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with self.assertRaisesRegex(ValueError, "differs from num_hidden_layers"):
                    main(arguments)
            config.write_text('{"num_hidden_layers": 1}', encoding="utf-8")
            stage_report["staged_files_sha256"]["llm/config.json"] = hashlib.sha256(
                config.read_bytes()
            ).hexdigest()
            (stage / "paged_onnx_provenance.json").write_text(
                json.dumps(stage_report), encoding="utf-8"
            )

            extra = llm / "stale.bin"
            extra.write_bytes(b"unlisted")
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with self.assertRaisesRegex(ValueError, "unlisted=.*stale.bin"):
                    main(arguments)
            extra.unlink()

            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            extra.symlink_to(outside)
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with self.assertRaisesRegex(ValueError, "contains symlinks"):
                    main(arguments)
            extra.unlink()
            self.assertFalse(engine.exists())

            without_audit = arguments.copy()
            without_audit.remove("--verify-paged-kv-source")
            index = without_audit.index("--expected-source-fingerprint")
            del without_audit[index:index + 2]
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    main(without_audit)
            self.assertEqual(error.exception.code, 2)
            self.assertIn("requires --verify-paged-kv-source", stderr.getvalue())
            self.assertFalse(engine.exists())

            graph_validator.side_effect = ValueError("staged ONNX kv_page_table dtype or shape is invalid")
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with self.assertRaisesRegex(ValueError, "kv_page_table dtype or shape"):
                    main(arguments)
            self.assertFalse(engine.exists())
            graph_validator.side_effect = None

            for binary, original, expected_error in (
                (builder, b"builder", "llm builder binary changed"),
                (plugin, b"plugin", "plugin binary changed"),
            ):
                with self.subTest(binary=binary.name):
                    binary.write_bytes(b"changed")
                    with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                        with self.assertRaisesRegex(ValueError, expected_error):
                            main(arguments)
                    self.assertFalse(engine.exists())
                    self.assertFalse(build_report.exists())
                    binary.write_bytes(original)

            (engine / "llm").mkdir(parents=True)
            (engine / "llm" / "config.json").write_text("stale", encoding="utf-8")
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with self.assertRaisesRegex(FileExistsError, "非空 engine 输出目录"):
                    main(arguments)
            self.assertFalse(build_report.exists())
            (engine / "llm" / "config.json").unlink()

            build_report.write_text("previous build evidence", encoding="utf-8")
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with self.assertRaisesRegex(FileExistsError, "build report"):
                    main(arguments)
                with self.assertRaisesRegex(FileExistsError, "build report"):
                    main([arg for arg in arguments if arg != "--check-only"])
            self.assertEqual(build_report.read_text(), "previous build evidence")
            build_report.unlink()
            (engine / "llm").rmdir()
            engine.rmdir()

            drifted_arguments = arguments.copy()
            drifted_arguments[drifted_arguments.index("--expected-source-fingerprint") + 1] = "b" * 64
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with self.assertRaisesRegex(ValueError, "source fingerprint changed"):
                    main(drifted_arguments)
            self.assertFalse(engine.exists())

            graph.write_bytes(graph.read_bytes() + b"changed")
            with patch("scripts.diagnostics.audit_edgellm_kv_cache.audit_kv_cache_source", return_value=audit):
                with self.assertRaisesRegex(ValueError, "changed or is missing"):
                    main(arguments)
            self.assertFalse(engine.exists())


if __name__ == "__main__":
    unittest.main()
