from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.audit_edgellm_kv_cache import KVCacheAuditError, audit_kv_cache_source


class EdgeLLMKVCacheAuditTests(unittest.TestCase):
    def test_reports_exposed_paged_path_without_modifying_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "edge-llm"
            (root / "examples/llm").mkdir(parents=True)
            (root / "cpp/runtime").mkdir(parents=True)
            _write(root / "examples/llm/llm_build.cpp", "int maxKVPoolPages = 0;\n")
            _write(
                root / "cpp/runtime/kvCacheManager.cpp",
                "int maxKVPoolPages = 0;\n",
            )
            _write(
                root / "cpp/runtime/llmInferenceRuntime.cpp",
                "bool usePagedKVCache = false;\nvoid run(pageList, tokensPerPage);\n",
            )
            _write(
                root / "cpp/runtime/pagePool.cpp",
                "allocatePage(); freePage(); pageTable = nullptr; pagePool = nullptr;\n",
            )
            (root / "cpp/plugins/attentionPlugin").mkdir(parents=True)
            _write(
                root / "cpp/plugins/attentionPlugin/attentionPlugin.cpp",
                "void enqueue(pageList, tokensPerPage);\n",
            )
            _git(root, "init")
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "TensorRT test")
            _git(root, "add", ".")
            _git(root, "commit", "-m", "base")
            revision = _git(root, "rev-parse", "HEAD").strip()

            report = audit_kv_cache_source(
                edge_llm_root=root, expected_revision=revision
            )

            self.assertEqual(
                report["summary"]["conclusion"],
                "paged_kv_path_exposed_requires_runtime_validation",
            )
            self.assertTrue(report["summary"]["builder_has_paged_kv_pool_option"])
            self.assertTrue(report["summary"]["runtime_has_paged_kv_symbols"])
            self.assertTrue(report["summary"]["pool_has_paged_kv_symbols"])
            self.assertTrue(report["summary"]["xqa_has_page_symbols"])
            self.assertTrue(report["summary"]["attention_plugin_has_paged_bindings"])
            self.assertTrue(report["summary"]["runtime_has_page_allocator"])
            self.assertTrue(report["summary"]["complete_source_path"])
            self.assertFalse(
                report["summary"]["attention_plugin_hardcodes_paged_kv_disabled"]
            )
            self.assertEqual(_git(root, "status", "--porcelain"), "")

    def test_detects_paged_path_not_wired_by_attention_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "edge-llm"
            (root / "examples/llm").mkdir(parents=True)
            (root / "cpp/runtime").mkdir(parents=True)
            (root / "cpp/plugins/attentionPlugin").mkdir(parents=True)
            _write(root / "examples/llm/llm_build.cpp", "")
            _write(root / "cpp/runtime/kvCacheManager.cpp", "tokensPerPage = 64;\n")
            _write(root / "cpp/runtime/llmInferenceRuntime.cpp", "pageList = nullptr;\n")
            _write(
                root / "cpp/plugins/attentionPlugin/attentionPlugin.cpp",
                "bool const usePagedKVCache = false;\n",
            )
            _git(root, "init")
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "TensorRT test")
            _git(root, "add", ".")
            _git(root, "commit", "-m", "base")

            report = audit_kv_cache_source(edge_llm_root=root)

            self.assertEqual(
                report["summary"]["conclusion"],
                "paged_kv_not_wired_in_checkout",
            )
            self.assertTrue(
                report["summary"]["attention_plugin_hardcodes_paged_kv_disabled"]
            )

    def test_rejects_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "edge-llm"
            (root / "examples/llm").mkdir(parents=True)
            (root / "cpp/runtime").mkdir(parents=True)
            for relative_path in (
                "examples/llm/llm_build.cpp",
                "cpp/runtime/kvCacheManager.cpp",
                "cpp/runtime/llmInferenceRuntime.cpp",
            ):
                _write(root / relative_path, "")
            _git(root, "init")
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "TensorRT test")
            _git(root, "add", ".")
            _git(root, "commit", "-m", "base")

            with self.assertRaisesRegex(KVCacheAuditError, "revision mismatch"):
                audit_kv_cache_source(edge_llm_root=root, expected_revision="0" * 40)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


if __name__ == "__main__":
    unittest.main()
