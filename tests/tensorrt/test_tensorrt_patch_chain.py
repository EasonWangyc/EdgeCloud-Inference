from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostics.check_tensorrt_patch_chain import PatchChainError, check_patch_chain


class TensorRTPatchChainTests(unittest.TestCase):
    def test_checks_patches_in_sequence_without_changing_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "edge-llm"
            root.mkdir()
            _git(root, "init")
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "TensorRT test")

            source = root / "kernel.txt"
            source.write_text("value=0\n", encoding="utf-8")
            _git(root, "add", "kernel.txt")
            _git(root, "commit", "-m", "base")
            revision = _git(root, "rev-parse", "HEAD").strip()

            patch_one = Path(temporary_directory) / "001.patch"
            source.write_text("value=1\n", encoding="utf-8")
            patch_one.write_text(_git(root, "diff"), encoding="utf-8")
            source.write_text("value=0\n", encoding="utf-8")

            patch_two = Path(temporary_directory) / "002.patch"
            source.write_text("value=1\n", encoding="utf-8")
            _git(root, "add", "kernel.txt")
            source.write_text("value=2\n", encoding="utf-8")
            patch_two.write_text(_git(root, "diff"), encoding="utf-8")
            source.write_text("value=0\n", encoding="utf-8")
            _git(root, "reset", "--mixed", "HEAD")

            report = check_patch_chain(
                edge_llm_root=root,
                patches=(patch_one, patch_two),
                expected_revision=revision,
            )

            self.assertEqual(report["status"], "succeeded")
            self.assertEqual(report["edge_llm_revision"], revision)
            self.assertEqual(report["applied_in_order"], ["001.patch", "002.patch"])
            self.assertEqual(source.read_text(encoding="utf-8"), "value=0\n")
            self.assertEqual(_git(root, "status", "--porcelain"), "")

    def test_rejects_revision_mismatch_before_creating_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "edge-llm"
            root.mkdir()
            _git(root, "init")
            source = root / "kernel.txt"
            source.write_text("value=0\n", encoding="utf-8")
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "TensorRT test")
            _git(root, "add", "kernel.txt")
            _git(root, "commit", "-m", "base")
            patch = Path(temporary_directory) / "001.patch"
            patch.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(PatchChainError, "revision mismatch"):
                check_patch_chain(
                    edge_llm_root=root,
                    patches=(patch,),
                    expected_revision="0" * 40,
                )


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
