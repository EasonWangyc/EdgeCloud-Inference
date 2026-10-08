"""The native probe protects original engine metadata without copying large weights."""

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from scripts.run_edgellm_native_probe import construct_with_binding, digest, isolated_engine_view


class NativeProbeTests(unittest.TestCase):
    def test_binding_selection_reuses_module_and_restores_loader_after_success_or_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                original = lambda: "original"
                engine = SimpleNamespace(_import_runtime=original)
                candidate = object()
                def factory(**kwargs):
                    self.assertIs(engine._import_runtime(), candidate)
                    if fail:
                        raise RuntimeError("constructor failed")
                    return "llm"
                engine.LLM = factory
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                        construct_with_binding(engine, candidate)
                else:
                    self.assertEqual(construct_with_binding(engine, candidate), "llm")
                self.assertIs(engine._import_runtime, original)

    def test_engine_view_follows_visual_directory_alias_without_aliasing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidate"
            source.mkdir()
            shared = root / "shared_visual"
            shared.mkdir()
            (shared / "visual.engine").write_bytes(b"visual engine")
            (shared / "config.json").write_text('{"value": 1}')
            (source / "visual").symlink_to(shared, target_is_directory=True)
            view = root / "view"
            evidence = isolated_engine_view(source, view)
            self.assertEqual((view / "visual/visual.engine").read_bytes(), b"visual engine")
            (view / "visual/config.json").write_text('{"value": 2}')
            self.assertEqual((shared / "config.json").read_text(), '{"value": 1}')
            self.assertEqual(len(evidence), 2)

    def test_engine_view_rejects_directory_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "loop").symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "cyclic"):
                isolated_engine_view(source, root / "view")

    def test_engine_view_copies_metadata_and_keeps_binary_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            llm = source / "llm"
            llm.mkdir(parents=True)
            config = llm / "config.json"
            config.write_text('{"value": 1}')
            engine = llm / "llm.engine"
            engine.write_bytes(b"engine")
            weights = llm / "embedding.safetensors"
            weights.write_bytes(b"weights")
            view = root / "view"
            evidence = isolated_engine_view(source, view)
            (view / "llm/config.json").write_text('{"value": 2}')
            self.assertEqual(config.read_text(), '{"value": 1}')
            self.assertTrue((view / "llm/llm.engine").is_symlink())
            self.assertEqual(digest(view / "llm/llm.engine"), digest(engine))
            self.assertEqual({row["path"]: row["sha256"] for row in evidence},
                             {str(path): digest(path) for path in (config, engine, weights)})
