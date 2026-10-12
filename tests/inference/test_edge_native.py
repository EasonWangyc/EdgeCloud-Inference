"""Candidate identity and import lifetime checks without loading CUDA."""

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from parksight_vlm.inference.edge_native import binding_identity, load_native_binding


class NativeBindingTests(unittest.TestCase):
    def test_identity_rejects_changed_ambiguous_and_missing_modules(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sha = hashlib.sha256(b"candidate").hexdigest()
            with self.assertRaises(ValueError):
                binding_identity(root, sha)
            path = root / "_edgellm_runtime.test.so"
            path.write_bytes(b"candidate")
            self.assertEqual(binding_identity(root, sha), {"path": str(path), "sha256": sha})
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                binding_identity(root, sha)
            (root / "_edgellm_runtime.other.so").write_bytes(b"candidate")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                binding_identity(root, sha)

    def test_preloaded_module_must_match_verified_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "_edgellm_runtime.so"
            path.write_bytes(b"candidate")
            sha = hashlib.sha256(b"candidate").hexdigest()
            module = SimpleNamespace(__file__=str(path), _parksight_loaded_sha256=sha)
            with patch.dict(sys.modules, {"_edgellm_runtime": module}):
                self.assertIs(load_native_binding(root, sha), module)
            with patch.dict(sys.modules, {"_edgellm_runtime": SimpleNamespace(__file__=str(path))}):
                with self.assertRaisesRegex(RuntimeError, "unverified"):
                    load_native_binding(root, sha)
            with patch.dict(sys.modules, {"_edgellm_runtime": SimpleNamespace(__file__="/other/module.so")}):
                with self.assertRaisesRegex(RuntimeError, "fresh process"):
                    load_native_binding(root, sha)

    def test_import_failure_does_not_leave_partial_module_registered(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules):
            sys.modules.pop("_edgellm_runtime", None)
            root = Path(folder)
            (root / "_edgellm_runtime.so").write_bytes(b"candidate")
            sha = hashlib.sha256(b"candidate").hexdigest()
            loader = SimpleNamespace(exec_module=lambda module: (_ for _ in ()).throw(RuntimeError("import failed")))
            with patch("importlib.util.spec_from_file_location", return_value=SimpleNamespace(loader=loader)), \
                 patch("importlib.util.module_from_spec", return_value=SimpleNamespace()):
                with self.assertRaisesRegex(RuntimeError, "import failed"):
                    load_native_binding(root, sha)
            self.assertNotIn("_edgellm_runtime", sys.modules)
