"""Source identity and LF/CRLF application checks for the native timing bridge."""

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_edgellm_timing_binding import timing_binding_patch


SOURCE = '''#include "profiling/metrics.h"
void bind(py::module_& m)
{
    m.def("get_profiling_enabled", &getProfilingEnabled, "Check if profiling is currently enabled");
}
'''


class TimingBindingPatchTests(unittest.TestCase):
    def test_metrics_fix_applies_and_keeps_unrelated_bindings(self):
        names = ("LLMPrefillMetrics", "LLMGenerationMetrics",
                 "SpecDecodeGenerationMetrics", "MultimodalMetrics")
        source = (SOURCE + "\n".join(
            f'.def("get_total_runs", &metrics::{name}::getTotalRuns);'
            for name in names) + "\n").replace("\n", "\r\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "experimental/pybind/edgellm_pybind.cpp"
            target.parent.mkdir(parents=True)
            target.write_bytes(source)
            patch, expected = timing_binding_patch(
                source, hashlib.sha256(source).hexdigest(), fix_metrics_runs=True)
            patch_file = root / "timing.patch"
            patch_file.write_bytes(patch.encode())
            subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=root, check=True)
            subprocess.run(["git", "apply", str(patch_file)], cwd=root, check=True)
            updated = target.read_bytes()
            self.assertEqual(hashlib.sha256(updated).hexdigest(), expected)
            for name in names:
                self.assertIn(f'[](metrics::{name} const& self)'.encode(), updated)
            self.assertNotIn(b"\n", updated.replace(b"\r\n", b""))

    def test_metrics_fix_rejects_missing_derived_class_binding(self):
        source = SOURCE.encode()
        with self.assertRaisesRegex(ValueError, "run-count binding"):
            timing_binding_patch(source, hashlib.sha256(source).hexdigest(), fix_metrics_runs=True)

    def test_patch_applies_without_changing_existing_line_endings(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)), tempfile.TemporaryDirectory() as directory:
                source = SOURCE.replace("\n", newline).encode()
                patch, expected = timing_binding_patch(source, hashlib.sha256(source).hexdigest())
                root = Path(directory)
                target = root / "experimental/pybind/edgellm_pybind.cpp"
                target.parent.mkdir(parents=True)
                target.write_bytes(source)
                patch_file = root / "timing.patch"
                patch_file.write_bytes(patch.encode())
                result = subprocess.run(["git", "apply", "--check", str(patch_file)],
                                        cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                subprocess.run(["git", "apply", str(patch_file)], cwd=root, check=True)
                updated = target.read_bytes()
                self.assertEqual(hashlib.sha256(updated).hexdigest(), expected)
                self.assertEqual(updated.count(b"get_stage_timing_snapshot"), 1)
                if newline == "\r\n":
                    self.assertNotIn(b"\n", updated.replace(b"\r\n", b""))

    def test_rejects_source_drift_and_existing_extension(self):
        source = SOURCE.encode()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            timing_binding_patch(source, "0" * 64)
        for changed in (source + b"get_stage_timing_snapshot", source.replace(b"get_profiling_enabled", b"other")):
            with self.subTest(source=changed), self.assertRaises(ValueError):
                timing_binding_patch(changed, hashlib.sha256(changed).hexdigest())
