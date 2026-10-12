from pathlib import Path
import tempfile
import unittest

from scripts.build_edgellm_vlm_engines import require_builder_controls


class BuilderControlTests(unittest.TestCase):
    def test_missing_compiled_controls_do_not_silently_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm_build"
            path.write_bytes(b"EDGELLM_BUILDER_OPT_LEVEL")
            with self.assertRaisesRegex(RuntimeError, "profiling control"):
                require_builder_controls(path, component="llm", profiling_verbosity="detailed", timing_cache=None)
            with self.assertRaisesRegex(RuntimeError, "TIMING_CACHE_PATH"):
                require_builder_controls(path, component="llm", profiling_verbosity=None, timing_cache=Path("cache"))
            self.assertIsNone(require_builder_controls(path, component="llm", profiling_verbosity=None, timing_cache=None))

    def test_control_name_across_read_boundary_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm_build"
            path.write_bytes(b"x" * (1024 * 1024 - 8) + b"EDGELLM_PROFILING_VERBOSITY\0EDGELLM_TIMING_CACHE_PATH")
            self.assertEqual(
                require_builder_controls(path, component="llm", profiling_verbosity="detailed", timing_cache=Path("cache")),
                "environment",
            )

    def test_native_llm_profiling_flag_needs_no_environment_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm_build"
            path.write_bytes(b"--profilingDetailed\0EDGELLM_TIMING_CACHE_PATH")
            self.assertEqual(
                require_builder_controls(path, component="llm", profiling_verbosity="detailed", timing_cache=Path("cache")),
                "native_cli",
            )
            with self.assertRaisesRegex(RuntimeError, "profiling control"):
                require_builder_controls(path, component="visual", profiling_verbosity="detailed", timing_cache=None)
            with self.assertRaisesRegex(RuntimeError, "profiling control"):
                require_builder_controls(path, component="llm", profiling_verbosity="none", timing_cache=None)


if __name__ == "__main__":
    unittest.main()
