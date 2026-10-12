"""Preserve launch and signal failures without guessing an OOM cause."""

import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.build_edgellm_vlm_engines import _run_build_step


class EngineBuildFailureTests(unittest.TestCase):
    def run_failure(self, error):
        report = {"steps": []}
        process = SimpleNamespace(pid=12345, wait=lambda: error.returncode) if isinstance(error, subprocess.CalledProcessError) else None
        with patch("scripts.build_edgellm_vlm_engines.subprocess.Popen", return_value=process,
                   side_effect=error if isinstance(error, OSError) else None):
            with self.assertRaises(type(error)):
                _run_build_step(component="llm", command=["llm_build"], environment={}, report=report)
        self.assertEqual(len(report["steps"]), 1)
        step = report["steps"][0]
        self.assertEqual(step["status"], "failed")
        self.assertGreaterEqual(step["duration_ms"], 0)
        self.assertIsNone(step["oom_confirmed"])
        return step

    def test_sigkill_is_signal_failure_and_not_automatically_oom(self):
        step = self.run_failure(subprocess.CalledProcessError(-signal.SIGKILL, ["llm_build"]))
        self.assertEqual(step["returncode"], -9)
        self.assertEqual(step["termination_signal"], "SIGKILL")
        self.assertEqual(step["termination_signal_number"], 9)
        self.assertEqual(step["failure_category"], "process_signal")
        self.assertEqual(step["pid"], 12345)

    def test_nonzero_exit_and_missing_executable_have_distinct_facts(self):
        step = self.run_failure(subprocess.CalledProcessError(2, ["llm_build"]))
        self.assertEqual(step["failure_category"], "builder_nonzero_exit")
        self.assertIsNone(step["termination_signal"])
        step = self.run_failure(FileNotFoundError("builder was removed"))
        self.assertEqual(step["failure_category"], "launch_error")
        self.assertIsNone(step["returncode"])
        self.assertEqual(step["error_type"], "FileNotFoundError")
        self.assertIsNone(step["pid"])

    def test_success_keeps_normal_completed_step(self):
        report = {"steps": []}
        with patch("scripts.build_edgellm_vlm_engines.subprocess.Popen",
                   return_value=SimpleNamespace(pid=12345, wait=lambda: 0)):
            _run_build_step(component="llm", command=["llm_build"], environment={}, report=report)
        self.assertEqual(report["steps"][0]["status"], "succeeded")
        self.assertNotIn("failure_category", report["steps"][0])

    def test_real_child_pid_is_preserved_for_later_kernel_oom_audit(self):
        report = {"steps": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pid.txt"
            command = [sys.executable, "-c",
                       "import os,sys;from pathlib import Path;Path(sys.argv[1]).write_text(str(os.getpid()));sys.exit(2)",
                       str(path)]
            with self.assertRaises(subprocess.CalledProcessError):
                _run_build_step(component="llm", command=command, environment={}, report=report)
            self.assertEqual(report["steps"][0]["pid"], int(path.read_text()))
        self.assertEqual(report["steps"][0]["returncode"], 2)
        self.assertIsNone(report["steps"][0]["oom_confirmed"])
