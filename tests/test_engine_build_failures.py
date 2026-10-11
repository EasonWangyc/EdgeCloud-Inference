"""Preserve launch and signal failures without guessing an OOM cause."""

import signal
import subprocess
import unittest
from unittest.mock import patch

from scripts.build_edgellm_vlm_engines import _run_build_step


class EngineBuildFailureTests(unittest.TestCase):
    def run_failure(self, error):
        report = {"steps": []}
        with patch("scripts.build_edgellm_vlm_engines.subprocess.run", side_effect=error):
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

    def test_nonzero_exit_and_missing_executable_have_distinct_facts(self):
        step = self.run_failure(subprocess.CalledProcessError(2, ["llm_build"]))
        self.assertEqual(step["failure_category"], "builder_nonzero_exit")
        self.assertIsNone(step["termination_signal"])
        step = self.run_failure(FileNotFoundError("builder was removed"))
        self.assertEqual(step["failure_category"], "launch_error")
        self.assertIsNone(step["returncode"])
        self.assertEqual(step["error_type"], "FileNotFoundError")

    def test_success_keeps_normal_completed_step(self):
        report = {"steps": []}
        with patch("scripts.build_edgellm_vlm_engines.subprocess.run"):
            _run_build_step(component="llm", command=["llm_build"], environment={}, report=report)
        self.assertEqual(report["steps"][0]["status"], "succeeded")
        self.assertNotIn("failure_category", report["steps"][0])
