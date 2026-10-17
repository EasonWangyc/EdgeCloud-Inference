from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.diagnostics.analyze_edgellm_nsight_replay import analyze_replay


class NsightReplayTests(unittest.TestCase):
    def make_database(self, root):
        path = root / "trace.sqlite"
        with sqlite3.connect(path) as connection:
            connection.executescript("""
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INTEGER, correlationId INTEGER, returnValue INTEGER, nameId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid INTEGER, correlationId INTEGER, graphNodeId INTEGER, start INTEGER, end INTEGER, demangledName INTEGER);
                INSERT INTO StringIds VALUES(1, 'cudaGraphLaunch_v10000'), (2, 'kernel');
            """)
            pid = 83 << 24
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,1)",
                                   [(pid + 10, 100, 0), (pid + 11, 101, 0)])
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,2)",
                                   [(pid, 100, 7, 0, 1000000), (pid, 101, 8, 2000000, 3000000),
                                    (84 << 24, 100, 9, 0, 99000000)])
        return path

    def probe(self):
        return {"stage_before": {"llm_generation": {"run_count": 5}},
                "stage_after": {"llm_generation": {"run_count": 7}}}

    def test_correlations_include_process_namespace_and_use_counter_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_database(Path(directory))
            original = path.read_bytes()
            result = analyze_replay(path, self.probe())
            self.assertEqual(path.read_bytes(), original)
        self.assertTrue(result["replay_matches_native_steps"])
        self.assertEqual(result["graph_correlated_kernel_count"], 2)
        self.assertEqual(result["top_graph_kernels"][0]["total_gpu_event_ms"], 2.0)

    def test_failed_or_missing_graph_events_do_not_verify_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_database(Path(directory))
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1 WHERE correlationId=101")
            result = analyze_replay(path, self.probe())
            self.assertFalse(result["replay_matches_native_steps"])
            self.assertEqual(result["failed_graph_launch_count"], 1)
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=0")
                connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=101")
            result = analyze_replay(path, self.probe())
            self.assertFalse(result["replay_matches_native_steps"])
            self.assertEqual(len(result["launches_without_kernel_events"]), 1)

    def test_duplicate_launch_correlation_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_database(Path(directory))
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId=100")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                analyze_replay(path, self.probe())

    def test_empty_initial_timer_requires_fresh_native_counter_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_database(Path(directory))
            probe = self.probe()
            probe["stage_before"] = {}
            probe["stage_after"]["llm_generation"]["run_count"] = 2
            with self.assertRaisesRegex(ValueError, "verified fresh runtime"):
                analyze_replay(path, probe)
            probe["native_counts_before"] = {"generation_runs": False}
            with self.assertRaisesRegex(ValueError, "verified fresh runtime"):
                analyze_replay(path, probe)
            probe["native_counts_before"] = {"generation_runs": 0}
            self.assertTrue(analyze_replay(path, probe)["replay_matches_native_steps"])


if __name__ == "__main__":
    unittest.main()
