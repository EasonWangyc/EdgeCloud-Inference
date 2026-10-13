import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_edgellm_concurrency_runs import compare


def write_run(root: Path, name: str, *, concurrency: int, output: str = "same") -> tuple[Path, Path]:
    jsonl = root / f"{name}.jsonl"
    metadata = root / f"{name}.run.json"
    row = {
        "sample_id": "sample-1", "repetition": 1, "status": "completed",
        "output_tokens": 3,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }
    jsonl.write_text(json.dumps(row) + "\n")
    metadata.write_text(json.dumps({
        "cases": 1, "repetitions": 1, "warmup": 1,
        "concurrency": concurrency, "max_batch_size": 4,
        "max_kv_pool_pages": None, "stream_responses": True,
        "reuse_http_connection": True, "steady_state_requests": 1,
        "steady_state_completed_requests": 1, "steady_state_output_tokens": 3,
        "steady_state_wall_clock_ms": 150.0 if concurrency == 1 else 100.0,
        "aggregate_output_tokens_per_second": 20.0 if concurrency == 1 else 30.0,
    }))
    return jsonl, metadata


class ConcurrencyAbaComparisonTests(unittest.TestCase):
    def test_exact_output_and_throughput_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = write_run(root, "before", concurrency=1)
            candidate = write_run(root, "candidate", concurrency=4)
            after = write_run(root, "after", concurrency=1)
            result = compare(
                before_jsonl=before[0], before_metadata=before[1],
                candidate_jsonl=candidate[0], candidate_metadata=candidate[1],
                after_jsonl=after[0], after_metadata=after[1],
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["candidate_to_baseline_mean_ratio"], 1.5)
            self.assertEqual(result["concurrency_order"], [1, 4, 1])
            self.assertEqual(result["exact_output_mismatches"], [])

    def test_same_token_count_different_text_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = write_run(root, "before", concurrency=1)
            candidate = write_run(root, "candidate", concurrency=4, output="else")
            after = write_run(root, "after", concurrency=1)
            result = compare(
                before_jsonl=before[0], before_metadata=before[1],
                candidate_jsonl=candidate[0], candidate_metadata=candidate[1],
                after_jsonl=after[0], after_metadata=after[1],
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(len(result["exact_output_mismatches"]), 1)

    def test_missing_hash_or_failed_request_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = write_run(root, "before", concurrency=1)
            candidate = write_run(root, "candidate", concurrency=4)
            after = write_run(root, "after", concurrency=1)
            row = json.loads(candidate[0].read_text())
            row.pop("output_sha256")
            candidate[0].write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "output_sha256"):
                compare(
                    before_jsonl=before[0], before_metadata=before[1],
                    candidate_jsonl=candidate[0], candidate_metadata=candidate[1],
                    after_jsonl=after[0], after_metadata=after[1],
                )

    def test_slower_candidate_does_not_pass_even_when_output_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = write_run(root, "before", concurrency=1)
            candidate = write_run(root, "candidate", concurrency=4)
            after = write_run(root, "after", concurrency=1)
            metadata = json.loads(candidate[1].read_text())
            metadata["steady_state_wall_clock_ms"] = 300.0
            metadata["aggregate_output_tokens_per_second"] = 10.0
            candidate[1].write_text(json.dumps(metadata))
            result = compare(
                before_jsonl=before[0], before_metadata=before[1],
                candidate_jsonl=candidate[0], candidate_metadata=candidate[1],
                after_jsonl=after[0], after_metadata=after[1],
            )
            self.assertTrue(result["exact_output_equal"])
            self.assertFalse(result["candidate_faster_than_both_baselines"])
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
