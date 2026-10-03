from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_edgellm_concurrency_sweep import build_runner_command
from scripts.run_edgellm_direct_benchmark import (
    build_command,
    parse_e2e_csv,
    run_direct_benchmark,
)


class DirectRuntimeBenchmarkTests(unittest.TestCase):
    def test_build_command_keeps_prefill_and_decode_contracts_separate(self) -> None:
        decode = build_command(
            llm_bench=Path("/opt/llm_bench"),
            engine_dir=Path("/engine/llm"),
            mode="decode",
            iterations=20,
            warmup=10,
            seed=0,
            output_dir=Path("/tmp/decode"),
            past_kv_len=768,
            osl=1,
        )
        self.assertIn("--pastKVLen", decode)
        self.assertNotIn("--inputLen", decode)
        self.assertEqual(decode[decode.index("--pastKVLen") + 1], "768")

        prefill = build_command(
            llm_bench=Path("/opt/llm_bench"),
            engine_dir=Path("/engine/llm"),
            mode="prefill",
            iterations=20,
            warmup=10,
            seed=0,
            output_dir=Path("/tmp/prefill"),
            input_len=768,
            reuse_kv_len=0,
        )
        self.assertIn("--inputLen", prefill)
        self.assertNotIn("--pastKVLen", prefill)

    def test_parse_e2e_csv_preserves_shape_and_numeric_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "e2e_decode_pastkvlen768.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "mode",
                        "batch_size",
                        "osl",
                        "e2e_time_ms",
                        "per_token_ms",
                        "throughput_tps",
                        "past_kv_len",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "mode": "decode",
                        "batch_size": "1",
                        "osl": "1",
                        "e2e_time_ms": "27.4",
                        "per_token_ms": "27.4",
                        "throughput_tps": "36.5",
                        "past_kv_len": "768",
                    }
                )
            parsed = parse_e2e_csv(path, expected_mode="decode")

        self.assertEqual(parsed["past_kv_len"], 768)
        self.assertEqual(parsed["e2e_time_ms"], 27.4)

    def test_direct_runner_writes_summary_compatible_jsonl_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_jsonl = root / "decode.jsonl"

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                output_dir = Path(command[command.index("--outputDir") + 1])
                csv_path = output_dir / "e2e_decode_pastkvlen768.csv"
                csv_path.write_text(
                    "mode,batch_size,osl,e2e_time_ms,per_token_ms,throughput_tps,past_kv_len\n"
                    "decode,1,1,27.4,27.4,36.5,768\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="CUDA graph enabled\n",
                    stderr="",
                )

            with patch("scripts.run_edgellm_direct_benchmark.subprocess.run", fake_run):
                metadata = run_direct_benchmark(
                    llm_bench=Path("/opt/llm_bench"),
                    engine_dir=Path("/engine/llm"),
                    mode="decode",
                    output_jsonl=output_jsonl,
                    repetitions=2,
                    iterations=20,
                    warmup=10,
                )

            rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["timings_ms"]["decode_ms"], 27.4)
            self.assertEqual(metadata["completed_runs"], 2)
            metadata_path = output_jsonl.with_suffix(".jsonl.metadata.json")
            self.assertTrue(metadata_path.is_file())

    def test_concurrency_command_carries_engine_capacity_and_connection_flags(self) -> None:
        command = build_runner_command(
            runner_script=Path("scripts/run_edgellm_benchmark.py"),
            manifest=Path("manifest.jsonl"),
            workload=Path("workload.json"),
            data_root=Path("data"),
            output_jsonl=Path("out.jsonl"),
            warmup_output_jsonl=Path("warmup.jsonl"),
            run_metadata_json=Path("run.json"),
            concurrency=4,
            split="test",
            limit=20,
            repetitions=3,
            warmup=1,
            endpoint="http://127.0.0.1:8000",
            model_name="local",
            timeout_seconds=120.0,
            no_stream_responses=False,
            reuse_http_connection=True,
            engine_max_batch_size=4,
            engine_max_kv_pool_pages=128,
        )
        self.assertIn("--concurrency", command)
        self.assertEqual(command[command.index("--concurrency") + 1], "4")
        self.assertIn("--engine-max-batch-size", command)
        self.assertIn("--reuse-http-connection", command)


if __name__ == "__main__":
    unittest.main()
