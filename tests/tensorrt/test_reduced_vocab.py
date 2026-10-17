from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.diagnostics.evaluate_reduced_vocab_coverage import evaluate_coverage
from scripts.diagnostics.validate_reduced_vocab import validate_mapping


def _write_safetensors(path: Path, values: list[int]) -> None:
    payload = struct.pack(f"<{len(values)}i", *values)
    header = json.dumps(
        {
            "vocab_map": {
                "dtype": "I32",
                "shape": [len(values)],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


class ReducedVocabularyValidationTests(unittest.TestCase):
    def test_coverage_can_include_successful_study_report_outputs(self) -> None:
        class Tokenizer:
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                del add_special_tokens
                return [ord(char) for char in text]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "vocab_map.safetensors"
            metadata_path = root / "reduced_vocab.json"
            _write_safetensors(map_path, [97, 98])
            metadata_path.write_text(
                json.dumps({"vocab_size": 256, "reduced_vocab_size": 2}),
                encoding="utf-8",
            )
            report_path = root / "study.json"
            report_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {"case_id": "ok", "raw_output": "ab"},
                            {"case_id": "failed", "raw_output": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = evaluate_coverage(
                tokenizer=Tokenizer(),
                map_path=map_path,
                metadata_path=metadata_path,
                original_vocab_size=256,
                input_paths=[],
                fields=["assessment"],
                study_report_paths=[report_path],
            )

        self.assertTrue(result["valid"])
        self.assertEqual(result["reference_sample_count"], 1)
        self.assertEqual(result["study_reports"], [str(report_path)])
        self.assertEqual(result["total_reference_tokens"], 2)

    def test_coverage_reports_missing_reference_tokens(self) -> None:
        class Tokenizer:
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                del add_special_tokens
                return [ord(char) for char in text]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "vocab_map.safetensors"
            metadata_path = root / "reduced_vocab.json"
            _write_safetensors(map_path, [97, 98])
            metadata_path.write_text(
                json.dumps({"vocab_size": 256, "reduced_vocab_size": 2}),
                encoding="utf-8",
            )
            input_path = root / "holdout.jsonl"
            input_path.write_text(
                json.dumps({"assessment": {"evidence": ["abx"]}}) + "\n",
                encoding="utf-8",
            )
            result = evaluate_coverage(
                tokenizer=Tokenizer(),
                map_path=map_path,
                metadata_path=metadata_path,
                original_vocab_size=256,
                input_paths=[input_path],
                fields=["assessment"],
            )

        self.assertFalse(result["valid"])
        self.assertEqual(result["samples_with_missing_tokens"], 1)
        self.assertEqual(result["missing_token_counts_by_sample"], {"120": 1})
        self.assertAlmostEqual(result["token_coverage_fraction"], 2 / 3)

    def test_accepts_sorted_map_and_required_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "vocab_map.safetensors"
            metadata_path = root / "reduced_vocab.json"
            _write_safetensors(map_path, [1, 3, 8, 151645])
            metadata_path.write_text(
                json.dumps({"vocab_size": 151936, "reduced_vocab_size": 4}),
                encoding="utf-8",
            )

            result = validate_mapping(
                map_path=map_path,
                metadata_path=metadata_path,
                original_vocab_size=151936,
                required_token_ids=(8, 151645),
            )

        self.assertTrue(result["valid"])
        self.assertEqual(result["map"]["count"], 4)
        self.assertEqual(result["missing_required_token_ids"], [])

    def test_packed_int4_lm_head_requires_group_size_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "vocab_map.safetensors"
            metadata_path = root / "reduced_vocab.json"
            _write_safetensors(map_path, [1, 3, 8, 151645])
            metadata_path.write_text(
                json.dumps({"vocab_size": 151936, "reduced_vocab_size": 4}),
                encoding="utf-8",
            )

            result = validate_mapping(
                map_path=map_path,
                metadata_path=metadata_path,
                original_vocab_size=151936,
                packed_int4_lm_head=True,
            )

        self.assertFalse(result["valid"])
        self.assertIn("multiple of 128", result["reasons"][0])
        self.assertEqual(result["packed_int4_group_size"], 128)

    def test_rejects_unsorted_map_and_missing_required_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "vocab_map.safetensors"
            metadata_path = root / "reduced_vocab.json"
            _write_safetensors(map_path, [3, 1, 200000])
            metadata_path.write_text(
                json.dumps({"vocab_size": 151936, "reduced_vocab_size": 3}),
                encoding="utf-8",
            )

            result = validate_mapping(
                map_path=map_path,
                metadata_path=metadata_path,
                original_vocab_size=151936,
                required_token_ids=(8,),
            )

        self.assertFalse(result["valid"])
        self.assertIn("map contains a token id outside the original vocabulary", result["reasons"])
        self.assertIn("map token ids must be strictly increasing", result["reasons"])
        self.assertIn(8, result["missing_required_token_ids"])


if __name__ == "__main__":
    unittest.main()
