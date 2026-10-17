from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.experiments.build_reduced_vocab_map import (
    collect_token_counts,
    collect_decoded_character_coverage_token_ids,
    collect_study_report_token_counts,
    select_token_ids,
    write_reduced_vocab_artifacts,
)


class _FakeTokenizer:
    eos_token_id = 99
    bos_token_id = 98
    pad_token_id = None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [ord(char) % 16 for char in text]


class ReducedVocabularyBuilderTests(unittest.TestCase):
    def test_collects_successful_study_report_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "study.json"
            report_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {"raw_output": "ab"},
                            {"raw_output": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            counts = collect_study_report_token_counts(
                _FakeTokenizer(), [report_path]
            )

        self.assertEqual(counts[ord("a") % 16], 1)
        self.assertEqual(sum(counts.values()), 2)

    def test_collects_decoded_cjk_and_json_character_tokens(self) -> None:
        class Tokenizer:
            def get_vocab(self) -> dict[str, int]:
                return {"zh": 1, "json": 2, "plain": 3}

            def decode(
                self,
                token_ids: list[int],
                *,
                clean_up_tokenization_spaces: bool,
                skip_special_tokens: bool,
            ) -> str:
                del clean_up_tokenization_spaces, skip_special_tokens
                return {1: "障", 2: "{", 3: "plain"}[token_ids[0]]

        self.assertEqual(
            collect_decoded_character_coverage_token_ids(Tokenizer()),
            {1, 2},
        )

    def test_selects_frequency_then_required_and_filler(self) -> None:
        token_ids, report = select_token_ids(
            Counter({4: 10, 7: 5, 8: 5, 12: 1}),
            target_size=128,
            original_vocab_size=256,
            required_token_ids=(200, 4),
        )

        self.assertEqual(len(token_ids), 128)
        self.assertEqual(token_ids, sorted(token_ids))
        self.assertIn(200, token_ids)
        self.assertIn(4, token_ids)
        self.assertEqual(report["observed_selected_count"], 4)
        self.assertEqual(report["observed_selection_fraction"], 0.03125)
        self.assertGreater(report["filler_token_count"], 0)
        self.assertIsNotNone(report["quality_warning"])

    def test_collects_nested_annotation_strings_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "annotations.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "assessment": {
                            "evidence": ["ab"],
                            "driver_advice": ["cd"],
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            counts = collect_token_counts(
                _FakeTokenizer(), [input_path], ["assessment"]
            )
            self.assertGreater(counts[ord("a") % 16], 0)

            token_ids, report = select_token_ids(
                counts,
                target_size=128,
                original_vocab_size=256,
                required_token_ids=(99,),
            )
            paths = write_reduced_vocab_artifacts(
                root / "reduced_vocab",
                token_ids,
                original_vocab_size=256,
                selection_report={**report, "tokenizer": {"path": str(root)}},
            )

            self.assertTrue(paths["map"].is_file())
            self.assertTrue(paths["metadata"].is_file())
            self.assertTrue(paths["report"].is_file())
            self.assertEqual(
                json.loads(paths["metadata"].read_text(encoding="utf-8")),
                {"vocab_size": 256, "reduced_vocab_size": 128},
            )
            selection_report = json.loads(
                paths["report"].read_text(encoding="utf-8")
            )
            self.assertEqual(selection_report["tokenizer"], {"path": str(root)})

    def test_file_identity_is_stable_and_contains_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text("{\"assessment\": \"ok\"}\n", encoding="utf-8")

            from scripts.experiments.build_reduced_vocab_map import _file_identity

            identity = _file_identity(path)
            self.assertEqual(identity["path"], str(path.resolve()))
            self.assertEqual(identity["size_bytes"], path.stat().st_size)
            self.assertEqual(len(identity["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
