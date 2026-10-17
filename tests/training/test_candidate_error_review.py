from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "data"
    / "build_candidate_error_review.py"
)
SPEC = importlib.util.spec_from_file_location("build_candidate_error_review", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidateErrorReviewTests(unittest.TestCase):
    def test_build_review_aligns_records_and_prioritizes_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            annotations = root / "annotations.jsonl"
            report = root / "report.json"
            _write_jsonl(
                manifest,
                [
                    {
                        "case_id": "case-1",
                        "image_ref": "images/1.jpg",
                        "source_group_id": "group-1",
                        "split": "train",
                    },
                    {
                        "case_id": "case-2",
                        "image_ref": "images/2.jpg",
                        "source_group_id": "group-2",
                        "split": "validation",
                    },
                ],
            )
            _write_jsonl(
                annotations,
                [
                    {"case_id": "case-1", "assessment": {"risk_level": "high", "events": ["narrow_passage"]}},
                    {"case_id": "case-2", "assessment": {"risk_level": "low", "events": []}},
                ],
            )
            report.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "case_id": "case-1",
                                "assessment": {"risk_level": "high", "events": []},
                                "failure": None,
                                "raw_output": "{}",
                            },
                            {
                                "case_id": "case-2",
                                "assessment": None,
                                "failure": {"category": "json_parse_error"},
                                "raw_output": "not-json",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = MODULE.build_error_review(
                manifest_path=manifest,
                annotations_path=annotations,
                study_report_paths=[report],
            )

            self.assertEqual(result["summary"]["sample_count"], 2)
            self.assertEqual(result["summary"]["json_failure_count"], 1)
            self.assertEqual(result["summary"]["event_error_count"], 1)
            self.assertEqual(result["summary"]["failure_counts"], {"json_parse_error": 1})
            self.assertEqual(result["summary"]["review_priority_counts"], {"high": 2})
            self.assertEqual(result["summary"]["event_false_negative"], {"narrow_passage": 1})

    def test_build_review_rejects_duplicate_study_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            annotations = root / "annotations.jsonl"
            report = root / "report.json"
            row = {
                "case_id": "case-1",
                "image_ref": "images/1.jpg",
                "source_group_id": "group-1",
                "split": "train",
            }
            _write_jsonl(manifest, [row])
            _write_jsonl(
                annotations,
                [{"case_id": "case-1", "assessment": {"risk_level": "low", "events": []}}],
            )
            report.write_text(
                json.dumps(
                    {
                        "records": [
                            {"case_id": "case-1", "assessment": {}, "failure": None},
                            {"case_id": "case-1", "assessment": {}, "failure": None},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate study record"):
                MODULE.build_error_review(
                    manifest_path=manifest,
                    annotations_path=annotations,
                    study_report_paths=[report],
                )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
