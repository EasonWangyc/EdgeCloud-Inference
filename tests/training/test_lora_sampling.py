import unittest
import json
import tempfile
from pathlib import Path

from scripts.finetune_qwen3_vl_lora import (
    enforce_training_event_coverage,
    _oversample_training_records,
    _oversample_non_low_records,
    _validate_label_provenance,
    validate_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "inference" / "scene.jpg"
WORKLOAD = PROJECT_ROOT / "configs" / "workloads" / "parking_risk_v1.json"


class LoraSamplingTests(unittest.TestCase):
    def test_event_coverage_gate_rejects_missing_event_support(self) -> None:
        coverage = {
            "train": {"narrow_passage": 3, "vehicle_near_maneuver_path": 2},
            "validation": {"narrow_passage": 1, "vehicle_near_maneuver_path": 1},
        }
        with self.assertRaisesRegex(ValueError, "event coverage gate failed"):
            enforce_training_event_coverage(coverage, min_train_event_support=3)

    def test_event_coverage_gate_can_require_validation_support(self) -> None:
        coverage = {
            "train": {"narrow_passage": 3},
            "validation": {"narrow_passage": 0},
        }
        with self.assertRaisesRegex(ValueError, "validation:narrow_passage=0"):
            enforce_training_event_coverage(coverage, min_train_event_support=3)

    def test_event_oversampling_increases_event_exposure(self) -> None:
        no_event = {
            "sample_id": "no-event",
            "assessment": {"risk_level": "low", "events": []},
        }
        event = {
            "sample_id": "event",
            "assessment": {"risk_level": "medium", "events": ["narrow_passage"]},
        }
        result = _oversample_training_records([no_event, event], 2, 3)
        self.assertEqual(
            [record["sample_id"] for record in result],
            ["no-event", "event", "event", "event"],
        )

    def test_event_oversampling_rejects_invalid_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "event_oversampling_factor"):
            _oversample_training_records([], 1, 0)

    def test_oversamples_only_non_low_records(self) -> None:
        low = {"sample_id": "low", "assessment": {"risk_level": "low"}}
        medium = {
            "sample_id": "medium",
            "assessment": {"risk_level": "medium"},
        }
        result = _oversample_non_low_records([low, medium], factor=2)
        self.assertEqual(
            [record["sample_id"] for record in result],
            ["low", "medium", "medium"],
        )

    def test_rejects_invalid_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            _oversample_non_low_records([], factor=0)

    def test_training_requires_human_confirmed_source(self) -> None:
        records = [{"label_source": "codex_visual_review_v1_single_pass"}]
        with self.assertRaisesRegex(ValueError, "candidate Codex labels are blocked"):
            _validate_label_provenance(
                records,
                {
                    "label_source": "codex_visual_review_v1_single_pass",
                    "allow_candidate_labels": False,
                },
            )

    def test_training_rejects_mixed_or_mismatched_sources(self) -> None:
        records = [
            {"label_source": "human_confirmed_v1"},
            {"label_source": "codex_visual_review_v1_single_pass"},
        ]
        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_label_provenance(
                records, {"label_source": "human_confirmed_v1"}
            )

    def test_training_accepts_human_confirmed_source(self) -> None:
        records = [{"label_source": "human_confirmed_v1"}]
        self.assertEqual(
            _validate_label_provenance(
                records, {"label_source": "human_confirmed_v1"}
            ),
            "human_confirmed_v1",
        )

    def test_validate_training_config_checks_inputs_without_cuda(self) -> None:
        assessment = {
            "schema_version": "parking_risk_v1",
            "risk_level": "low",
            "events": [],
            "evidence": ["可见区域内未发现风险目标。"],
            "driver_advice": ["maintain_observation"],
        }
        train_assessment = {**assessment, "risk_level": "medium"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset.jsonl"
            model = root / "model"
            model.mkdir()
            rows = [
                {
                    "image": str(FIXTURE_IMAGE),
                    "split": "train",
                    "label_source": "human_confirmed_v1",
                    "assessment": train_assessment,
                },
                {
                    "image": str(FIXTURE_IMAGE),
                    "split": "validation",
                    "label_source": "human_confirmed_v1",
                    "assessment": assessment,
                },
            ]
            dataset.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            result = validate_training_config(
                {
                    "dataset_path": str(dataset),
                    "workload_path": str(WORKLOAD),
                    "model_path": str(model),
                    "label_source": "human_confirmed_v1",
                    "allow_candidate_labels": False,
                    "non_low_oversampling_factor": 2,
                }
            )

        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["effective_train_samples"], 2)
        self.assertEqual(result["validation_samples"], 1)
        self.assertEqual(result["event_coverage"]["train"]["narrow_passage"], 0)
        self.assertIn(
            {"type": "missing_from_train", "event": "narrow_passage"},
            result["event_coverage"]["warnings"],
        )

    def test_validate_training_config_rejects_candidate_before_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "dataset.jsonl"
            model = root / "model"
            model.mkdir()
            row = {
                "image": str(FIXTURE_IMAGE),
                "split": "train",
                "label_source": "codex_visual_review_v1_single_pass",
                "assessment": {
                    "schema_version": "parking_risk_v1",
                    "risk_level": "low",
                    "events": [],
                    "evidence": ["未见风险目标。"],
                    "driver_advice": ["maintain_observation"],
                },
            }
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate Codex labels are blocked"):
                validate_training_config(
                    {
                        "dataset_path": str(dataset),
                        "workload_path": str(WORKLOAD),
                        "model_path": str(model),
                        "label_source": "codex_visual_review_v1_single_pass",
                        "allow_candidate_labels": False,
                    }
                )


if __name__ == "__main__":
    unittest.main()
