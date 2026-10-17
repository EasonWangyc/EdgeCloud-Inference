"""人工复核 LoRA 数据和独立校准拆分测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from parksight_vlm.assessment import ParkingAssessment, ParkingRiskEvent
from parksight_vlm.workload import FrozenWorkload
from scripts.data.prepare_reviewed_lora_dataset import (
    enforce_event_coverage,
    prepare_datasets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = FrozenWorkload.load(
    PROJECT_ROOT / "configs" / "workloads" / "parking_risk_v1.json"
)


class ReviewedLoraDataTests(unittest.TestCase):
    def test_event_coverage_gate_rejects_missing_calibration_event(self) -> None:
        coverage = {
            "train": {event.value: 3 for event in ParkingRiskEvent},
            "validation": {event.value: 1 for event in ParkingRiskEvent},
            "calibration": {event.value: 1 for event in ParkingRiskEvent},
        }
        coverage["calibration"][ParkingRiskEvent.PARKING_SPACE_CONFLICT.value] = 0
        with self.assertRaisesRegex(ValueError, "calibration:parking_space_conflict=0"):
            enforce_event_coverage(coverage)

    def test_calibration_is_removed_from_supervised_splits(self) -> None:
        image_root = PROJECT_ROOT / "tests" / "fixtures" / "inference"
        teacher_records = []
        annotations = {}
        for index, split in enumerate(("train", "train", "validation"), start=1):
            sample_id = f"sample-{index}"
            assessment = ParkingAssessment.from_mapping(
                {
                    "schema_version": "parking_risk_v1",
                    "risk_level": "low",
                    "events": [],
                    "evidence": ["可见区域内未发现风险目标。"],
                    "driver_advice": ["maintain_observation"],
                }
            )
            annotations[sample_id] = assessment
            teacher_records.append(
                {
                    "sample_id": sample_id,
                    "image": "scene.jpg",
                    "source_group_id": f"group-{index}",
                    "split": split,
                    "assessment": assessment.to_mapping(),
                }
            )

        lora, calibration, summary = prepare_datasets(
            teacher_records=teacher_records,
            annotations=annotations,
            calibration_id="calibration-test",
            calibration_groups={"group-2"},
            frozen_test_groups={"frozen-group"},
            image_root=image_root,
            workload=WORKLOAD,
            label_source="human_confirmed_v1",
            dataset_id="human_confirmed_test",
        )

        self.assertEqual([record["split"] for record in lora], ["train", "validation"])
        self.assertEqual([record["split"] for record in calibration], ["calibration"])
        self.assertIn("助手输出：", calibration[0]["text"])
        self.assertEqual(summary["group_overlap"]["lora_calibration"], 0)
        self.assertEqual(summary["group_overlap"]["development_frozen_test"], 0)
        self.assertEqual(summary["label_source"], "human_confirmed_v1")
        self.assertEqual(summary["dataset_id"], "human_confirmed_test")
        self.assertEqual(lora[0]["label_source"], "human_confirmed_v1")


if __name__ == "__main__":
    unittest.main()
