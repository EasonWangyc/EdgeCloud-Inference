"""扩大 validation split 工具测试。"""

from __future__ import annotations

import unittest

from scripts.data.rebalance_validation_split import rebalance_manifest, rebalance_records


class ValidationRebalanceTests(unittest.TestCase):
    def test_rebalance_preserves_records_and_target_count(self) -> None:
        records = [
            {
                "sample_id": f"sample-{index}",
                "source_group_id": f"group-{index}",
                "split": "validation" if index < 2 else "train",
                "assessment": {
                    "risk_level": "high" if index == 5 else "low",
                    "events": ["vru_near_maneuver_path"] if index == 5 else [],
                },
            }
            for index in range(8)
        ]
        updated, audit = rebalance_records(
            records,
            target_validation_count=4,
            seed=20260830,
        )

        self.assertEqual(len(updated), 8)
        self.assertEqual(sum(row["split"] == "validation" for row in updated), 4)
        self.assertEqual(audit["protected_events"], ["vru_near_maneuver_path"])
        self.assertEqual(
            {row["sample_id"] for row in updated if row["split"] == "validation"}
            & {"sample-5"},
            set(),
        )

    def test_rebalance_rejects_non_train_validation_records(self) -> None:
        records = [
            {
                "sample_id": "sample-1",
                "source_group_id": "group-1",
                "split": "calibration",
                "assessment": {"risk_level": "low", "events": []},
            },
            {
                "sample_id": "sample-2",
                "source_group_id": "group-2",
                "split": "train",
                "assessment": {"risk_level": "low", "events": []},
            },
        ]
        with self.assertRaisesRegex(ValueError, "only train and validation"):
            rebalance_records(records, target_validation_count=1, seed=1)

    def test_manifest_keeps_records_outside_lora_dataset(self) -> None:
        dataset = [
            {
                "sample_id": "sample-1",
                "source_group_id": "group-1",
                "split": "validation",
            }
        ]
        manifest = [
            {"case_id": "sample-1", "split": "train"},
            {"case_id": "calibration-1", "split": "train"},
        ]

        updated = rebalance_manifest(manifest, dataset)

        self.assertEqual(updated[0]["split"], "validation")
        self.assertEqual(updated[1]["split"], "train")


if __name__ == "__main__":
    unittest.main()
