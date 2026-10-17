import unittest

from scripts.data.select_calibration_groups import select_calibration_groups


def _record(group_id: str, risk: str, events: list[str]) -> dict[str, object]:
    return {
        "case_id": f"case-{group_id}",
        "source_group_id": group_id,
        "split": "train",
        "human_assessment": {
            "schema_version": "parking_risk_v1",
            "risk_level": risk,
            "events": events,
            "evidence": ["可见证据"],
            "driver_advice": ["maintain_observation"],
        },
    }


class CalibrationSelectionTests(unittest.TestCase):
    def test_prefers_new_event_coverage_deterministically(self) -> None:
        result = select_calibration_groups(
            [
                _record("g3", "low", []),
                _record("g1", "medium", ["narrow_passage"]),
                _record("g2", "high", ["vehicle_near_maneuver_path"]),
            ],
            sample_count=2,
            calibration_id="cal-v1",
            source_dataset="dataset-v1",
        )
        self.assertEqual(result["source_group_ids"], ["g2", "g1"])
        self.assertEqual(
            result["selection_audit"]["covered_events"],
            ["narrow_passage", "vehicle_near_maneuver_path"],
        )

    def test_rejects_unfinalized_or_duplicate_inputs_at_call_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_calibration_groups(
                [_record("g1", "low", []), _record("g1", "low", [])],
                sample_count=1,
                calibration_id="cal-v1",
                source_dataset="dataset-v1",
            )


if __name__ == "__main__":
    unittest.main()
