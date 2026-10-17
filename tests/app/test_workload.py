"""冻结 workload 配置与溯源测试。"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from parksight_vlm.assessment import DriverAdvice, ParkingRiskEvent
from parksight_vlm.workload import FrozenWorkload, WorkloadValidationError


WORKLOAD_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "workloads"
    / "parking_risk_v1.json"
)


class FrozenWorkloadTests(unittest.TestCase):
    def test_rendered_user_prompt_contains_strict_json_contract(self) -> None:
        workload = FrozenWorkload.load(WORKLOAD_PATH)

        rendered_prompt = workload.render_user_prompt()

        self.assertIn(workload.user_prompt, rendered_prompt)
        self.assertIn("events 必须是 JSON 数组", rendered_prompt)
        self.assertIn("evidence 必须是 JSON 字符串数组", rendered_prompt)
        self.assertIn("driver_advice 必须是非空 JSON 数组", rendered_prompt)
        self.assertIn("禁止翻译、改写或扩写为自然语言", rendered_prompt)
        for event in ParkingRiskEvent:
            self.assertIn(f'"{event.value}"', rendered_prompt)
        for advice in DriverAdvice:
            self.assertIn(f'"{advice.value}"', rendered_prompt)

    def test_strict_json_v2_workload_adds_semantic_risk_contract(self) -> None:
        workload = FrozenWorkload.load(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "workloads"
            / "parking_risk_v2_strict_json.json"
        )
        rendered_prompt = workload.render_user_prompt()
        self.assertEqual(workload.schema_version, "parking_risk_v1")
        self.assertNotEqual(
            workload.identity,
            FrozenWorkload.load(WORKLOAD_PATH).identity,
        )
        self.assertIn("按定义输出严格 JSON", rendered_prompt)
        self.assertIn("prepare_to_stop", workload.system_prompt)

    def test_loads_frozen_workload_and_builds_stable_identity(self) -> None:
        workload = FrozenWorkload.load(WORKLOAD_PATH)

        self.assertEqual(workload.workload_id, "parking_risk_v1")
        self.assertEqual((workload.input_size.width, workload.input_size.height), (448, 448))
        self.assertEqual(set(workload.risk_events), set(ParkingRiskEvent))
        self.assertEqual(set(workload.driver_advice), set(DriverAdvice))
        self.assertEqual(len(workload.fingerprint), 64)
        self.assertEqual(
            FrozenWorkload.from_mapping(workload.to_mapping()).identity,
            workload.identity,
        )

    def test_rejects_schema_drift_and_incomplete_event_set(self) -> None:
        payload = FrozenWorkload.load(WORKLOAD_PATH).to_mapping()
        payload["schema_version"] = "parking_risk_v2"
        with self.assertRaisesRegex(WorkloadValidationError, "unsupported schema_version"):
            FrozenWorkload.from_mapping(payload)

        payload = FrozenWorkload.load(WORKLOAD_PATH).to_mapping()
        payload["risk_events"] = payload["risk_events"][:-1]
        with self.assertRaisesRegex(WorkloadValidationError, "missing values"):
            FrozenWorkload.from_mapping(payload)

    def test_rejects_invalid_input_and_generation_parameters(self) -> None:
        valid_payload = FrozenWorkload.load(WORKLOAD_PATH).to_mapping()
        cases = (
            ("input_size", {"width": 0, "height": 448}),
            ("generation", {"max_new_tokens": True, "do_sample": False}),
            ("generation", {"max_new_tokens": 256, "do_sample": "false"}),
        )
        for field_name, invalid_value in cases:
            with self.subTest(field_name=field_name, invalid_value=invalid_value):
                payload = copy.deepcopy(valid_payload)
                payload[field_name] = invalid_value
                with self.assertRaises(WorkloadValidationError):
                    FrozenWorkload.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
