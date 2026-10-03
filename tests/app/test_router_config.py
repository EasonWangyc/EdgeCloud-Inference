"""端云组合 Runtime 配置工厂的无硬件测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from parksight_vlm.app import AppStudyConfig, RuntimeConfig, build_runtime
from parksight_vlm.inference import EdgeCloudRouterRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTER_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "studies"
    / "jetson_edge_vllm_router_ps20_pilot.example.json"
)


class RouterConfigTests(unittest.TestCase):
    def test_router_study_template_builds_two_lazy_http_runtimes(self) -> None:
        study = AppStudyConfig.load(ROUTER_CONFIG_PATH)
        payload = json.loads(ROUTER_CONFIG_PATH.read_text(encoding="utf-8"))
        runtime_payload = payload["runtime"]
        runtime_payload["options"]["cloud_runtime"]["backend_revision"] = "vllm-test"
        runtime_config = RuntimeConfig.from_mapping(runtime_payload)

        runtime = build_runtime(runtime_config, data_root=study.data_root)

        self.assertIsInstance(runtime, EdgeCloudRouterRuntime)
        self.assertEqual(runtime.identity.backend, "edge_vllm_router")
        self.assertEqual(runtime.identity.precision, "routed")
        self.assertFalse(
            runtime_payload["options"]["signals"]["cloud_allowed"]
        )

    def test_router_factory_rejects_wrong_nested_backend(self) -> None:
        payload = json.loads(ROUTER_CONFIG_PATH.read_text(encoding="utf-8"))["runtime"]
        payload["options"]["edge_runtime"]["backend"] = "transformers"
        payload["options"]["cloud_runtime"]["backend_revision"] = "vllm-test"
        runtime_config = RuntimeConfig.from_mapping(payload)

        with self.assertRaisesRegex(ValueError, "edge_runtime.backend"):
            build_runtime(runtime_config, data_root=PROJECT_ROOT / "data")


if __name__ == "__main__":
    unittest.main()
