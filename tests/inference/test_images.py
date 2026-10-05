"""无硬件、无 Pillow 安装的输入预处理契约测试。"""

from __future__ import annotations

import base64
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from parksight_vlm.inference import RuntimeDependencyError, RuntimeInputError, VllmHttpBackend
from parksight_vlm.app import AppConfigError, AppStudyConfig, build_runtime
from parksight_vlm.inference.images import load_workload_image
from parksight_vlm.inference.runtime import _classify_failure
from parksight_vlm.workload import FrozenWorkload


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = FrozenWorkload.load(ROOT / "configs/workloads/parking_risk_v1.json")


class ImagePreprocessingTests(unittest.TestCase):
    def test_rgb_resize_uses_frozen_dimensions_and_bicubic(self) -> None:
        source = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=source)
        context.__exit__ = Mock(return_value=False)
        image_module = types.SimpleNamespace(
            open=Mock(return_value=context),
            Resampling=types.SimpleNamespace(BICUBIC=3),
            DecompressionBombError=type("DecompressionBombError", (Exception,), {}),
        )
        with patch.dict(sys.modules, {"PIL": types.SimpleNamespace(Image=image_module)}):
            result = load_workload_image(Path("scene.jpg"), WORKLOAD)
        source.convert.assert_called_once_with("RGB")
        source.convert.return_value.resize.assert_called_once_with(
            (WORKLOAD.input_size.width, WORKLOAD.input_size.height), resample=3
        )
        self.assertIs(result, source.convert.return_value.resize.return_value)
        context.__exit__.assert_called_once()

    def test_missing_pillow_is_dependency_failure(self) -> None:
        with patch.dict(sys.modules, {"PIL": None}):
            with self.assertRaises(RuntimeDependencyError):
                load_workload_image(Path("scene.jpg"), WORKLOAD)

    def test_bad_image_is_input_failure(self) -> None:
        image_module = types.SimpleNamespace(
            open=Mock(side_effect=OSError("invalid image bytes")),
            DecompressionBombError=type("DecompressionBombError", (Exception,), {}),
        )
        with patch.dict(sys.modules, {"PIL": types.SimpleNamespace(Image=image_module)}):
            with self.assertRaises(RuntimeInputError) as caught:
                load_workload_image(Path("scene.jpg"), WORKLOAD)
        self.assertEqual(_classify_failure(caught.exception).category.value, "input_error")

    def test_workload_resize_sends_lossless_png_and_closes_image(self) -> None:
        image = Mock()
        image.save.side_effect = lambda buffer, **kwargs: buffer.write(b"encoded-png")
        backend = VllmHttpBackend(
            base_url="http://127.0.0.1:8000", model_name="test",
            image_preprocessing="workload_resize",
        )
        with patch("parksight_vlm.inference.vllm.load_workload_image", return_value=image) as load:
            payload = backend.build_request_payload(image_path=Path("scene.jpg"), workload=WORKLOAD)
        load.assert_called_once_with(Path("scene.jpg"), WORKLOAD)
        uri = payload["messages"][1]["content"][0]["image_url"]["url"]
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), b"encoded-png")
        self.assertEqual(image.save.call_args.kwargs, {"format": "PNG"})
        image.close.assert_called_once()

    def test_invalid_policy_is_rejected_before_http(self) -> None:
        with self.assertRaisesRegex(ValueError, "image_preprocessing"):
            VllmHttpBackend(base_url="http://127.0.0.1:8000", model_name="test", image_preprocessing="invalid")

    def test_application_passes_policy_and_rejects_invalid_policy(self) -> None:
        config = AppStudyConfig.load(ROOT / "configs/studies/wsl_vllm_workload_resize_ps20.json")
        runtime = build_runtime(config.runtime, data_root=config.data_root)
        self.assertEqual(runtime._backend._image_preprocessing, "workload_resize")
        invalid = replace(config.runtime, options={**config.runtime.options, "image_preprocessing": "invalid"})
        with self.assertRaises(AppConfigError):
            build_runtime(invalid, data_root=config.data_root)
