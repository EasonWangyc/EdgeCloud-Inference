"""无硬件、无 Pillow 安装的输入预处理契约测试。"""

from __future__ import annotations

import base64
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from parksight_vlm.inference import (
    EdgeLlmHttpBackend, RuntimeDependencyError, RuntimeGeneration, RuntimeInputError,
    StageTimings, VllmHttpBackend,
)
from parksight_vlm.app import AppConfigError, AppStudyConfig, build_runtime
from parksight_vlm.inference.images import load_workload_image
from parksight_vlm.inference.runtime import _classify_failure
from parksight_vlm.workload import FrozenWorkload


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = FrozenWorkload.load(ROOT / "configs/workloads/parking_risk_v1.json")


class ImagePreprocessingTests(unittest.TestCase):
    def test_edge_resize_lives_through_request_and_is_cleaned_with_timings(self) -> None:
        image = Mock()
        image.save.side_effect = lambda path, **kwargs: path.write_bytes(b"png")
        paths = []
        def generate(*, image_path, workload):
            self.assertEqual(image_path.read_bytes(), b"png")
            paths.append(image_path)
            return RuntimeGeneration(raw_output="{}", stage_timings=StageTimings(
                preprocess_ms=3, time_to_first_token_ms=7, client_total_ttft_ms=10,
                http_round_trip_ms=9, decode_ms=20,
            ))
        backend = EdgeLlmHttpBackend(image_preprocessing="workload_resize")
        with patch("parksight_vlm.inference.edge_llm.load_workload_image", return_value=image), patch(
            "parksight_vlm.inference.edge_llm.OpenAICompatibleHttpBackend.generate", side_effect=generate
        ), patch("parksight_vlm.inference.edge_llm.time.perf_counter", side_effect=[0, .05, .1, .15]):
            for _ in range(2):
                result = backend.generate(image_path=Path("scene.jpg"), workload=WORKLOAD)
                self.assertAlmostEqual(result.stage_timings.preprocess_ms, 53)
                self.assertAlmostEqual(result.stage_timings.client_total_ttft_ms, 60)
                self.assertEqual(result.stage_timings.time_to_first_token_ms, 7)
                self.assertEqual(result.stage_timings.http_round_trip_ms, 9)
                self.assertEqual(result.stage_timings.decode_ms, 20)
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(not p.parent.exists() for p in paths))
        self.assertEqual(image.close.call_count, 2)
        self.assertTrue(all(c.kwargs == {"format": "PNG"} for c in image.save.call_args_list))

    def test_edge_resize_cleans_images_and_connection_on_save_or_http_failure(self) -> None:
        for failure_stage in ("save", "http"):
            with self.subTest(stage=failure_stage):
                paths = []
                image = Mock()
                def save(path, **kwargs):
                    paths.append(path)
                    path.write_bytes(b"png")
                    if failure_stage == "save":
                        raise OSError("cannot save image")
                image.save.side_effect = save
                backend = EdgeLlmHttpBackend(image_preprocessing="workload_resize")
                with patch("parksight_vlm.inference.edge_llm.load_workload_image", return_value=image), patch(
                    "parksight_vlm.inference.edge_llm.OpenAICompatibleHttpBackend.generate",
                    side_effect=TimeoutError("HTTP timed out"),
                ), patch.object(backend, "close") as close:
                    with self.assertRaises((OSError, TimeoutError)):
                        backend.generate(image_path=Path("scene.jpg"), workload=WORKLOAD)
                    close.assert_called_once()
                image.close.assert_called_once()
                self.assertFalse(paths[0].parent.exists())

    def test_edge_application_passes_policy_and_preserves_source_default(self) -> None:
        config = AppStudyConfig.load(ROOT / "configs/studies/jetson_edgellm_fp16_workload_resize_ps20_pilot.json")
        runtime = build_runtime(config.runtime, data_root=config.data_root)
        self.assertEqual(runtime._backend._image_preprocessing, "workload_resize")
        invalid = replace(config.runtime, options={**config.runtime.options, "image_preprocessing": "invalid"})
        with self.assertRaises(AppConfigError):
            build_runtime(invalid, data_root=config.data_root)
        self.assertEqual(EdgeLlmHttpBackend()._image_preprocessing, "source")

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
