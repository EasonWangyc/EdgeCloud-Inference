"""A CLI-created runtime is closed even when analysis or serialization fails."""

from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import Mock, patch

from parksight_vlm.app.analyze_image import main
from parksight_vlm.app.config import AppStudyConfig
from parksight_vlm.app.run_study import run_configured_study


ROOT = Path(__file__).resolve().parents[2]
ARGS = [
    "--image", str(ROOT / "tests/fixtures/inference/scene.jpg"),
    "--workload", str(ROOT / "configs/workloads/parking_risk_v1.json"),
    "--runtime", "tensorrt_edge_llm_http", "--backend-revision", "test",
    "--model-revision", "revision", "--precision", "fp16",
]


class SingleImageCleanupTests(unittest.TestCase):
    def test_successful_and_failed_records_both_release_the_runtime(self):
        for succeeded in (True, False):
            with self.subTest(succeeded=succeeded):
                runtime, record = Mock(), Mock(succeeded=succeeded)
                record.to_mapping.return_value = {"succeeded": succeeded}
                with patch("parksight_vlm.app.analyze_image.build_runtime", return_value=runtime), \
                        patch("parksight_vlm.app.analyze_image.analyze_image", return_value=record), \
                        patch("builtins.print"):
                    self.assertEqual(main(ARGS), 0 if succeeded else 2)
                runtime.close.assert_called_once_with()

    def test_analysis_exception_still_releases_the_runtime(self):
        runtime = Mock()
        with patch("parksight_vlm.app.analyze_image.build_runtime", return_value=runtime), \
                patch("parksight_vlm.app.analyze_image.analyze_image", side_effect=RuntimeError("analysis failed")), \
                self.assertRaisesRegex(RuntimeError, "analysis failed"):
            main(ARGS)
        runtime.close.assert_called_once_with()

    def test_serialization_exception_occurs_after_cleanup(self):
        runtime, record = Mock(), Mock(succeeded=True)

        def serialize():
            runtime.close.assert_called_once_with()
            raise ValueError("serialization failed")

        record.to_mapping.side_effect = serialize
        with patch("parksight_vlm.app.analyze_image.build_runtime", return_value=runtime), \
                patch("parksight_vlm.app.analyze_image.analyze_image", return_value=record), \
                self.assertRaisesRegex(ValueError, "serialization failed"):
            main(ARGS)


class StudyCleanupTests(unittest.TestCase):
    def config(self, output):
        return replace(AppStudyConfig.load(ROOT / "configs/studies/wsl_vllm_workload_resize_ps20.json"),
                       output_path=output)

    def test_cleanup_failure_keeps_completed_report_on_disk(self):
        runtime, report = Mock(), Mock()
        runtime.close.side_effect = RuntimeError("cleanup failed")
        report.write_json.side_effect = lambda path: path.write_text('{"records": []}')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch("parksight_vlm.app.run_study.build_runtime", return_value=runtime), \
                    patch("parksight_vlm.app.run_study.ParkingCaseCatalog.load"), \
                    patch("parksight_vlm.app.run_study.StudyRunner") as runner, \
                    self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                runner.return_value.run.return_value = report
                run_configured_study(self.config(output))
            self.assertEqual(output.read_text(), '{"records": []}')
            runtime.close.assert_called_once_with()

    def test_report_write_failure_still_closes_runtime(self):
        runtime, report = Mock(), Mock()
        report.write_json.side_effect = OSError("report write failed")
        with tempfile.TemporaryDirectory() as directory:
            with patch("parksight_vlm.app.run_study.build_runtime", return_value=runtime), \
                    patch("parksight_vlm.app.run_study.ParkingCaseCatalog.load"), \
                    patch("parksight_vlm.app.run_study.StudyRunner") as runner, \
                    self.assertRaisesRegex(OSError, "report write failed"):
                runner.return_value.run.return_value = report
                run_configured_study(self.config(Path(directory) / "report.json"))
        runtime.close.assert_called_once_with()
