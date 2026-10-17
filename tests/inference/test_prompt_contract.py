"""Transformers 与 Edge-LLM Prompt 契约诊断测试。"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from parksight_vlm.inference.prompt_contract import build_prompt_contract_report
from parksight_vlm.workload import FrozenWorkload
from scripts.diagnostics.inspect_prompt_contract import main as inspect_prompt_contract_main


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_PATH = ROOT / "configs" / "workloads" / "parking_risk_v1.json"
IMAGE_PATH = ROOT / "tests" / "fixtures" / "inference" / "scene.jpg"
MODEL_SOURCE = ROOT / "tests" / "fixtures" / "prompt_contract" / "model"
PROCESSED_TEMPLATE_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "prompt_contract"
    / "processed_chat_template.json"
)


class FakeProcessor:
    """模拟外部 Transformers processor，只保留公开 chat 接口。"""

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> object:
        self.messages = messages
        if not add_generation_prompt:
            raise AssertionError("诊断必须保留 generation prompt")
        if tokenize:
            self.assert_tokenize_kwargs(kwargs)
            return {"input_ids": [[1, 2, 3]]}
        return "<system>strict-json</system><assistant>"

    @staticmethod
    def assert_tokenize_kwargs(kwargs: dict[str, object]) -> None:
        if kwargs != {"return_dict": True, "return_tensors": "pt"}:
            raise AssertionError(f"unexpected tokenization kwargs: {kwargs}")


class PromptContractReportTests(unittest.TestCase):
    def test_report_exposes_runtime_messages_and_template_identity(self) -> None:
        workload = FrozenWorkload.load(WORKLOAD_PATH)
        processor = FakeProcessor()
        report = build_prompt_contract_report(
            workload=workload,
            image_path=IMAGE_PATH,
            model_source=MODEL_SOURCE,
            processed_chat_template_path=PROCESSED_TEMPLATE_PATH,
            processor=processor,
        )

        self.assertEqual(
            report["schema_version"],
            "parksight_prompt_contract_report_v1",
        )
        self.assertEqual(report["workload_identity"], workload.identity)
        self.assertEqual(report["prompt_tokens"], {"count": 3, "budget": None, "within_budget": None})
        self.assertEqual(
            report["transformers_rendered_prompt"],
            "<system>strict-json</system><assistant>",
        )
        self.assertEqual(
            report["processed_chat_template"]["sha256"],
            "4dd998520d6e1991b70ed6ec1398d6c19d52da12d8fe18f96281a90ec64bfe49",
        )
        self.assertEqual(
            report["processed_chat_template"]["payload"],
            {"template": "edge"},
        )
        self.assertEqual(
            report["edge_http_request"]["messages"],
            processor.messages,
        )
        self.assertEqual(
            report["message_contract"],
            {
                "messages_equal": True,
                "system_content_type": "array",
                "user_content_types": ["image", "text"],
            },
        )

    def test_cli_prints_machine_readable_report_without_model_inference(self) -> None:
        processor = FakeProcessor()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = inspect_prompt_contract_main(
                [
                    "--workload",
                    str(WORKLOAD_PATH),
                    "--image",
                    str(IMAGE_PATH),
                    "--model-source",
                    str(MODEL_SOURCE),
                    "--processed-chat-template",
                    str(PROCESSED_TEMPLATE_PATH),
                ],
                processor_loader=lambda _: processor,
            )

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            report["schema_version"],
            "parksight_prompt_contract_report_v1",
        )
        self.assertEqual(report["message_contract"]["messages_equal"], True)

    def test_cli_rejects_prompt_over_budget(self) -> None:
        processor = FakeProcessor()
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = inspect_prompt_contract_main(
                [
                    "--workload",
                    str(WORKLOAD_PATH),
                    "--image",
                    str(IMAGE_PATH),
                    "--model-source",
                    str(MODEL_SOURCE),
                    "--processed-chat-template",
                    str(PROCESSED_TEMPLATE_PATH),
                    "--max-input-tokens",
                    "2",
                ],
                processor_loader=lambda _: processor,
            )

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(report["prompt_tokens"]["within_budget"], False)


if __name__ == "__main__":
    unittest.main()
