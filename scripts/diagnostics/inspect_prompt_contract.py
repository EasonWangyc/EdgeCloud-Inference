"""生成 Transformers 与 TensorRT Edge-LLM 的 Prompt 契约诊断报告。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from parksight_vlm.inference.prompt_contract import (
    ChatTemplateProcessor,
    build_prompt_contract_report,
)
from parksight_vlm.workload import FrozenWorkload


ProcessorLoader = Callable[[Path], ChatTemplateProcessor]


def load_local_processor(model_source: Path) -> ChatTemplateProcessor:
    """从本地固定模型目录加载 processor，不访问 Hugging Face 网络。"""
    try:
        from transformers import AutoProcessor
    except ImportError as error:
        raise RuntimeError("该诊断需要已安装 transformers 的 Python 环境") from error
    return AutoProcessor.from_pretrained(
        str(model_source),
        local_files_only=True,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    processor_loader: ProcessorLoader = load_local_processor,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model-source", required=True, type=Path)
    parser.add_argument("--processed-chat-template", required=True, type=Path)
    parser.add_argument("--model-name", default="local")
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        help="可选的输入 token 上限；超出时返回退出码 2",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    workload = FrozenWorkload.load(args.workload)
    processor = processor_loader(args.model_source.resolve())
    report = build_prompt_contract_report(
        workload=workload,
        image_path=args.image,
        model_source=args.model_source,
        processed_chat_template_path=args.processed_chat_template,
        processor=processor,
        model_name=args.model_name,
        max_input_tokens=args.max_input_tokens,
    )
    serialized_report = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized_report + "\n", encoding="utf-8")
    print(serialized_report)
    within_budget = report["prompt_tokens"]["within_budget"]
    return 2 if within_budget is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
