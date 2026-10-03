"""单图泊车风险分析命令。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from parksight_vlm.casebook import DatasetSplit, ParkingCase
from parksight_vlm.inference import InferenceRecord, RiskRuntime
from parksight_vlm.workload import FrozenWorkload

from .config import RuntimeConfig
from .runtime_factory import build_runtime


def analyze_image(
    *, image_path: Path, runtime: RiskRuntime, workload: FrozenWorkload # *表明后续参数必须写出参数名
) -> InferenceRecord:
    """通过已经组合完成的 Runtime 分析一张无标注图片。"""
    case = ParkingCase(
        case_id=image_path.stem,
        image_ref=PurePosixPath(image_path.name),   # 文件名称
        source_group_id=f"single-image:{image_path.stem}", # 使用stem得到无后缀的文件名称
        split=DatasetSplit.TEST,
        reference_assessment=None,
    )
    return runtime.analyze(case, workload)


def main(argv: list[str] | None = None) -> int: # 当 argv=None 时，argparse 自动读取真实命令行参数
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("configs/workloads/parking_risk_v1.json"),
    )
    parser.add_argument(
        "--runtime",
        choices=("transformers", "tensorrt_edge_llm_http", "vllm_http"),
        required=True,
    )
    parser.add_argument("--backend-revision", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision", default="none")
    parser.add_argument("--precision", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--adapter-path")
    parser.add_argument("--edge-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-model-name")
    parser.add_argument("--api-key-env")
    args = parser.parse_args(argv)

    image_path = args.image.resolve()
    workload = FrozenWorkload.load(args.workload)
    options: dict[str, object] = {}
    if args.runtime == "transformers":
        options.update(
            {
                "device_map": args.device_map,
                "dtype": args.dtype,
                "attn_implementation": args.attn_implementation,
            }
        )
        if args.adapter_path is not None:
            options["adapter_path"] = args.adapter_path
    elif args.runtime == "tensorrt_edge_llm_http":
        options["base_url"] = args.edge_url
    else:
        options.update(
            {
                "base_url": args.vllm_url,
                "model_name": args.vllm_model_name or args.model_id,
                "json_mode": True,
                "stream_responses": True,
                "reuse_http_connection": True,
            }
        )
        if args.api_key_env is not None:
            options["api_key_env"] = args.api_key_env
    runtime = build_runtime(
        RuntimeConfig(
            backend=args.runtime,
            backend_revision=args.backend_revision,
            model_id=args.model_id,
            model_revision=args.model_revision,
            adapter_revision=args.adapter_revision,
            precision=args.precision,
            options=options,
        ),
        data_root=image_path.parent,
    )
    record = analyze_image(image_path=image_path, runtime=runtime, workload=workload)
    print(json.dumps(record.to_mapping(), ensure_ascii=False, indent=2))
    return 0 if record.succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
