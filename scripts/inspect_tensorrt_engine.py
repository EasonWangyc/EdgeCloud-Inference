"""在 Jetson 上导出 TensorRT Engine Inspector 的层级与 tactic 信息。"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Any

from parksight_vlm.tensorrt import sha256_file


def inspect_engine(
    engine_path: Path | str, *, plugin_path: Path | str | None = None
) -> dict[str, Any]:
    """加载 engine 并导出 inspector 信息；需要目标机安装 TensorRT Python。"""
    path = Path(engine_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT engine not found: {path}")

    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python module is required on the target Jetson") from error

    logger = trt.Logger(trt.Logger.WARNING)
    if plugin_path is not None:
        plugin = Path(plugin_path).resolve()
        if not plugin.is_file():
            raise FileNotFoundError(f"TensorRT plugin not found: {plugin}")
        ctypes.CDLL(str(plugin), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    trt.init_libnvinfer_plugins(logger, "")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize engine: {path}")
    inspector = engine.create_engine_inspector()
    if inspector is None:
        raise RuntimeError("TensorRT engine inspector is unavailable")

    output_format = trt.LayerInformationFormat.JSON
    engine_information = _decode_json_or_text(
        inspector.get_engine_information(output_format)
    )
    layers: list[Any] = []
    for layer_index in range(int(engine.num_layers)):
        layers.append(
            _decode_json_or_text(
                inspector.get_layer_information(layer_index, output_format)
            )
        )

    return {
        "schema_version": "parksight_tensorrt_engine_inspector_v1",
        "engine": {
            "path": str(path),
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "tensor_rt_version": getattr(trt, "__version__", None),
            "num_layers": int(engine.num_layers),
            "num_io_tensors": int(getattr(engine, "num_io_tensors", 0)),
            "device_memory_size": _optional_int(
                getattr(engine, "device_memory_size", None)
            ),
            "device_memory_size_v2": _optional_int(
                getattr(engine, "device_memory_size_v2", None)
            ),
        },
        "engine_information": engine_information,
        "layers": layers,
        "evidence_boundary": (
            "Engine Inspector reports serialized layer metadata and tactic fields when "
            "the engine was built with sufficient profiling verbosity; it is not a "
            "substitute for runtime timing or Nsight kernel traces"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--plugin-path", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = inspect_engine(args.engine, plugin_path=args.plugin_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


def _decode_json_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _optional_int(value: Any) -> int | None:
    if callable(value):
        value = value()
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


if __name__ == "__main__":
    raise SystemExit(main())
