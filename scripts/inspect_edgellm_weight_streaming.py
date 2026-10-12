"""Query an existing TensorRT engine's weight budgets without creating a context."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import platform
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_budget_rows(engine, budgets: list[int], *, on_row=None) -> list[dict]:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in budgets):
        raise ValueError("budgets must be non-negative integer bytes")
    if len(set(budgets)) != len(budgets) or not budgets:
        raise ValueError("provide distinct budgets")
    if engine.streamable_weights_size <= 0:
        raise ValueError("engine has no streamable weights")
    rows = []
    for budget in budgets:
        engine.weight_streaming_budget_v2 = budget
        row = {"requested_budget_bytes": budget, "actual_budget_bytes": engine.weight_streaming_budget_v2,
               "scratch_bytes": engine.weight_streaming_scratch_memory_size,
               "context_bytes": engine.device_memory_size_v2,
               "profile_context_bytes": [engine.get_device_memory_size_for_profile_v2(i)
                                         for i in range(engine.num_optimization_profiles)]}
        for name in ("actual_budget_bytes", "scratch_bytes", "context_bytes"):
            if not isinstance(row[name], int) or isinstance(row[name], bool) or row[name] < 0:
                raise ValueError(f"invalid TensorRT {name}")
        if row["context_bytes"] < row["scratch_bytes"]:
            raise ValueError("context query omitted weight-streaming scratch")
        row["budget_applied"] = (row["actual_budget_bytes"] == budget or
                                 (budget >= engine.streamable_weights_size and
                                  row["actual_budget_bytes"] >= engine.streamable_weights_size))
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--plugin-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--budgets-bytes", type=int, nargs="+", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output_directory.exists():
        parser.error("output directory already exists")
    if not args.budgets_bytes or any(v < 0 for v in args.budgets_bytes) or len(set(args.budgets_bytes)) != len(args.budgets_bytes):
        parser.error("budgets must be distinct non-negative integer bytes")
    identities = [{"path": str(path.resolve()), "sha256": expected} for path, expected in
                  [(args.engine, args.engine_sha256), (args.plugin, args.plugin_sha256)]]
    for entry in identities:
        if sha256(Path(entry["path"])) != entry["sha256"]:
            parser.error("engine or plugin identity changed")
    result = {"status": "ready_for_explicit_execution", "inputs": identities,
              "requested_budgets_bytes": args.budgets_bytes,
              "scope": "Engine-only requirements query; budget setters may allocate device weights; "
                       "no user context creation, graph capture, generation or fit proof",
              "profile_query_scope": "Raw profile API values; global context_bytes is used for the full scratch-inclusive requirement"}
    if not args.execute:
        print(json.dumps(result, indent=2))
        return 0
    output = args.output_directory.resolve()
    output.mkdir(parents=True)
    result.update(status="running", script_sha256=sha256(Path(__file__)),
                  python=sys.version, platform=platform.platform(), rows=[])
    started = time.perf_counter()
    try:
        import tensorrt as trt
        result.update(tensorrt_version=trt.__version__, tensorrt_module_path=trt.__file__)
        plugin = ctypes.CDLL(str(args.plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, "")
        with trt.Runtime(logger) as runtime, args.engine.open("rb") as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                engine = runtime.deserialize_cuda_engine(mapped)
                if engine is None:
                    raise RuntimeError("TensorRT deserialization returned no engine")
                result["streamable_weights_bytes"] = engine.streamable_weights_size
                try:
                    inspect_budget_rows(engine, args.budgets_bytes, on_row=result["rows"].append)
                finally:
                    del engine
        del plugin
        result.update(status="queried" if all(row["budget_applied"] for row in result["rows"])
                      else "queried_with_rejected_budgets", original_inputs_unchanged=all(
            sha256(Path(entry["path"])) == entry["sha256"] for entry in identities))
        if not result["original_inputs_unchanged"]:
            raise RuntimeError("engine or plugin changed during query")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        (output / "budget_requirements.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "queried" else 2


if __name__ == "__main__":
    raise SystemExit(main())
