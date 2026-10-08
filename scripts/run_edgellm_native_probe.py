"""Explicit single-image native diagnostic with isolated engine metadata and CUDA timers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def construct_with_binding(engine_module, runtime_module, **options):
    """Reuse the validated module; upstream's path loader otherwise imports it twice."""
    original = engine_module._import_runtime
    engine_module._import_runtime = lambda: runtime_module
    try:
        return engine_module.LLM(**options)
    finally:
        engine_module._import_runtime = original


def isolated_engine_view(source: Path, destination: Path) -> list[dict]:
    """Copy mutable metadata; reference only engine/weight binaries read-only."""
    evidence = []
    def files(folder: Path, ancestors: frozenset[Path]):
        resolved = folder.resolve()
        if resolved in ancestors:
            raise ValueError(f"cyclic engine directory symlink: {folder}")
        for path in sorted(folder.iterdir()):
            if path.is_dir():
                yield from files(path, ancestors | {resolved})
            elif path.is_file():
                yield path
    for path in files(source, frozenset()):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        identity = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix in {".engine", ".safetensors"}:
            target.symlink_to(path.resolve())
            identity["view_mode"] = "binary_symlink"
        else:
            shutil.copy2(path, target)
            identity["view_mode"] = "metadata_copy"
        evidence.append(identity)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-directory", type=Path, required=True)
    parser.add_argument("--edge-llm-root", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    required = [args.engine_root / "llm/llm.engine", args.engine_root / "visual/visual.engine",
                args.image, args.workload]
    if any(not path.is_file() for path in required):
        parser.error("required engine, image or workload is missing")
    if not args.execute:
        print(json.dumps({"status": "ready_for_explicit_execution", "required_inputs": [str(p) for p in required]}))
        return 0
    output.mkdir(parents=True)
    result = {"status": "running", "scope": "Single native SSE diagnostic with profiling; no HTTP transport or baseline claim",
              "workload_sha256": digest(args.workload), "image_sha256": digest(args.image),
              "script_sha256": digest(Path(__file__)), "events": []}
    destination = output / "probe.json"
    def save():
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    save()
    rt = None
    try:
        from parksight_vlm.assessment import ParkingAssessment
        from parksight_vlm.inference import EdgeLlmHttpBackend
        from parksight_vlm.inference.edge_llm import OpenAICompatibleHttpBackend
        from parksight_vlm.inference.edge_server_metrics import install_stream_usage
        from parksight_vlm.inference.images import load_workload_image
        from parksight_vlm.workload import FrozenWorkload

        workload = FrozenWorkload.load(args.workload)
        if workload.generation.do_sample:
            raise ValueError("native probe requires a frozen greedy workload")
        sys.path.insert(0, str(args.edge_llm_root.resolve()))
        sys.path.insert(0, str(args.binding_directory.resolve()))
        import _edgellm_runtime as rt
        if Path(rt.__file__).resolve().parent != args.binding_directory.resolve():
            raise RuntimeError("unexpected native binding loaded")
        result.update(binding_path=rt.__file__, binding_sha256=digest(Path(rt.__file__)),
                      workload_identity=workload.identity)
        from experimental.server import SamplingParams, api_server, engine

        install_stream_usage(api_server)
        result["phase"] = "engine_view_preparation"
        result["engine_files"] = isolated_engine_view(args.engine_root.resolve(), output / "engines")
        rt.set_profiling_enabled(False)
        result["phase"] = "runtime_initialization"
        started = time.perf_counter()
        llm = construct_with_binding(
            engine, rt, engine_dir=str(output / "engines/llm"),
            visual_engine_dir=str(output / "engines/visual"), max_batch_size=1)
        result["runtime_initialization_ms"] = (time.perf_counter() - started) * 1000
        result["has_draft_model"] = llm.has_draft_model
        if llm.has_draft_model:
            raise RuntimeError("probe requires vanilla single-sequence decoding")
        started = time.perf_counter()
        prepared = load_workload_image(args.image, workload)
        try:
            prepared_path = output / "input.png"
            prepared.save(prepared_path, format="PNG")
        finally:
            prepared.close()
        result["image_prepare_ms"] = (time.perf_counter() - started) * 1000
        messages = EdgeLlmHttpBackend().build_request_payload(
            image_path=prepared_path, workload=workload)["messages"]
        rt.set_profiling_enabled(True)
        result["phase"] = "native_generation"
        result["stage_before"] = rt.get_stage_timing_snapshot()
        def native_counts():
            prefill = llm._runtime.get_prefill_metrics()
            generation = llm._runtime.get_generation_metrics()
            return {"prefill_runs": prefill.get_total_runs(), "computed_tokens": prefill.computed_tokens,
                    "reused_tokens": prefill.reused_tokens, "generation_runs": generation.get_total_runs(),
                    "generated_tokens": generation.generated_tokens}
        result["native_counts_before"] = native_counts()
        known_threads = {thread.ident for thread in threading.enumerate()}
        started = time.perf_counter()
        stream = api_server._generate_stream_sse(
            llm, messages, SamplingParams(temperature=0.0, max_tokens=workload.generation.max_new_tokens),
            "parksight-native-probe", False)
        def captured():
            for event in stream:
                result["events"].append(event)
                yield event.encode()
        payload, ttft, _, arrivals = OpenAICompatibleHttpBackend.read_stream_response(captured(), request_start=started)
        result["native_stream_total_ms"] = (time.perf_counter() - started) * 1000
        if any(thread.ident not in known_threads and thread.is_alive() for thread in threading.enumerate()):
            raise RuntimeError("new worker still alive; refusing global Timer read")
        result.update(stage_after=rt.get_stage_timing_snapshot(), native_stream_ttft_ms=ttft,
                      native_counts_after=native_counts(),
                      stream_timings=arrivals.to_mapping(), usage=payload.get("usage", {}),
                      raw_output=payload["choices"][0]["message"]["content"])
        try:
            assessment = ParkingAssessment.from_mapping(json.loads(result["raw_output"]))
            result.update(status="assessment_valid", assessment=assessment.to_mapping())
        except (ValueError, TypeError) as error:
            result.update(status="assessment_failed", assessment_error=str(error))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", failure_phase=result.get("phase"))
        raise
    finally:
        if rt is not None:
            rt.set_profiling_enabled(False)
        result["original_metadata_unchanged"] = all(
            digest(Path(entry["path"])) == entry["sha256"] for entry in result.get("engine_files", [])
            if entry["view_mode"] == "metadata_copy")
        save()
    print(destination)
    return 0 if result["status"] == "assessment_valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
