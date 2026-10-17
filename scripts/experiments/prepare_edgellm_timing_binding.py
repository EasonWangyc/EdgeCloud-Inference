"""Prepare a source-bound native timing binding patch without editing the checkout."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


def timing_binding_patch(
    source: bytes, expected_sha256: str, *, fix_metrics_runs: bool = False,
) -> tuple[str, str]:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("binding source SHA-256 does not match the inspected snapshot")
    text = source.decode("utf-8")
    if "get_stage_timing_snapshot" in text:
        raise ValueError("stage timing binding already exists")
    newline = "\r\n" if "\r\n" in text else "\n"
    include = '#include "profiling/metrics.h"'
    anchor = '    m.def("get_profiling_enabled", &getProfilingEnabled, "Check if profiling is currently enabled");'
    if text.count(include) != 1 or text.count(anchor) != 1:
        raise ValueError("unsupported Edge-LLM profiling binding layout")
    block = '''

    m.def("get_stage_timing_snapshot", []() {
        py::dict result;
        for (auto const& entry : trt_edgellm::gTimer.getAllTimingData())
        {
            py::dict timing;
            timing["total_gpu_time_ms"] = entry.second.getTotalGpuTimeMs();
            timing["run_count"] = entry.second.getTotalRuns();
            result[py::str(entry.first)] = timing;
        }
        return result;
    }, "Read cumulative CUDA event timings only while native runtime workers are quiescent");'''.replace("\n", newline)
    updated = text.replace(include, include + newline + '#include "profiling/timer.h"', 1)
    updated = updated.replace(anchor, anchor + block, 1)
    if fix_metrics_runs:
        # Inherited member pointers infer BaseMetrics as self, but the upstream
        # binding does not register that base. Typed lambdas bind the actual class.
        for name in ("LLMPrefillMetrics", "LLMGenerationMetrics",
                     "SpecDecodeGenerationMetrics", "MultimodalMetrics"):
            original = f'.def("get_total_runs", &metrics::{name}::getTotalRuns)'
            if updated.count(original) != 1:
                raise ValueError(f"unsupported metrics run-count binding: {name}")
            replacement = (f'.def("get_total_runs", [](metrics::{name} const& self) '
                           '{ return self.getTotalRuns(); })')
            updated = updated.replace(original, replacement, 1)
    patch = "".join(difflib.unified_diff(
        text.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="a/experimental/pybind/edgellm_pybind.cpp",
        tofile="b/experimental/pybind/edgellm_pybind.cpp",
    ))
    return patch, hashlib.sha256(updated.encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fix-metrics-runs", action="store_true",
                        help="bind inherited getTotalRuns through typed derived-class lambdas")
    args = parser.parse_args(argv)
    record_path = args.output.with_suffix(args.output.suffix + ".json")
    for path in (args.output, record_path):
        if path.exists():
            parser.error(f"evidence already exists: {path}")
    source = args.source.read_bytes()
    patch, resulting_sha256 = timing_binding_patch(
        source, args.source_sha256, fix_metrics_runs=args.fix_metrics_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as file:
        file.write(patch)
    record = {
        "status": "prepared",
        "source_path": str(args.source.resolve()),
        "source_sha256": args.source_sha256,
        "resulting_source_sha256": resulting_sha256,
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "fix_metrics_runs": args.fix_metrics_runs,
        "line_ending": "CRLF" if b"\r\n" in source else "LF",
        "evidence_boundary": "Patch prepared only; no checkout edits, compilation or GPU validation",
        "measurement_boundary": "Cumulative GPU event time and run count; read only with all workers quiescent",
    }
    with record_path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
