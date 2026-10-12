"""Correlate Nsight CUDA Graph launches with native decode counters, read-only."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


def analyze_replay(database: Path, probe: dict[str, Any]) -> dict[str, Any]:
    """Analyze the CUPTI global-ID schema observed in Nsight Systems 2024.5."""
    stage_before = probe["stage_before"]
    initial_runs = probe.get("native_counts_before", {}).get("generation_runs")
    if stage_before == {} and type(initial_runs) is int and initial_runs == 0:
        # This runtime lazily creates timer entries on the first instrumented run.
        before = 0
    elif "llm_generation" in stage_before:
        before = stage_before["llm_generation"]["run_count"]
    else:
        raise ValueError("missing initial decode timer without a verified fresh runtime")
    after = probe["stage_after"]["llm_generation"]["run_count"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (before, after)) or after < before:
        raise ValueError("invalid native decode counters")
    expected = after - before
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        launches = connection.execute(
            "SELECT r.globalTid, r.correlationId, r.returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId = s.id WHERE s.value LIKE 'cudaGraphLaunch%'"
        ).fetchall()
        kernels = connection.execute(
            "SELECT k.globalPid, k.correlationId, k.graphNodeId, k.start, k.end, s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id"
        ).fetchall()
    # In this schema, the low 24 bits of globalTid encode the thread namespace.
    keys = [(tid & ~((1 << 24) - 1), correlation) for tid, correlation, code in launches if code == 0]
    if len(set(keys)) != len(keys):
        raise ValueError("ambiguous repeated Graph correlation ID in one process")
    successful = set(keys)
    linked = [row for row in kernels if (row[0], row[1]) in successful and row[2] not in (None, 0)]
    per_launch = Counter((row[0], row[1]) for row in linked)
    missing = sorted(successful - per_launch.keys())
    by_name: dict[str, dict[str, Any]] = {}
    for pid, correlation, node, start, end, name in linked:
        if end < start:
            raise ValueError("negative GPU kernel duration")
        item = by_name.setdefault(name, {"name": name, "count": 0, "total_gpu_event_ms": 0.0})
        item["count"] += 1
        item["total_gpu_event_ms"] += (end - start) / 1_000_000
    ordered = sorted(by_name.values(), key=lambda item: (-item["total_gpu_event_ms"], item["name"]))
    failed = sum(code != 0 for _, _, code in launches)
    matches = (expected > 0 and len(keys) == expected and failed == 0 and not missing
               and len({pid for pid, _ in successful}) == 1)
    return {
        "schema_version": "parksight_edgellm_nsight_replay_v1",
        "native_decode_steps": expected,
        "successful_graph_launch_count": len(keys),
        "failed_graph_launch_count": failed,
        "graph_process_count": len({pid for pid, _ in successful}),
        "graph_correlated_kernel_count": len(linked),
        "kernel_count_per_launch_histogram": dict(sorted(Counter(per_launch.values()).items())),
        "launches_without_kernel_events": [list(key) for key in missing],
        "replay_matches_native_steps": matches,
        "top_graph_kernels": ordered[:20],
        "evidence_boundary": (
            "Correlations use process namespace plus correlationId and nonzero graphNodeId. "
            "Native counter agreement does not prove model-layer attribution or engine weight provenance. "
            "GPU event durations may overlap; do not add CPU synchronization waits. "
            "The global-ID encoding is scoped to the observed Nsight Systems 2024.5 CUPTI schema."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh evidence path")
    result = analyze_replay(args.sqlite, json.loads(args.probe.read_text(encoding="utf-8")))
    result["inputs"] = {
        "sqlite": {"path": str(args.sqlite), "sha256": hashlib.sha256(args.sqlite.read_bytes()).hexdigest()},
        "probe": {"path": str(args.probe), "sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest()},
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
