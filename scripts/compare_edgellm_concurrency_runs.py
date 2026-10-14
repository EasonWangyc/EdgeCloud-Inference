"""Check exact outputs and aggregate throughput across an Edge-LLM c=A/B/A run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any


_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_FIXED_FIELDS = (
    "cases", "repetitions", "warmup", "max_batch_size", "max_kv_pool_pages",
    "stream_responses", "reuse_http_connection",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(jsonl: Path, metadata_path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for line_number, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("status") != "completed":
            raise ValueError(f"{jsonl}:{line_number}: request did not complete")
        case_id = row.get("sample_id")
        repetition = row.get("repetition")
        output_tokens = row.get("output_tokens")
        output_hash = row.get("output_sha256")
        if not isinstance(case_id, str) or not case_id or isinstance(repetition, bool) or not isinstance(repetition, int) or repetition <= 0:
            raise ValueError(f"{jsonl}:{line_number}: invalid sample identity")
        if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0:
            raise ValueError(f"{jsonl}:{line_number}: invalid output token count")
        if not isinstance(output_hash, str) or _HASH_RE.fullmatch(output_hash) is None:
            raise ValueError(f"{jsonl}:{line_number}: output_sha256 is missing or invalid")
        key = (case_id, repetition)
        if key in rows:
            raise ValueError(f"{jsonl}:{line_number}: duplicate sample identity {key}")
        rows[key] = row
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not rows or not isinstance(metadata, dict):
        raise ValueError("empty request rows or invalid run metadata")
    missing = [field for field in _FIXED_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"{metadata_path}: missing fixed workload fields {missing}")
    for field in ("cases", "repetitions", "max_batch_size"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{metadata_path}: invalid {field}")
    warmup = metadata["warmup"]
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"{metadata_path}: invalid warmup")
    count = len(rows)
    if any(metadata.get(key) != count for key in ("steady_state_requests", "steady_state_completed_requests")):
        raise ValueError(f"{metadata_path}: request counts do not match JSONL")
    if metadata.get("steady_state_output_tokens") != sum(row["output_tokens"] for row in rows.values()):
        raise ValueError(f"{metadata_path}: output token total does not match JSONL")
    if metadata["cases"] * metadata["repetitions"] != count:
        raise ValueError(f"{metadata_path}: cases × repetitions does not match JSONL")
    assessment_flags = [row.get("assessment_valid") for row in rows.values()]
    if any(flag is not None for flag in assessment_flags):
        if not all(isinstance(flag, bool) for flag in assessment_flags):
            raise ValueError(f"{jsonl}: assessment_valid must be a boolean on every row")
        valid_count = sum(assessment_flags)
        if metadata.get("steady_state_valid_assessments") != valid_count:
            raise ValueError(f"{metadata_path}: valid assessment count does not match JSONL")
        if metadata.get("assessment_valid_rate") != valid_count / count:
            raise ValueError(f"{metadata_path}: assessment valid rate does not match JSONL")
    concurrency = metadata.get("concurrency")
    throughput = metadata.get("aggregate_output_tokens_per_second")
    wall_ms = metadata.get("steady_state_wall_clock_ms")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError(f"{metadata_path}: invalid concurrency")
    if isinstance(throughput, bool) or not isinstance(throughput, (int, float)) or not math.isfinite(throughput) or throughput <= 0:
        raise ValueError(f"{metadata_path}: invalid aggregate throughput")
    if isinstance(wall_ms, bool) or not isinstance(wall_ms, (int, float)) or not math.isfinite(wall_ms) or wall_ms <= 0:
        raise ValueError(f"{metadata_path}: invalid steady-state wall time")
    actual_throughput = metadata["steady_state_output_tokens"] / (wall_ms / 1000)
    if not math.isclose(throughput, actual_throughput, rel_tol=1e-6):
        raise ValueError(f"{metadata_path}: throughput does not match tokens and wall time")
    return rows, metadata


def compare(
    *, before_jsonl: Path, before_metadata: Path,
    candidate_jsonl: Path, candidate_metadata: Path,
    after_jsonl: Path, after_metadata: Path,
) -> dict[str, Any]:
    inputs = ((before_jsonl, before_metadata), (candidate_jsonl, candidate_metadata),
              (after_jsonl, after_metadata))
    loaded = [_load_run(*pair) for pair in inputs]
    row_maps = [run[0] for run in loaded]
    metadata = [run[1] for run in loaded]
    fixed = [{field: item.get(field) for field in _FIXED_FIELDS} for item in metadata]
    if fixed[0] != fixed[1] or fixed[0] != fixed[2]:
        raise ValueError("run metadata fixed workload fields differ")
    if metadata[0]["concurrency"] != metadata[2]["concurrency"]:
        raise ValueError("baseline concurrency changed between A runs")
    if metadata[1]["concurrency"] <= metadata[0]["concurrency"]:
        raise ValueError("candidate concurrency must exceed baseline concurrency")
    keys = set(row_maps[0])
    if any(set(rows) != keys for rows in row_maps[1:]):
        raise ValueError("run sample/repetition identities differ")
    validity_recorded = ["assessment_valid" in next(iter(rows.values())) for rows in row_maps]
    if len(set(validity_recorded)) != 1:
        raise ValueError("assessment_valid recording differs between runs")
    mismatches = []
    invalid_assessments = []
    for key in sorted(keys):
        outputs = [rows[key] for rows in row_maps]
        if len({row["output_sha256"] for row in outputs}) != 1 or len({row["output_tokens"] for row in outputs}) != 1:
            mismatches.append({"sample_id": key[0], "repetition": key[1],
                               "output_sha256": [row["output_sha256"] for row in outputs],
                               "output_tokens": [row["output_tokens"] for row in outputs]})
        if validity_recorded[0] and not all(row["assessment_valid"] for row in outputs):
            invalid_assessments.append({"sample_id": key[0], "repetition": key[1],
                                        "assessment_valid": [row["assessment_valid"] for row in outputs]})
    throughputs = [item["aggregate_output_tokens_per_second"] for item in metadata]
    faster_than_both = throughputs[1] > max(throughputs[0], throughputs[2])
    return {
        "schema_version": "parksight_edgellm_concurrency_aba_comparison_v1",
        "passed": not mismatches and not invalid_assessments and faster_than_both,
        "exact_output_equal": not mismatches,
        "assessment_contract_checked": validity_recorded[0],
        "invalid_assessments": invalid_assessments,
        "candidate_faster_than_both_baselines": faster_than_both,
        "sample_count": len(keys),
        "fixed_workload": fixed[0],
        "concurrency_order": [item["concurrency"] for item in metadata],
        "aggregate_output_tokens_per_second": throughputs,
        "candidate_to_baseline_mean_ratio": throughputs[1] / mean((throughputs[0], throughputs[2])),
        "baseline_drift_percent": (throughputs[2] / throughputs[0] - 1) * 100,
        "exact_output_mismatches": mismatches,
        "inputs": [
            {"jsonl": str(jsonl), "jsonl_sha256": _sha256(jsonl),
             "run_metadata": str(meta), "run_metadata_sha256": _sha256(meta)}
            for jsonl, meta in inputs
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "candidate", "after"):
        parser.add_argument(f"--{name}-jsonl", type=Path, required=True)
        parser.add_argument(f"--{name}-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(
        before_jsonl=args.before_jsonl, before_metadata=args.before_metadata,
        candidate_jsonl=args.candidate_jsonl, candidate_metadata=args.candidate_metadata,
        after_jsonl=args.after_jsonl, after_metadata=args.after_metadata,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
