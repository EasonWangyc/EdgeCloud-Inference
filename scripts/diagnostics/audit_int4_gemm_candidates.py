"""审计固定 Edge-LLM W4A16 GEMM 候选的静态资源和 shape 约束。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any


DEFAULT_OUTPUT_DIMENSIONS = (2048, 1024, 6144)
DEFAULT_SHARED_MEMORY_LIMIT_BYTES = 99 * 1024
DEFAULT_MAX_THREADS_PER_BLOCK = 1024

_CANDIDATES: tuple[dict[str, int | str], ...] = (
    {"candidate_id": "control_cta_n128", "cta_m": 64, "cta_n": 128, "cta_k": 64, "stages": 4},
    {"candidate_id": "cta_n256", "cta_m": 64, "cta_n": 256, "cta_k": 64, "stages": 4},
    {"candidate_id": "cta_m128", "cta_m": 128, "cta_n": 128, "cta_k": 64, "stages": 4},
    {"candidate_id": "cta_k128", "cta_m": 64, "cta_n": 128, "cta_k": 128, "stages": 4},
    {"candidate_id": "cta_n512", "cta_m": 64, "cta_n": 512, "cta_k": 64, "stages": 4},
)


def estimate_shared_memory_bytes(
    *, cta_m: int, cta_n: int, cta_k: int, stages: int, element_size_bytes: int = 2
) -> int:
    """Match v0.9.1 int4WoqGemmCuda.cu's kSmemByteSize formula."""
    values = (cta_m, cta_n, cta_k, stages, element_size_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("CTA dimensions, stages and element size must be positive integers")
    # v0.9.1 defines SMEM_PAD_A/B as 0 and kInterleave as 4.
    return (cta_m * cta_k + cta_n * cta_k // 4 + cta_n) * stages * element_size_bytes


def audit_candidates(
    *,
    output_dimensions: Sequence[int] = DEFAULT_OUTPUT_DIMENSIONS,
    shared_memory_limit_bytes: int = DEFAULT_SHARED_MEMORY_LIMIT_BYTES,
    max_threads_per_block: int = DEFAULT_MAX_THREADS_PER_BLOCK,
) -> dict[str, Any]:
    """Return a static audit; successful compilation and runtime remain separate gates."""
    dimensions = tuple(output_dimensions)
    if not dimensions or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("output_dimensions must contain positive integers")
    if shared_memory_limit_bytes <= 0 or max_threads_per_block <= 0:
        raise ValueError("resource limits must be positive integers")

    records: list[dict[str, Any]] = []
    for candidate in _CANDIDATES:
        cta_m = int(candidate["cta_m"])
        cta_n = int(candidate["cta_n"])
        cta_k = int(candidate["cta_k"])
        stages = int(candidate["stages"])
        shared_memory_bytes = estimate_shared_memory_bytes(
            cta_m=cta_m, cta_n=cta_n, cta_k=cta_k, stages=stages
        )
        num_warps = (cta_m // 64) * (cta_n // 32)
        threads_per_block = num_warps * 32
        shape_aligned = all(dimension % cta_n == 0 for dimension in dimensions)
        resource_ok = (
            shared_memory_bytes < shared_memory_limit_bytes
            and threads_per_block <= max_threads_per_block
        )
        records.append(
            {
                **candidate,
                "shared_memory_bytes": shared_memory_bytes,
                "shared_memory_limit_bytes": shared_memory_limit_bytes,
                "num_warps": num_warps,
                "threads_per_block": threads_per_block,
                "output_dimensions": list(dimensions),
                "shape_aligned": shape_aligned,
                "resource_ok": resource_ok,
                "build_queue_eligible": resource_ok and shape_aligned,
            }
        )

    return {
        "schema_version": "parksight_int4_gemm_candidate_audit_v1",
        "source_assumptions": {
            "smem_pad_a": 0,
            "smem_pad_b": 0,
            "k_interleave": 4,
            "element_size_bytes": 2,
            "static_assert": f"kSmemByteSize < {shared_memory_limit_bytes}",
        },
        "candidates": records,
        "evidence_boundary": (
            "Static resource and shape checks do not prove compilation, numerical correctness, "
            "kernel selection or performance; retain build, alignment, Nsight and soak gates."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str)
    args = parser.parse_args(argv)
    report = audit_candidates()
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        from pathlib import Path

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(output)
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
