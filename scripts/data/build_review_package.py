"""生成可供人工复盘的 Codex 候选标注 package。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import ParkingAssessment


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError(f"JSONL file must not be empty: {path}")
    return records


def build_review_package(
    *,
    manifest_path: Path,
    candidate_annotations_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifests = _load_jsonl(manifest_path)
    candidates: dict[str, ParkingAssessment] = {}
    for record in _load_jsonl(candidate_annotations_path):
        if set(record) != {"case_id", "assessment"}:
            raise ValueError("candidate annotation must contain case_id and assessment")
        case_id = str(record["case_id"])
        if case_id in candidates:
            raise ValueError(f"duplicate candidate annotation: {case_id}")
        candidates[case_id] = ParkingAssessment.from_mapping(record["assessment"])

    package: list[dict[str, Any]] = []
    for manifest in manifests:
        expected_fields = {"case_id", "image_ref", "source_group_id", "split"}
        if set(manifest) != expected_fields:
            raise ValueError("development manifest has invalid fields")
        case_id = str(manifest["case_id"])
        if case_id not in candidates:
            raise ValueError(f"missing candidate annotation: {case_id}")
        package.append(
            {
                "case_id": case_id,
                "image_ref": str(manifest["image_ref"]),
                "source_group_id": str(manifest["source_group_id"]),
                "split": str(manifest["split"]),
                "label_source": "codex_visual_review_v1_single_pass",
                "review_status": "candidate",
                "candidate_assessment": candidates[case_id].to_mapping(),
                "human_assessment": None,
                "review_note": "",
            }
        )

    unexpected = set(candidates) - {str(record["case_id"]) for record in manifests}
    if unexpected:
        raise ValueError(f"candidate annotations without manifest: {sorted(unexpected)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in package:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    risk_counts = Counter(
        record["candidate_assessment"]["risk_level"] for record in package
    )
    event_counts = Counter(
        event
        for record in package
        for event in record["candidate_assessment"]["events"]
    )
    return {
        "package_id": "ps80_codex_review_package_v1",
        "label_source": "codex_visual_review_v1_single_pass",
        "review_status": "candidate",
        "sample_count": len(package),
        "risk_level_counts": dict(sorted(risk_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "output_path": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate-annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_review_package(
                manifest_path=args.manifest,
                candidate_annotations_path=args.candidate_annotations,
                output_path=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
