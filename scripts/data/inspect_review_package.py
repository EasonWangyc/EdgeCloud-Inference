"""检查复核 package 的完成度，并报告是否可以进入定稿。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import ParkingAssessment


PACKAGE_FIELDS = {
    "case_id",
    "image_ref",
    "source_group_id",
    "split",
    "label_source",
    "review_status",
    "candidate_assessment",
    "human_assessment",
    "review_note",
}
FINAL_STATUSES = {"confirmed", "corrected"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"review package must not be empty: {path}")
    return records


def inspect_review_package(package_path: Path) -> dict[str, Any]:
    """Return a read-only completion report without writing any artifact."""
    package = _load_jsonl(package_path)
    statuses: Counter[str] = Counter()
    candidate_risks: Counter[str] = Counter()
    candidate_events: Counter[str] = Counter()
    pending_case_ids: list[str] = []
    invalid_case_ids: list[str] = []
    seen_case_ids: set[str] = set()

    for record in package:
        if set(record) != PACKAGE_FIELDS:
            raise ValueError("review package has invalid fields")
        case_id = str(record["case_id"])
        if not case_id.strip() or case_id in seen_case_ids:
            raise ValueError(f"review package has invalid or duplicate case_id: {case_id!r}")
        seen_case_ids.add(case_id)

        status = str(record["review_status"])
        statuses[status] += 1
        candidate = ParkingAssessment.from_mapping(record["candidate_assessment"])
        candidate_risks[candidate.risk_level] += 1
        candidate_events.update(candidate.events)

        human_payload = record["human_assessment"]
        if status not in FINAL_STATUSES or human_payload is None:
            pending_case_ids.append(case_id)
            continue

        try:
            human = ParkingAssessment.from_mapping(human_payload)
            review_note = record["review_note"]
            if not isinstance(review_note, str):
                raise ValueError("review_note must be a string")
            if status == "confirmed" and candidate.to_mapping() != human.to_mapping():
                raise ValueError("confirmed review differs from candidate")
            if status == "corrected":
                if candidate.to_mapping() == human.to_mapping():
                    raise ValueError("corrected review is identical to candidate")
                if not review_note.strip():
                    raise ValueError("corrected review requires review_note")
        except (TypeError, ValueError, KeyError):
            invalid_case_ids.append(case_id)

    complete = not pending_case_ids and not invalid_case_ids
    return {
        "package_path": str(package_path),
        "sample_count": len(package),
        "review_status_counts": dict(sorted(statuses.items())),
        "candidate_risk_level_counts": dict(sorted(candidate_risks.items())),
        "candidate_event_counts": dict(sorted(candidate_events.items())),
        "pending_count": len(pending_case_ids),
        "pending_case_ids": pending_case_ids,
        "invalid_finalized_count": len(invalid_case_ids),
        "invalid_finalized_case_ids": invalid_case_ids,
        "ready_for_finalize": complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="return exit code 2 unless every sample is finalized and valid",
    )
    args = parser.parse_args()
    report = inspect_review_package(args.package)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_finalize"] or not args.fail_on_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
