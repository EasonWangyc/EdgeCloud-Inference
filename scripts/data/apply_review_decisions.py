"""将人工复核决策安全地应用到候选 review package。"""

from __future__ import annotations

import argparse
import json
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
DECISION_FIELDS = {
    "case_id",
    "review_status",
    "human_assessment",
    "review_note",
}
FINAL_STATUSES = {"confirmed", "corrected"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file must not be empty: {path}")
    return rows


def _assessment_mapping(value: Any, *, field_name: str, case_id: str) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"{field_name} is required: {case_id}")
    return ParkingAssessment.from_mapping(value).to_mapping()


def apply_review_decisions(
    *,
    package_path: Path,
    decisions_path: Path,
    output_path: Path,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Apply a subset of final decisions while preserving untouched package rows."""
    package = _load_jsonl(package_path)
    decisions = _load_jsonl(decisions_path)

    package_by_id: dict[str, dict[str, Any]] = {}
    for record in package:
        if set(record) != PACKAGE_FIELDS:
            raise ValueError("review package has invalid fields")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("review package case_id must be a non-blank string")
        if case_id in package_by_id:
            raise ValueError(f"review package has duplicate case_id: {case_id}")
        candidate = ParkingAssessment.from_mapping(record["candidate_assessment"])
        record["candidate_assessment"] = candidate.to_mapping()
        package_by_id[case_id] = record

    decision_ids: set[str] = set()
    for decision in decisions:
        if set(decision) != DECISION_FIELDS:
            raise ValueError(
                "decision must contain exactly case_id, review_status, "
                "human_assessment and review_note"
            )
        case_id = decision.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("decision case_id must be a non-blank string")
        if case_id in decision_ids:
            raise ValueError(f"duplicate decision case_id: {case_id}")
        if case_id not in package_by_id:
            raise ValueError(f"decision case_id is not in package: {case_id}")
        decision_ids.add(case_id)

        status = decision["review_status"]
        if status not in FINAL_STATUSES:
            raise ValueError(
                f"decision must be confirmed or corrected: {case_id} has {status!r}"
            )
        note = decision["review_note"]
        if not isinstance(note, str):
            raise ValueError(f"review_note must be a string: {case_id}")

        record = package_by_id[case_id]
        candidate = ParkingAssessment.from_mapping(record["candidate_assessment"])
        human_value = decision["human_assessment"]
        if status == "confirmed" and human_value is None:
            human = candidate
        else:
            human = ParkingAssessment.from_mapping(
                _assessment_mapping(
                    human_value,
                    field_name="human_assessment",
                    case_id=case_id,
                )
            )

        if status == "confirmed" and candidate.to_mapping() != human.to_mapping():
            raise ValueError(f"confirmed review differs from candidate: {case_id}")
        if status == "corrected":
            if candidate.to_mapping() == human.to_mapping():
                raise ValueError(f"corrected review is identical to candidate: {case_id}")
            if not note.strip():
                raise ValueError(f"corrected review requires review_note: {case_id}")

        record["review_status"] = status
        record["human_assessment"] = human.to_mapping()
        record["review_note"] = note

    finalized_count = sum(
        record["review_status"] in FINAL_STATUSES for record in package_by_id.values()
    )
    if require_complete and finalized_count != len(package_by_id):
        raise ValueError(
            f"review package is incomplete: {finalized_count}/{len(package_by_id)} finalized"
        )

    ordered_package = [package_by_id[record["case_id"]] for record in package]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered_package:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    return {
        "package_path": str(package_path),
        "decisions_path": str(decisions_path),
        "output_path": str(output_path),
        "sample_count": len(ordered_package),
        "decision_count": len(decisions),
        "finalized_count": finalized_count,
        "pending_count": len(ordered_package) - finalized_count,
        "ready_for_finalize": finalized_count == len(ordered_package),
    }


def create_decision_template(*, package_path: Path, output_path: Path) -> dict[str, Any]:
    """Create an editable, intentionally non-final decision JSONL template."""
    package = _load_jsonl(package_path)
    case_ids: set[str] = set()
    template: list[dict[str, Any]] = []
    for record in package:
        if set(record) != PACKAGE_FIELDS:
            raise ValueError("review package has invalid fields")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("review package case_id must be a non-blank string")
        if case_id in case_ids:
            raise ValueError(f"review package has duplicate case_id: {case_id}")
        case_ids.add(case_id)
        ParkingAssessment.from_mapping(record["candidate_assessment"])
        template.append(
            {
                "case_id": case_id,
                "review_status": "candidate",
                "human_assessment": None,
                "review_note": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in template:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return {
        "package_path": str(package_path),
        "output_path": str(output_path),
        "sample_count": len(template),
        "review_status": "candidate",
        "ready_for_apply": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--template-output",
        type=Path,
        help="create a candidate template instead of applying decisions",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="require every package record to be finalized before writing",
    )
    args = parser.parse_args()
    if (args.decisions is None) == (args.template_output is None):
        parser.error("必须且只能提供 --decisions 或 --template-output")
    if args.template_output is not None:
        print(
            json.dumps(
                create_decision_template(
                    package_path=args.package,
                    output_path=args.template_output,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.output is None:
        parser.error("使用 --decisions 时必须提供 --output")
    print(
        json.dumps(
            apply_review_decisions(
                package_path=args.package,
                decisions_path=args.decisions,
                output_path=args.output,
                require_complete=args.require_complete,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
