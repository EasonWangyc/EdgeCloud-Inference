"""将人工确认后的 review package 转换为标准 assessment annotation JSONL。"""

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


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _candidate_human_comparison(
    pairs: list[tuple[ParkingAssessment, ParkingAssessment]],
) -> dict[str, Any]:
    """量化候选标注与人工确认之间的差异。"""
    risk_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    assessment_changed = 0
    risk_level_changed = 0
    event_set_changed = 0
    evidence_changed = 0
    driver_advice_changed = 0

    for candidate, human in pairs:
        candidate_mapping = candidate.to_mapping()
        human_mapping = human.to_mapping()
        if candidate_mapping != human_mapping:
            assessment_changed += 1
        if candidate.risk_level == human.risk_level:
            risk_correct += 1
        else:
            risk_level_changed += 1

        candidate_events = set(candidate.events)
        human_events = set(human.events)
        true_positive += len(candidate_events & human_events)
        false_positive += len(candidate_events - human_events)
        false_negative += len(human_events - candidate_events)
        if candidate_events != human_events:
            event_set_changed += 1
        if candidate.evidence != human.evidence:
            evidence_changed += 1
        if set(candidate.driver_advice) != set(human.driver_advice):
            driver_advice_changed += 1

    sample_count = len(pairs)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        "sample_count": sample_count,
        "risk_level_accuracy": _safe_divide(risk_correct, sample_count),
        "event_micro_precision": precision,
        "event_micro_recall": recall,
        "event_micro_f1": _safe_divide(
            2 * precision * recall,
            precision + recall,
        ),
        "assessment_changed": assessment_changed,
        "risk_levels_changed": risk_level_changed,
        "event_sets_changed": event_set_changed,
        "evidence_changed": evidence_changed,
        "driver_advice_changed": driver_advice_changed,
    }


def finalize_review_package(
    *,
    package_path: Path,
    annotations_output: Path,
    summary_output: Path | None = None,
) -> dict[str, Any]:
    package = _load_jsonl(package_path)
    annotations: list[dict[str, Any]] = []
    comparison_pairs: list[tuple[ParkingAssessment, ParkingAssessment]] = []
    statuses: Counter[str] = Counter()
    for record in package:
        required = {
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
        if set(record) != required:
            raise ValueError("review package has invalid fields")
        case_id = str(record["case_id"])
        if not case_id.strip():
            raise ValueError("review package case_id must not be blank")
        if not isinstance(record["review_note"], str):
            raise ValueError(f"review_note must be a string: {case_id}")
        status = str(record["review_status"])
        statuses[status] += 1
        if status not in {"confirmed", "corrected"}:
            raise ValueError(
                f"review record is not finalized: {case_id} has status {status!r}"
            )
        if record["human_assessment"] is None:
            raise ValueError(f"missing human assessment: {case_id}")
        candidate = ParkingAssessment.from_mapping(record["candidate_assessment"])
        assessment = ParkingAssessment.from_mapping(record["human_assessment"])
        if status == "confirmed" and candidate.to_mapping() != assessment.to_mapping():
            raise ValueError(
                f"confirmed review differs from candidate assessment: {case_id}"
            )
        if status == "corrected" and candidate.to_mapping() == assessment.to_mapping():
            raise ValueError(
                f"corrected review is identical to candidate assessment: {case_id}"
            )
        if status == "corrected" and not record["review_note"].strip():
            raise ValueError(f"corrected review requires review_note: {case_id}")
        comparison_pairs.append((candidate, assessment))
        annotations.append({"case_id": case_id, "assessment": assessment.to_mapping()})

    if len({record["case_id"] for record in annotations}) != len(annotations):
        raise ValueError("review package contains duplicate case_id values")

    annotations_output.parent.mkdir(parents=True, exist_ok=True)
    with annotations_output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in annotations:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    summary = {
        "package_id": "ps80_codex_review_package_v1",
        "source_package": str(package_path),
        "sample_count": len(annotations),
        "review_status_counts": dict(sorted(statuses.items())),
        "candidate_human_comparison": _candidate_human_comparison(comparison_pairs),
        "annotations_output": str(annotations_output),
    }
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--annotations-output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_review_package(
                package_path=args.package,
                annotations_output=args.annotations_output,
                summary_output=args.summary_output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
