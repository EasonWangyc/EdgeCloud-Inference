"""从候选标注和 StudyReport 生成面向人工终审的逐样本错误清单。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_error_review(
    *,
    manifest_path: Path,
    annotations_path: Path,
    study_report_paths: list[Path],
    label_source: str = "codex_visual_review_v1_single_pass",
) -> dict[str, Any]:
    manifests = _load_jsonl(manifest_path)
    annotations = _load_jsonl(annotations_path)
    records = _load_study_records(study_report_paths)

    annotation_by_id = {
        _identifier(row, "case_id"): row["assessment"] for row in annotations
    }
    manifest_by_id = {
        _identifier(row, "case_id"): row for row in manifests
    }
    if set(manifest_by_id) != set(annotation_by_id):
        raise ValueError("manifest and candidate annotation case_ids must match")
    if set(manifest_by_id) != set(records):
        missing = sorted(set(manifest_by_id) - set(records))
        unexpected = sorted(set(records) - set(manifest_by_id))
        raise ValueError(
            f"study reports must cover manifest exactly; missing={missing}, unexpected={unexpected}"
        )

    items: list[dict[str, Any]] = []
    priority_counts: Counter[str] = Counter()
    event_false_positive: Counter[str] = Counter()
    event_false_negative: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    json_valid_count = 0
    risk_match_count = 0
    event_exact_match_count = 0

    for case_id, manifest in manifest_by_id.items():
        record = records[case_id]
        candidate = annotation_by_id[case_id]
        model_assessment = record.get("assessment")
        failure = record.get("failure")
        if model_assessment is not None:
            json_valid_count += 1
            model_events = set(model_assessment.get("events", []))
            candidate_events = set(candidate.get("events", []))
            false_positive = sorted(model_events - candidate_events)
            false_negative = sorted(candidate_events - model_events)
            risk_level_match = (
                model_assessment.get("risk_level") == candidate.get("risk_level")
            )
            if risk_level_match:
                risk_match_count += 1
            if not false_positive and not false_negative:
                event_exact_match_count += 1
            for event in false_positive:
                event_false_positive[event] += 1
            for event in false_negative:
                event_false_negative[event] += 1
        else:
            false_positive = []
            false_negative = []
            risk_level_match = None
            failure_category = _failure_category(failure)
            failure_counts[failure_category] += 1

        if failure is not None:
            priority = "high"
        elif false_positive or false_negative:
            priority = "high"
        elif risk_level_match is False:
            priority = "medium"
        else:
            priority = "low"
        priority_counts[priority] += 1

        items.append(
            {
                "case_id": case_id,
                "image_ref": manifest["image_ref"],
                "source_group_id": manifest["source_group_id"],
                "split": manifest["split"],
                "label_source": label_source,
                "candidate_assessment": candidate,
                "model_assessment": model_assessment,
                "failure": failure,
                "raw_output": record.get("raw_output"),
                "risk_level_match": risk_level_match,
                "event_false_positive": false_positive,
                "event_false_negative": false_negative,
                "review_priority": priority,
            }
        )

    return {
        "schema_version": "parksight_candidate_error_review_v1",
        "label_source": label_source,
        "manifest_path": str(manifest_path.resolve()),
        "annotations_path": str(annotations_path.resolve()),
        "study_report_paths": [str(path.resolve()) for path in study_report_paths],
        "summary": {
            "sample_count": len(items),
            "json_valid_count": json_valid_count,
            "json_failure_count": len(items) - json_valid_count,
            "risk_level_match_count": risk_match_count,
            "event_exact_match_count": event_exact_match_count,
            "event_error_count": sum(
                1
                for item in items
                if item["event_false_positive"] or item["event_false_negative"]
            ),
            "failure_counts": dict(sorted(failure_counts.items())),
            "review_priority_counts": dict(sorted(priority_counts.items())),
            "event_false_positive": dict(sorted(event_false_positive.items())),
            "event_false_negative": dict(sorted(event_false_negative.items())),
        },
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument(
        "--study-report", required=True, action="append", type=Path,
        help="可重复传入；多个分片报告合并为一份清单",
    )
    parser.add_argument("--label-source", default="codex_visual_review_v1_single_pass")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    report = build_error_review(
        manifest_path=args.manifest,
        annotations_path=args.annotations,
        study_report_paths=args.study_report,
        label_source=args.label_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _load_study_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            case_id = _identifier(record, "case_id")
            if case_id in records:
                raise ValueError(f"duplicate study record case_id: {case_id}")
            records[case_id] = record
    return records


def _identifier(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _failure_category(failure: Any) -> str:
    if isinstance(failure, Mapping) and isinstance(failure.get("category"), str):
        return failure["category"]
    return "unknown_failure"


if __name__ == "__main__":
    raise SystemExit(main())
