"""从已定稿的 review package 中选择无泄漏、事件覆盖优先的 INT4 校准组。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import ParkingAssessment, ParkingRiskEvent


def load_finalized_package(path: Path) -> list[dict[str, Any]]:
    """读取并校验所有记录均已人工定稿的 review package。"""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"package row must be an object: {path}:{line_number}")
        if record.get("review_status") not in {"confirmed", "corrected"}:
            raise ValueError(
                f"package must be fully finalized before calibration selection: "
                f"{record.get('case_id')} has {record.get('review_status')!r}"
            )
        assessment = record.get("human_assessment")
        ParkingAssessment.from_mapping(assessment)
        if record.get("split") != "train":
            continue
        group_id = str(record.get("source_group_id", "")).strip()
        if not group_id:
            raise ValueError(f"train record has blank source_group_id: {record.get('case_id')}")
        rows.append(record)
    if not rows:
        raise ValueError("finalized package has no train records")
    return rows


def select_calibration_groups(
    records: list[dict[str, Any]],
    *,
    sample_count: int,
    calibration_id: str,
    source_dataset: str,
) -> dict[str, Any]:
    """用确定性的贪心集合覆盖选择 calibration source_group_ids。"""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(records):
        raise ValueError("sample_count cannot exceed finalized train records")
    if not calibration_id.strip() or not source_dataset.strip():
        raise ValueError("calibration_id and source_dataset must not be blank")

    candidates: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for record in records:
        group_id = str(record["source_group_id"])
        if group_id in seen_groups:
            raise ValueError(f"duplicate train source_group_id: {group_id}")
        seen_groups.add(group_id)
        assessment = ParkingAssessment.from_mapping(record["human_assessment"])
        candidates.append(
            {
                "group_id": group_id,
                "risk_level": assessment.risk_level.value,
                "events": {event.value for event in assessment.events},
            }
        )
    candidates.sort(key=lambda candidate: candidate["group_id"])

    selected: list[dict[str, Any]] = []
    covered_events: set[str] = set()
    covered_risks: set[str] = set()
    remaining = candidates[:]
    while remaining and len(selected) < sample_count:
        def score(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
            new_events = len(candidate["events"] - covered_events)
            new_risk = int(candidate["risk_level"] not in covered_risks)
            return (new_events, new_risk, len(candidate["events"]), candidate["group_id"])

        best = max(remaining, key=score)
        remaining.remove(best)
        selected.append(best)
        covered_events.update(best["events"])
        covered_risks.add(best["risk_level"])

    selected_groups = [candidate["group_id"] for candidate in selected]
    missing_events = [
        event.value for event in ParkingRiskEvent if event.value not in covered_events
    ]
    return {
        "calibration_id": calibration_id,
        "source_dataset": source_dataset,
        "selection_policy": (
            "仅从已人工定稿的 train 来源组中确定性选择；每步优先新增风险事件覆盖，"
            "再覆盖未出现的风险等级，不得用于 LoRA train、validation 或冻结测试。"
        ),
        "source_group_ids": selected_groups,
        "selection_audit": {
            "candidate_train_groups": len(candidates),
            "selected_groups": len(selected_groups),
            "covered_events": sorted(covered_events),
            "missing_events": missing_events,
            "covered_risk_levels": sorted(covered_risks),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = select_calibration_groups(
        load_finalized_package(args.package),
        sample_count=args.sample_count,
        calibration_id=args.calibration_id,
        source_dataset=args.source_dataset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
