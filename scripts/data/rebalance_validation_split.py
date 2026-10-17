"""在不跨来源组的前提下扩大 human-confirmed validation split。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"JSONL file must not be empty: {path}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _risk_move_counts(
    train: list[dict[str, Any]], move_count: int
) -> dict[str, int]:
    counts = Counter(row["assessment"]["risk_level"] for row in train)
    raw = {risk: move_count * count / len(train) for risk, count in counts.items()}
    selected = {risk: int(value) for risk, value in raw.items()}
    remainder = move_count - sum(selected.values())
    order = sorted(raw, key=lambda risk: (-(raw[risk] - selected[risk]), risk))
    for risk in order[:remainder]:
        selected[risk] += 1
    return selected


def rebalance_records(
    records: list[dict[str, Any]],
    *,
    target_validation_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_validation_count <= 0 or target_validation_count >= len(records):
        raise ValueError("target_validation_count must be between 1 and total_count-1")
    case_ids = [str(row["sample_id"]) for row in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("dataset contains duplicate sample_id")
    groups = [str(row["source_group_id"]) for row in records]
    if len(groups) != len(set(groups)):
        raise ValueError("dataset must contain one record per source group")
    if any(row.get("split") not in {"train", "validation"} for row in records):
        raise ValueError("dataset must contain only train and validation records")

    current_validation = [row for row in records if row["split"] == "validation"]
    current_train = [row for row in records if row["split"] == "train"]
    moved_to_train: set[str] = set()
    moved_to_validation: set[str] = set()

    train_event_counts = Counter(
        event
        for row in current_train
        for event in row.get("assessment", {}).get("events", [])
    )
    validation_only_events = {
        event
        for row in current_validation
        for event in row.get("assessment", {}).get("events", [])
        if train_event_counts[event] == 0
    }
    for event in sorted(validation_only_events):
        donors = [
            row
            for row in current_validation
            if event in row.get("assessment", {}).get("events", [])
        ]
        donor = min(donors, key=lambda row: _stable_key(row, seed))
        protected_event_names = {
            event_name
            for event_name, count in train_event_counts.items()
            if count <= 1
        }
        movable = [
            row
            for row in current_train
            if not set(row.get("assessment", {}).get("events", [])).intersection(
                protected_event_names
            )
        ]
        if not movable:
            raise ValueError(
                "no train record available to swap for validation-only event: "
                f"{event}"
            )
        same_risk = [
            row
            for row in movable
            if row["assessment"]["risk_level"] == donor["assessment"]["risk_level"]
        ]
        replacement = min(same_risk or movable, key=lambda row: _stable_key(row, seed))
        current_validation.remove(donor)
        current_train.remove(replacement)
        current_train.append(donor)
        current_validation.append(replacement)
        moved_to_train.add(str(donor["sample_id"]))
        moved_to_validation.add(str(replacement["sample_id"]))
        train_event_counts.update(donor.get("assessment", {}).get("events", []))

    move_count = target_validation_count - len(current_validation)
    if move_count <= 0:
        raise ValueError("target validation count must exceed current validation count")
    if move_count >= len(current_train):
        raise ValueError("target validation count leaves no training records")

    event_counts = Counter(
        event
        for row in current_train
        for event in row.get("assessment", {}).get("events", [])
    )
    protected_events = {event for event, count in event_counts.items() if count <= 1}
    candidates = [
        row
        for row in current_train
        if not protected_events.intersection(row.get("assessment", {}).get("events", []))
    ]
    move_by_risk = _risk_move_counts(current_train, move_count)
    selected_ids: list[str] = []
    for risk, count in sorted(move_by_risk.items()):
        pool = [row for row in candidates if row["assessment"]["risk_level"] == risk]
        pool.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['sample_id']}".encode("utf-8")
            ).hexdigest()
        )
        if len(pool) < count:
            raise ValueError(
                f"not enough movable {risk} records: required={count}, available={len(pool)}"
            )
        selected_ids.extend(str(row["sample_id"]) for row in pool[:count])

    selected = set(selected_ids) | moved_to_validation
    updated = [
        {
            **row,
            "split": (
                "train"
                if str(row["sample_id"]) in moved_to_train
                else "validation"
                if str(row["sample_id"]) in selected
                else row["split"]
            ),
        }
        for row in records
    ]
    if sum(row["split"] == "validation" for row in updated) != target_validation_count:
        raise AssertionError("rebalance produced an unexpected validation count")
    audit = {
        "seed": seed,
        "original_train_count": len(current_train),
        "original_validation_count": len(current_validation),
        "moved_to_validation": len(selected_ids),
        "target_validation_count": target_validation_count,
        "protected_events": sorted(protected_events),
        "swapped_validation_only_events_to_train": sorted(validation_only_events),
        "swapped_to_train_case_ids": sorted(moved_to_train),
        "swapped_to_validation_case_ids": sorted(moved_to_validation),
        "moved_case_ids": sorted(selected_ids),
        "moved_risk_counts": dict(sorted(Counter(
            row["assessment"]["risk_level"]
            for row in current_train
            if str(row["sample_id"]) in selected
        ).items())),
        "final_split_counts": dict(sorted(Counter(row["split"] for row in updated).items())),
    }
    return updated, audit


def _stable_key(row: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(
        f"{seed}:{row['sample_id']}".encode("utf-8")
    ).hexdigest()


def rebalance_manifest(
    manifest: list[dict[str, Any]], dataset: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    split_by_case = {str(row["sample_id"]): row["split"] for row in dataset}
    manifest_ids = {str(row["case_id"]) for row in manifest}
    missing_from_manifest = set(split_by_case) - manifest_ids
    if missing_from_manifest:
        raise ValueError(
            "dataset case_ids missing from manifest: "
            f"{sorted(missing_from_manifest)}"
        )
    return [
        {**row, "split": split_by_case[str(row["case_id"])]}
        if str(row["case_id"]) in split_by_case
        else row
        for row in manifest
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--target-validation-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    records = _load_jsonl(args.dataset)
    updated, audit = rebalance_records(
        records,
        target_validation_count=args.target_validation_count,
        seed=args.seed,
    )
    _write_jsonl(args.output_dataset, updated)
    _write_jsonl(args.output_manifest, rebalance_manifest(_load_jsonl(args.manifest), updated))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
