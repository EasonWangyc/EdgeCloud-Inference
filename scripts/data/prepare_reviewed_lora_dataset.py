"""校验视觉复核候选标注，并生成无泄漏的 LoRA 与 INT4 校准数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from parksight_vlm.assessment import ParkingAssessment, ParkingRiskEvent
from parksight_vlm.workload import FrozenWorkload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError(f"JSONL file must not be empty: {path}")
    return records


def _load_annotations(path: Path) -> dict[str, ParkingAssessment]:
    annotations: dict[str, ParkingAssessment] = {}
    for record in _load_jsonl(path):
        if set(record) != {"case_id", "assessment"}:
            raise ValueError("review annotation must contain only case_id and assessment")
        case_id = str(record["case_id"])
        if case_id in annotations:
            raise ValueError(f"duplicate review annotation: {case_id}")
        annotations[case_id] = ParkingAssessment.from_mapping(record["assessment"])
    return annotations


def _load_development_records(
    manifest_path: Path,
    weak_annotations_path: Path,
) -> list[dict[str, Any]]:
    weak_annotations = _load_annotations(weak_annotations_path)
    records: list[dict[str, Any]] = []
    manifest_case_ids: set[str] = set()
    for manifest in _load_jsonl(manifest_path):
        expected_fields = {"case_id", "image_ref", "source_group_id", "split"}
        if set(manifest) != expected_fields:
            raise ValueError("development manifest has invalid fields")
        case_id = str(manifest["case_id"])
        if case_id in manifest_case_ids:
            raise ValueError(f"duplicate development case_id: {case_id}")
        manifest_case_ids.add(case_id)
        if case_id not in weak_annotations:
            raise ValueError(f"missing weak annotation: {case_id}")
        records.append(
            {
                "sample_id": case_id,
                "image": str(manifest["image_ref"]),
                "source_group_id": str(manifest["source_group_id"]),
                "split": str(manifest["split"]),
                "assessment": weak_annotations[case_id].to_mapping(),
            }
        )
    unexpected_annotations = set(weak_annotations) - manifest_case_ids
    if unexpected_annotations:
        raise ValueError(
            f"weak annotations without manifest records: {sorted(unexpected_annotations)}"
        )
    return records


def _load_calibration_groups(path: Path) -> tuple[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("source_group_ids")
    if not isinstance(groups, list) or not groups:
        raise ValueError("calibration source_group_ids must be a non-empty array")
    normalized = {str(group) for group in groups}
    if len(normalized) != len(groups):
        raise ValueError("calibration source_group_ids must not contain duplicates")
    return str(payload["calibration_id"]), normalized


def _frozen_test_groups(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {str(record["source_group_id"]) for record in _load_jsonl(path)}


def _calibration_text(workload: FrozenWorkload, assessment: ParkingAssessment) -> str:
    """生成 LLM backbone 使用的领域文本校准样本。

    TensorRT Edge-LLM 当前只量化语言骨干，因此这里保留泊车任务的系统指令、
    结构化用户指令和人工答案；图片路径单独写入记录用于追溯，不伪装成视觉校准。
    """
    answer = json.dumps(
        assessment.to_mapping(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "\n".join(
        (
            f"系统指令：{workload.system_prompt}",
            f"用户指令：{workload.render_user_prompt()}",
            f"助手输出：{answer}",
        )
    )


def _event_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        event
        for record in records
        for event in record["assessment"]["events"]
    )
    return {event.value: counts.get(event.value, 0) for event in ParkingRiskEvent}


def audit_event_coverage(
    *,
    lora_records: list[dict[str, Any]],
    calibration_records: list[dict[str, Any]],
    recommended_min_train_support: int = 3,
) -> dict[str, Any]:
    """报告 train/validation/calibration 的事件覆盖，不改变数据拆分。"""
    if recommended_min_train_support < 1:
        raise ValueError("recommended_min_train_support must be at least 1")
    train_records = [record for record in lora_records if record["split"] == "train"]
    validation_records = [
        record for record in lora_records if record["split"] == "validation"
    ]
    train_counts = _event_counts(train_records)
    validation_counts = _event_counts(validation_records)
    calibration_counts = _event_counts(calibration_records)
    warnings: list[dict[str, Any]] = []
    for event in ParkingRiskEvent:
        event_name = event.value
        train_count = train_counts[event_name]
        validation_count = validation_counts[event_name]
        calibration_count = calibration_counts[event_name]
        if train_count == 0:
            warnings.append({"type": "missing_from_train", "event": event_name})
        elif train_count < recommended_min_train_support:
            warnings.append(
                {
                    "type": "low_train_support",
                    "event": event_name,
                    "count": train_count,
                    "recommended_minimum": recommended_min_train_support,
                }
            )
        if validation_count > 0 and train_count == 0:
            warnings.append(
                {"type": "validation_not_seen_in_train", "event": event_name}
            )
        if calibration_count == 0:
            warnings.append(
                {"type": "missing_from_calibration", "event": event_name}
            )
    return {
        "train": train_counts,
        "validation": validation_counts,
        "calibration": calibration_counts,
        "recommended_min_train_support": recommended_min_train_support,
        "warnings": warnings,
    }


def enforce_event_coverage(
    coverage: dict[str, Any],
    *,
    min_train_event_support: int = 3,
    require_validation_event_coverage: bool = True,
    require_calibration_event_coverage: bool = True,
) -> None:
    """将 train/validation/calibration 的六类事件覆盖变为硬门禁。"""
    if min_train_event_support < 1:
        raise ValueError("min_train_event_support must be at least 1")
    violations: list[str] = []
    for event in ParkingRiskEvent:
        event_name = event.value
        train_count = coverage["train"].get(event_name, 0)
        validation_count = coverage["validation"].get(event_name, 0)
        calibration_count = coverage["calibration"].get(event_name, 0)
        if train_count < min_train_event_support:
            violations.append(
                f"train:{event_name}={train_count}<{min_train_event_support}"
            )
        if require_validation_event_coverage and validation_count == 0:
            violations.append(f"validation:{event_name}=0")
        if require_calibration_event_coverage and calibration_count == 0:
            violations.append(f"calibration:{event_name}=0")
    if violations:
        raise ValueError("event coverage gate failed: " + ", ".join(violations))


def prepare_datasets(
    *,
    teacher_records: list[dict[str, Any]],
    annotations: dict[str, ParkingAssessment],
    calibration_id: str,
    calibration_groups: set[str],
    frozen_test_groups: set[str],
    image_root: Path,
    workload: FrozenWorkload,
    label_source: str = "codex_visual_review_v1_single_pass",
    dataset_id: str = "ps80_reviewed_v1",
    require_event_coverage: bool = False,
    min_train_event_support: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """返回 LoRA、校准记录和审计摘要。"""
    if not label_source.strip():
        raise ValueError("label_source must not be blank")
    if not dataset_id.strip():
        raise ValueError("dataset_id must not be blank")
    case_ids = [str(record["sample_id"]) for record in teacher_records]
    groups = [str(record["source_group_id"]) for record in teacher_records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("teacher records contain duplicate sample_id values")
    if len(groups) != len(set(groups)):
        raise ValueError("teacher records must contain one image per source group")

    missing_annotations = set(case_ids) - set(annotations)
    unexpected_annotations = set(annotations) - set(case_ids)
    if missing_annotations or unexpected_annotations:
        raise ValueError(
            "review annotations do not match teacher records: "
            f"missing={sorted(missing_annotations)}, "
            f"unexpected={sorted(unexpected_annotations)}"
        )

    known_groups = set(groups)
    unknown_calibration = calibration_groups - known_groups
    if unknown_calibration:
        raise ValueError(f"unknown calibration groups: {sorted(unknown_calibration)}")
    non_train_calibration = {
        str(record["source_group_id"])
        for record in teacher_records
        if str(record["source_group_id"]) in calibration_groups
        and record["split"] != "train"
    }
    if non_train_calibration:
        raise ValueError(
            "calibration groups must come from the original training pool: "
            f"{sorted(non_train_calibration)}"
        )
    overlap_with_test = known_groups & frozen_test_groups
    if overlap_with_test:
        raise ValueError(f"training pool overlaps frozen test groups: {sorted(overlap_with_test)}")

    lora_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    for teacher in teacher_records:
        case_id = str(teacher["sample_id"])
        group_id = str(teacher["source_group_id"])
        image_name = Path(str(teacher["image"])).name
        image_path = image_root / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"missing source image: {image_path}")
        assessment = annotations[case_id]
        common = {
            "sample_id": case_id,
            "image": image_path.as_posix(),
            "source_group_id": group_id,
            "label_source": label_source,
            "workload_identity": workload.identity,
            "assessment": assessment.to_mapping(),
            "weak_assessment": teacher["assessment"],
        }
        if group_id in calibration_groups:
            calibration_records.append(
                {
                    **common,
                    "split": "calibration",
                    "calibration_id": calibration_id,
                    "text": _calibration_text(workload, assessment),
                }
            )
        else:
            split = str(teacher["split"])
            if split not in {"train", "validation"}:
                raise ValueError(f"unsupported supervised split: {split}")
            lora_records.append({**common, "split": split})

    lora_groups = {record["source_group_id"] for record in lora_records}
    calibration_output_groups = {
        record["source_group_id"] for record in calibration_records
    }
    if lora_groups & calibration_output_groups:
        raise AssertionError("LoRA and calibration groups must be disjoint")

    risk_counts = Counter(
        record["assessment"]["risk_level"] for record in lora_records
    )
    event_counts = Counter(
        event
        for record in lora_records
        for event in record["assessment"]["events"]
    )
    weak_changed = sum(
        record["assessment"] != record["weak_assessment"]
        for record in (*lora_records, *calibration_records)
    )
    risk_changed = sum(
        record["assessment"]["risk_level"]
        != record["weak_assessment"]["risk_level"]
        for record in (*lora_records, *calibration_records)
    )
    events_changed = sum(
        set(record["assessment"]["events"])
        != set(record["weak_assessment"]["events"])
        for record in (*lora_records, *calibration_records)
    )
    event_coverage = audit_event_coverage(
        lora_records=lora_records,
        calibration_records=calibration_records,
        recommended_min_train_support=min_train_event_support,
    )
    if require_event_coverage:
        enforce_event_coverage(
            event_coverage,
            min_train_event_support=min_train_event_support,
        )
    summary = {
        "dataset_id": dataset_id,
        "label_source": label_source,
        "reviewed_samples": len(teacher_records),
        "weak_labels_changed": weak_changed,
        "risk_levels_changed": risk_changed,
        "event_sets_changed": events_changed,
        "lora_samples": len(lora_records),
        "train_samples": sum(record["split"] == "train" for record in lora_records),
        "validation_samples": sum(
            record["split"] == "validation" for record in lora_records
        ),
        "calibration_id": calibration_id,
        "calibration_samples": len(calibration_records),
        "group_overlap": {
            "lora_calibration": len(lora_groups & calibration_output_groups),
            "development_frozen_test": len(known_groups & frozen_test_groups),
        },
        "lora_risk_level_counts": dict(sorted(risk_counts.items())),
        "lora_event_counts": dict(sorted(event_counts.items())),
        "event_coverage": event_coverage,
        "require_event_coverage": require_event_coverage,
        "workload_identity": workload.identity,
    }
    return lora_records, calibration_records, summary


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--weak-annotations", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--calibration-config", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--frozen-test-manifest", type=Path)
    parser.add_argument("--lora-output", required=True, type=Path)
    parser.add_argument("--calibration-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument(
        "--label-source",
        required=True,
        help="本轮 annotation 的来源标识，例如 human_confirmed_v1",
    )
    parser.add_argument(
        "--dataset-id",
        default="ps80_reviewed_v1",
        help="输出数据集的逻辑 identity，候选数据应使用显式 candidate 名称",
    )
    parser.add_argument(
        "--require-event-coverage",
        action="store_true",
        help="要求六类事件满足 train、validation、calibration 覆盖后才写出数据",
    )
    parser.add_argument("--min-train-event-support", type=int, default=3)
    args = parser.parse_args()

    teacher_records = _load_development_records(
        args.development_manifest,
        args.weak_annotations,
    )
    calibration_id, calibration_groups = _load_calibration_groups(
        args.calibration_config
    )
    workload = FrozenWorkload.load(args.workload)
    lora_records, calibration_records, summary = prepare_datasets(
        teacher_records=teacher_records,
        annotations=_load_annotations(args.annotations),
        calibration_id=calibration_id,
        calibration_groups=calibration_groups,
        frozen_test_groups=_frozen_test_groups(args.frozen_test_manifest),
        image_root=args.image_root,
        workload=workload,
        label_source=args.label_source,
        dataset_id=args.dataset_id,
        require_event_coverage=args.require_event_coverage,
        min_train_event_support=args.min_train_event_support,
    )
    _write_jsonl(args.lora_output, lora_records)
    _write_jsonl(args.calibration_output, calibration_records)
    summary["inputs_sha256"] = {
        "development_manifest": hashlib.sha256(
            args.development_manifest.read_bytes()
        ).hexdigest(),
        "weak_annotations": hashlib.sha256(
            args.weak_annotations.read_bytes()
        ).hexdigest(),
        "annotations": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "calibration_config": hashlib.sha256(
            args.calibration_config.read_bytes()
        ).hexdigest(),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
