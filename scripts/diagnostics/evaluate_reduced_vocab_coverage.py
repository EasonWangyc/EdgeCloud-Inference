"""Evaluate a reduced vocabulary map against a holdout JSONL corpus."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts.experiments.build_reduced_vocab_map import _get_dotted_value, _string_values
    from scripts.diagnostics.validate_reduced_vocab import (
        read_mapping_token_ids,
        validate_mapping,
    )
except ModuleNotFoundError:
    # Support direct execution as ``python scripts/<name>.py``.
    from build_reduced_vocab_map import _get_dotted_value, _string_values
    from validate_reduced_vocab import read_mapping_token_ids, validate_mapping


def evaluate_coverage(
    *,
    tokenizer: Any,
    map_path: Path,
    metadata_path: Path,
    original_vocab_size: int,
    input_paths: Sequence[Path],
    fields: Sequence[str],
    required_token_ids: Iterable[int] = (),
    study_report_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Measure reference/output-token coverage without changing the map.

    ``input_paths`` contains JSONL records such as annotations or reference
    outputs.  ``study_report_paths`` optionally adds the actual ``raw_output``
    strings captured by prior StudyReport runs.  Keeping the two sources
    separate makes the evidence auditable and prevents a generated output from
    being mistaken for a human reference annotation.
    """
    validation = validate_mapping(
        map_path=map_path,
        metadata_path=metadata_path,
        original_vocab_size=original_vocab_size,
        required_token_ids=tuple(required_token_ids),
    )
    if not validation["valid"]:
        raise ValueError(f"invalid reduced-vocabulary map: {validation['reasons']}")

    allowed = set(read_mapping_token_ids(map_path))
    if not input_paths and not study_report_paths:
        raise ValueError("coverage requires at least one JSONL or StudyReport input")

    sample_count = 0
    string_count = 0
    total_tokens = 0
    covered_tokens = 0
    samples_with_missing: list[dict[str, Any]] = []
    missing_token_counts: dict[int, int] = {}
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_count += 1
                sample_missing: set[int] = set()
                sample_tokens = 0
                sample_covered = 0
                for field in fields:
                    value = _get_dotted_value(record, field)
                    for text in _string_values(value):
                        string_count += 1
                        token_ids = [
                            int(token_id)
                            for token_id in tokenizer.encode(
                                text,
                                add_special_tokens=False,
                            )
                        ]
                        sample_tokens += len(token_ids)
                        sample_covered += sum(token_id in allowed for token_id in token_ids)
                        sample_missing.update(
                            token_id for token_id in token_ids if token_id not in allowed
                        )
                if sample_tokens == 0:
                    raise ValueError(
                        f"no tokenizable values found in {input_path}:{line_number}"
                    )
                total_tokens += sample_tokens
                covered_tokens += sample_covered
                if sample_missing:
                    samples_with_missing.append(
                        {
                            "input_jsonl": str(input_path),
                            "line": line_number,
                            "missing_token_ids": sorted(sample_missing),
                            "missing_token_count": len(sample_missing),
                            "token_count": sample_tokens,
                            "covered_token_count": sample_covered,
                        }
                    )
                    for token_id in sample_missing:
                        missing_token_counts[token_id] = (
                            missing_token_counts.get(token_id, 0) + 1
                        )

    for report_path in study_report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ValueError(f"StudyReport must be an object: {report_path}")
        records = report.get("records")
        if not isinstance(records, list):
            raise ValueError(f"StudyReport records must be a list: {report_path}")
        for record_index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"StudyReport record {record_index} must be an object: {report_path}"
                )
            raw_output = record.get("raw_output")
            if not isinstance(raw_output, str) or not raw_output.strip():
                # Failed requests and reports without raw output are not
                # token evidence; leave them visible in the report instead of
                # silently counting an empty sample as covered.
                continue
            sample_count += 1
            string_count += 1
            token_ids = [
                int(token_id)
                for token_id in tokenizer.encode(
                    raw_output,
                    add_special_tokens=False,
                )
            ]
            if not token_ids:
                raise ValueError(
                    f"StudyReport record {record_index} has no tokenizable raw_output: "
                    f"{report_path}"
                )
            sample_tokens = len(token_ids)
            sample_covered = sum(token_id in allowed for token_id in token_ids)
            sample_missing = {
                token_id for token_id in token_ids if token_id not in allowed
            }
            total_tokens += sample_tokens
            covered_tokens += sample_covered
            if sample_missing:
                samples_with_missing.append(
                    {
                        "study_report": str(report_path),
                        "record_index": record_index,
                        "case_id": record.get("case_id"),
                        "missing_token_ids": sorted(sample_missing),
                        "missing_token_count": len(sample_missing),
                        "token_count": sample_tokens,
                        "covered_token_count": sample_covered,
                    }
                )
                for token_id in sample_missing:
                    missing_token_counts[token_id] = (
                        missing_token_counts.get(token_id, 0) + 1
                    )

    if sample_count == 0:
        raise ValueError("coverage input contains no tokenizable records")
    return {
        "schema_version": "parksight_reduced_vocab_coverage_v1",
        "map": validation["map"],
        "metadata": validation["metadata"],
        "input_jsonl": [str(path) for path in input_paths],
        "study_reports": [str(path) for path in study_report_paths],
        "fields": list(fields),
        "reference_sample_count": sample_count,
        "tokenized_string_count": string_count,
        "total_reference_tokens": total_tokens,
        "covered_reference_tokens": covered_tokens,
        "token_coverage_fraction": (
            covered_tokens / total_tokens if total_tokens else 0.0
        ),
        "samples_with_missing_tokens": len(samples_with_missing),
        "sample_coverage_fraction": (
            1.0 - len(samples_with_missing) / sample_count
        ),
        "missing_token_counts_by_sample": {
            str(token_id): count
            for token_id, count in sorted(missing_token_counts.items())
        },
        "samples": samples_with_missing,
        "valid": not samples_with_missing,
        "evidence_boundary": (
            "Reference-token coverage only; it does not prove model output quality "
            "or inference performance"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path, dest="map_path")
    parser.add_argument("--metadata", required=True, type=Path, dest="metadata_path")
    parser.add_argument("--original-vocab-size", required=True, type=int)
    parser.add_argument("--input-jsonl", action="append", default=[], type=Path)
    parser.add_argument(
        "--study-report",
        action="append",
        default=[],
        type=Path,
        help="可选 StudyReport JSON；对其中每条成功记录的 raw_output 做覆盖检查",
    )
    parser.add_argument("--field", action="append", default=None)
    parser.add_argument("--required-token-id", action="append", type=int, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    fields = args.field or ["assessment"]
    if not args.input_jsonl and not args.study_report:
        parser.error("at least one --input-jsonl or --study-report is required")
    if any(not path.is_file() for path in [*args.input_jsonl, *args.study_report]):
        parser.error("all coverage input paths must be files")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for coverage evaluation") from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
    )
    result = evaluate_coverage(
        tokenizer=tokenizer,
        map_path=args.map_path,
        metadata_path=args.metadata_path,
        original_vocab_size=args.original_vocab_size,
        input_paths=args.input_jsonl,
        fields=fields,
        required_token_ids=tuple(args.required_token_id),
        study_report_paths=args.study_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
