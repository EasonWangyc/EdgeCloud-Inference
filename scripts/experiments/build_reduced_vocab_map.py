"""Build a task-specific TensorRT Edge-LLM reduced vocabulary map.

The generated map is a candidate artifact.  It must be validated and evaluated
on a frozen holdout before it is used to build a production engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def collect_token_counts(
    tokenizer: Any,
    input_paths: Sequence[Path],
    fields: Sequence[str],
) -> Counter[int]:
    """Collect token frequencies from selected string fields in JSONL files."""
    counts: Counter[int] = Counter()
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in {input_path}:{line_number}"
                    ) from error
                for field in fields:
                    value = _get_dotted_value(record, field)
                    for text in _string_values(value):
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        counts.update(int(token_id) for token_id in ids)
    return counts


def collect_study_report_token_counts(
    tokenizer: Any,
    study_report_paths: Sequence[Path],
) -> Counter[int]:
    """Collect token frequencies from successful historical ``raw_output`` rows."""
    counts: Counter[int] = Counter()
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
            if raw_output is None:
                continue
            if not isinstance(raw_output, str) or not raw_output.strip():
                raise ValueError(
                    f"StudyReport record {record_index} raw_output must be a non-empty string "
                    f"when present: {report_path}"
                )
            counts.update(
                int(token_id)
                for token_id in tokenizer.encode(
                    raw_output,
                    add_special_tokens=False,
                )
            )
    return counts


def collect_decoded_character_coverage_token_ids(tokenizer: Any) -> set[int]:
    """Return token IDs whose standalone decode contains CJK or JSON syntax."""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    decode = getattr(tokenizer, "decode", None)
    if not callable(get_vocab) or not callable(decode):
        raise ValueError(
            "tokenizer must provide get_vocab() and decode() for character coverage"
        )
    token_ids = sorted({int(token_id) for token_id in get_vocab().values()})
    selected: set[int] = set()
    for token_id in token_ids:
        decoded = str(
            decode(
                [token_id],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
        )
        if re.search(r"[\u3400-\u9fff]", decoded) or re.search(
            r'[{}\[\]":,]', decoded
        ):
            selected.add(token_id)
    return selected


def select_token_ids(
    counts: Mapping[int, int],
    *,
    target_size: int,
    original_vocab_size: int,
    required_token_ids: Iterable[int] = (),
) -> tuple[list[int], dict[str, Any]]:
    """Select a sorted map while preserving required ids and target size."""
    if target_size <= 0 or target_size % 128 != 0:
        raise ValueError("target_size must be a positive multiple of 128")
    if original_vocab_size <= 0:
        raise ValueError("original_vocab_size must be positive")
    if target_size > original_vocab_size:
        raise ValueError("target_size cannot exceed original_vocab_size")

    required = sorted(set(int(token_id) for token_id in required_token_ids))
    invalid_required = [
        token_id
        for token_id in required
        if token_id < 0 or token_id >= original_vocab_size
    ]
    if invalid_required:
        raise ValueError(f"required token ids outside vocabulary: {invalid_required}")
    if len(required) > target_size:
        raise ValueError("required token ids exceed target vocabulary size")

    observed = {
        int(token_id): int(frequency)
        for token_id, frequency in counts.items()
        if 0 <= int(token_id) < original_vocab_size and int(frequency) > 0
    }
    selected = set(required)
    ranked_observed = sorted(
        (token_id for token_id in observed if token_id not in selected),
        key=lambda token_id: (-observed[token_id], token_id),
    )
    selected.update(ranked_observed[: target_size - len(selected)])

    filler_count = 0
    if len(selected) < target_size:
        for token_id in range(original_vocab_size):
            if token_id in selected:
                continue
            selected.add(token_id)
            filler_count += 1
            if len(selected) == target_size:
                break

    token_ids = sorted(selected)
    if len(token_ids) != target_size:
        raise RuntimeError("failed to construct a map with the requested size")
    observed_selected_count = sum(token_id in observed for token_id in token_ids)
    observed_selection_fraction = observed_selected_count / target_size
    return token_ids, {
        "observed_token_count": len(observed),
        "observed_selected_count": observed_selected_count,
        "observed_selection_fraction": round(observed_selection_fraction, 6),
        "required_token_count": len(required),
        "filler_token_count": filler_count,
        "selection_policy": "frequency_desc_then_token_id_then_low_id_filler",
        "quality_warning": (
            "observed corpus covers less than 10% of the reduced vocabulary; "
            "do not use this map for a production engine"
            if observed_selection_fraction < 0.10
            else None
        ),
    }


def write_reduced_vocab_artifacts(
    output_dir: Path,
    token_ids: Sequence[int],
    *,
    original_vocab_size: int,
    selection_report: Mapping[str, Any],
) -> dict[str, Path]:
    """Write Edge-LLM map/metadata plus an auditable selection report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "vocab_map.safetensors"
    metadata_path = output_dir / "reduced_vocab.json"
    report_path = output_dir / "selection_report.json"

    payload = struct.pack(f"<{len(token_ids)}i", *(int(token_id) for token_id in token_ids))
    header = json.dumps(
        {
            "vocab_map": {
                "dtype": "I32",
                "shape": [len(token_ids)],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    map_path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    metadata_path.write_text(
        json.dumps(
            {
                "vocab_size": original_vocab_size,
                "reduced_vocab_size": len(token_ids),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "parksight_reduced_vocab_selection_v1",
        "map": {
            "path": str(map_path),
            "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
            "count": len(token_ids),
            "first_token_id": token_ids[0],
            "last_token_id": token_ids[-1],
        },
        "original_vocab_size": original_vocab_size,
        **dict(selection_report),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "map": map_path,
        "metadata": metadata_path,
        "report": report_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--input-jsonl", action="append", default=[], type=Path)
    parser.add_argument(
        "--study-report",
        action="append",
        default=[],
        type=Path,
        help="可选 StudyReport；将成功记录的 raw_output 纳入 token 频次",
    )
    parser.add_argument(
        "--field",
        action="append",
        default=None,
        help="Dotted JSON field containing output text; may be repeated",
    )
    parser.add_argument("--target-size", required=True, type=int)
    parser.add_argument("--original-vocab-size", type=int)
    parser.add_argument("--required-token-id", action="append", type=int, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--include-decoded-character-coverage",
        action="store_true",
        help="保留单 token 解码后包含中文或 JSON 结构字符的 tokenizer token",
    )
    args = parser.parse_args(argv)

    if not args.input_jsonl and not args.study_report:
        parser.error("at least one --input-jsonl or --study-report is required")
    for input_path in [*args.input_jsonl, *args.study_report]:
        if not input_path.is_file():
            parser.error(f"coverage input file not found: {input_path}")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to build a token map") from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
    )
    original_vocab_size = args.original_vocab_size or len(tokenizer)
    required = list(args.required_token_id)
    for attribute in ("eos_token_id", "bos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            required.append(int(token_id))
    decoded_coverage_ids: set[int] = set()
    if args.include_decoded_character_coverage:
        decoded_coverage_ids = collect_decoded_character_coverage_token_ids(tokenizer)
        required.extend(sorted(decoded_coverage_ids))
    fields = args.field or ["assessment"]
    counts = collect_token_counts(tokenizer, args.input_jsonl, fields)
    counts.update(collect_study_report_token_counts(tokenizer, args.study_report))
    token_ids, selection_report = select_token_ids(
        counts,
        target_size=args.target_size,
        original_vocab_size=original_vocab_size,
        required_token_ids=required,
    )
    paths = write_reduced_vocab_artifacts(
        args.output_dir,
        token_ids,
        original_vocab_size=original_vocab_size,
        selection_report={
            **selection_report,
            "input_jsonl": [str(path) for path in args.input_jsonl],
            "input_file_identities": [_file_identity(path) for path in args.input_jsonl],
            "study_reports": [str(path) for path in args.study_report],
            "study_report_identities": [
                _file_identity(path) for path in args.study_report
            ],
            "fields": list(fields),
            "required_token_ids": sorted(set(required)),
            "decoded_character_coverage": {
                "enabled": args.include_decoded_character_coverage,
                "token_count": len(decoded_coverage_ids),
                "selection_policy": "standalone_decode_contains_cjk_or_json_syntax",
            },
            "tokenizer": _tokenizer_identity(args.tokenizer),
        },
    )
    for path in paths.values():
        print(path)
    return 0


def _get_dotted_value(record: Any, field: str) -> Any:
    value = record
    for component in field.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)


def _tokenizer_identity(path: Path) -> dict[str, Any]:
    root = path.resolve()
    tokenizer_json = root / "tokenizer.json" if root.is_dir() else root
    identity: dict[str, Any] = {"path": str(root)}
    if tokenizer_json.is_file():
        identity.update(
            {
                "tokenizer_json": str(tokenizer_json),
                "tokenizer_json_size_bytes": tokenizer_json.stat().st_size,
                "tokenizer_json_sha256": hashlib.sha256(
                    tokenizer_json.read_bytes()
                ).hexdigest(),
            }
        )
    else:
        identity["tokenizer_json"] = None
    return identity


def _file_identity(path: Path) -> dict[str, Any]:
    """Return a stable identity for corpus files used during map selection."""
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
