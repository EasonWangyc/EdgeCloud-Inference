"""Summarize Jetson tegrastats samples inside a frozen StudyReport request window.

The output describes whole-device telemetry, not process RSS or GPU allocation.
Tegrastats timestamps have one-second precision; boundary seconds are included.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any

from parksight_vlm.tensorrt import sha256_file


_TIME_RE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}) ")
_RAM_RE = re.compile(r"\bRAM (\d+)/(\d+)MB\b")
_SWAP_RE = re.compile(r"\bSWAP (\d+)/(\d+)MB\b")
_GR3D_RE = re.compile(r"\bGR3D_FREQ (\d+)%")
_GPU_RE = re.compile(r"\bgpu@([\d.]+)C\b")
_TJ_RE = re.compile(r"\btj@([\d.]+)C\b")
_VDD_IN_RE = re.compile(r"\bVDD_IN (\d+)mW/(\d+)mW\b")


def _timezone_from_offset(value: str) -> timezone:
    if not re.fullmatch(r"[+-](?:0\d|1[0-4])\d{2}", value):
        raise ValueError("timezone offset must have format +HHMM or -HHMM")
    hours = int(value[1:3])
    minutes = int(value[3:5])
    if minutes > 59 or (hours == 14 and minutes != 0):
        raise ValueError("invalid timezone offset")
    delta = timedelta(hours=hours, minutes=minutes)
    return timezone(delta if value[0] == "+" else -delta)


def parse_sample(line: str, *, local_timezone: timezone) -> dict[str, Any]:
    stamp = _TIME_RE.search(line)
    if stamp is None:
        raise ValueError("tegrastats sample lacks a timestamp")
    fields = {
        "ram": _RAM_RE.search(line),
        "swap": _SWAP_RE.search(line),
        "gr3d": _GR3D_RE.search(line),
        "gpu": _GPU_RE.search(line),
        "tj": _TJ_RE.search(line),
        "vdd_in": _VDD_IN_RE.search(line),
    }
    missing = [name for name, match in fields.items() if match is None]
    if missing:
        raise ValueError(f"tegrastats sample lacks fields: {missing}")
    return {
        "sample_time_utc": datetime.strptime(stamp.group(1), "%m-%d-%Y %H:%M:%S")
        .replace(tzinfo=local_timezone).astimezone(timezone.utc),
        "system_ram_used_mb": int(fields["ram"].group(1)),
        "system_ram_total_mb": int(fields["ram"].group(2)),
        "system_swap_used_mb": int(fields["swap"].group(1)),
        "system_swap_total_mb": int(fields["swap"].group(2)),
        "gr3d_percent": int(fields["gr3d"].group(1)),
        "gpu_temp_c": float(fields["gpu"].group(1)),
        "junction_temp_c": float(fields["tj"].group(1)),
        "vdd_in_instant_w": int(fields["vdd_in"].group(1)) / 1000.0,
    }


def summarize(tegrastats: Path, report: Path, *, timezone_offset: str) -> dict[str, Any]:
    local_timezone = _timezone_from_offset(timezone_offset)
    study = json.loads(report.read_text(encoding="utf-8"))
    records = study.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("StudyReport has no records")
    starts = []
    ends = []
    for record in records:
        started = datetime.fromisoformat(record["started_at_utc"])
        duration_ms = record["stage_timings"]["end_to_end_ms"]
        if started.tzinfo is None or not isinstance(duration_ms, (int, float)) or not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError("StudyReport record lacks a valid UTC start or E2E duration")
        started = started.astimezone(timezone.utc)
        starts.append(started)
        ends.append(started + timedelta(milliseconds=duration_ms))
    start = min(starts)
    end = max(ends)
    lower = start.replace(microsecond=0)
    upper = end.replace(microsecond=0)
    samples = []
    for number, line in enumerate(tegrastats.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = parse_sample(line, local_timezone=local_timezone)
        except ValueError as error:
            raise ValueError(f"tegrastats line {number}: {error}") from error
        if lower <= sample["sample_time_utc"] <= upper:
            samples.append(sample)
    if not samples:
        raise ValueError("no tegrastats samples overlap the StudyReport request window")
    if len({sample["system_ram_total_mb"] for sample in samples}) != 1 or len({sample["system_swap_total_mb"] for sample in samples}) != 1:
        raise ValueError("tegrastats system memory totals changed during the request window")
    return {
        "schema_version": "parksight_jetson_tegrastats_study_window_v1",
        "study_id": study["study_identity"]["study_id"],
        "inputs": {
            "tegrastats": {"path": str(tegrastats), "sha256": sha256_file(tegrastats)},
            "study_report": {"path": str(report), "sha256": sha256_file(report)},
            "timezone_offset": timezone_offset,
        },
        "window_utc": {"first_request_start": start.isoformat(), "last_request_end": end.isoformat(), "sample_boundary_precision_seconds": 1},
        "sample_count": len(samples),
        "first_sample_utc": min(sample["sample_time_utc"] for sample in samples).isoformat(),
        "last_sample_utc": max(sample["sample_time_utc"] for sample in samples).isoformat(),
        "peak_system_ram_used_mb": max(sample["system_ram_used_mb"] for sample in samples),
        "system_ram_total_mb": samples[0]["system_ram_total_mb"],
        "peak_system_swap_used_mb": max(sample["system_swap_used_mb"] for sample in samples),
        "system_swap_total_mb": samples[0]["system_swap_total_mb"],
        "mean_vdd_in_instant_w": mean(sample["vdd_in_instant_w"] for sample in samples),
        "peak_vdd_in_instant_w": max(sample["vdd_in_instant_w"] for sample in samples),
        "peak_gpu_temp_c": max(sample["gpu_temp_c"] for sample in samples),
        "peak_junction_temp_c": max(sample["junction_temp_c"] for sample in samples),
        "peak_gr3d_percent": max(sample["gr3d_percent"] for sample in samples),
        "scope": "Whole-device tegrastats during request times, sampled at 500 ms with one-second timestamps; not process RSS, GPU allocator peak, or per-request power",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tegrastats", type=Path, required=True)
    parser.add_argument("--study-report", type=Path, required=True)
    parser.add_argument("--timezone-offset", required=True, help="Jetson local UTC offset, e.g. +0800")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.tegrastats, args.study_report, timezone_offset=args.timezone_offset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
