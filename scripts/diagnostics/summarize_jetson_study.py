"""从板端 StudyReport 与 tegrastats 日志生成运行证据摘要。"""

from __future__ import annotations

import argparse
from pathlib import Path

from parksight_vlm.studies.jetson_evidence import write_jetson_study_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-report", required=True, type=Path)
    parser.add_argument("--tegrastats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    write_jetson_study_summary(
        study_report_path=args.study_report,
        tegrastats_path=args.tegrastats,
        output_path=args.output,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
