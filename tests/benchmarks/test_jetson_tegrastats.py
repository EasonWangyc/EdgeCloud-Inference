import json
from pathlib import Path
import tempfile
import unittest

from scripts.diagnostics.summarize_jetson_tegrastats import _timezone_from_offset, parse_sample, summarize


def sample(stamp: str, *, ram: int, power: int) -> str:
    return (
        f"{stamp} RAM {ram}/7619MB (lfb 1x4MB) SWAP 1652/3810MB (cached 1MB) "
        f"CPU [5%@729] GR3D_FREQ 95% gpu@51.5C tj@52.125C "
        f"VDD_IN {power}mW/10943mW"
    )


class TegrastatsSummaryTests(unittest.TestCase):
    def test_parser_converts_jetson_local_time_and_instant_power(self):
        row = parse_sample(
            sample("10-12-2026 23:20:59", ram=5136, power=12617),
            local_timezone=_timezone_from_offset("+0800"),
        )
        self.assertEqual(row["sample_time_utc"].isoformat(), "2026-10-12T15:20:59+00:00")
        self.assertEqual(row["system_ram_used_mb"], 5136)
        self.assertEqual(row["vdd_in_instant_w"], 12.617)
        self.assertEqual(row["junction_temp_c"], 52.125)

    def test_summary_uses_request_window_and_excludes_startup_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "tegrastats.log"
            report = root / "study.json"
            log.write_text("\n".join([
                sample("10-12-2026 23:20:58", ram=2000, power=7000),
                sample("10-12-2026 23:20:59", ram=5000, power=10000),
                sample("10-12-2026 23:21:00", ram=5100, power=12000),
                sample("10-12-2026 23:21:01", ram=1000, power=5000),
            ]) + "\n")
            report.write_text(json.dumps({
                "study_identity": {"study_id": "pilot"},
                "records": [{
                    "started_at_utc": "2026-10-12T15:20:59.250000+00:00",
                    "stage_timings": {"end_to_end_ms": 1000.0},
                }],
            }))
            result = summarize(log, report, timezone_offset="+0800")
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["peak_system_ram_used_mb"], 5100)
            self.assertEqual(result["mean_vdd_in_instant_w"], 11.0)
            self.assertEqual(result["first_sample_utc"], "2026-10-12T15:20:59+00:00")

    def test_invalid_offset_and_missing_power_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone offset"):
            _timezone_from_offset("+1460")
        with self.assertRaisesRegex(ValueError, "vdd_in"):
            parse_sample("10-12-2026 23:20:59 RAM 5000/7619MB SWAP 1/3810MB GR3D_FREQ 95% gpu@51C tj@52C", local_timezone=_timezone_from_offset("+0800"))


if __name__ == "__main__":
    unittest.main()
