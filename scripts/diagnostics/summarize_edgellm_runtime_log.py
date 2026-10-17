"""汇总 Edge-LLM runtime 日志中的 attention/kernel 与 CUDA Graph 证据。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_FMHA_RE = re.compile(
    r"(?:Loading FMHA cubin|Skipping FMHA cubin[^:]*): index=(?P<index>\d+) "
    r"sm=(?P<sm>\d+) size=(?P<size>\d+) function=(?P<function>\S+)"
)
_FMHA_FORCED_TILED_RE = re.compile(
    r"forced tiled head_dim=128 candidate",
    re.IGNORECASE,
)
_CONFIG_RE = re.compile(r"LLMEngineConfig\{(?P<body>[^}]*)\}")
_AUX_STREAM_RE = re.compile(r"Number of aux streams is (?P<count>\d+)")
_WORKER_STREAM_RE = re.compile(r"Number of total worker streams is (?P<count>\d+)")
_PAGED_KV_RE = re.compile(r"usePagedKVCache\s*[=:]\s*(?P<value>true|false)", re.IGNORECASE)
_PROFILE_SWITCH_RE = re.compile(
    r"Switching optimization profile from: (?P<from>\d+) to (?P<to>\d+)"
)
_PROFILE_SWITCH_CALL_RE = re.compile(
    r"(?P<kind>profile switch call|pinned profile initialization call) profile=(?P<profile>\d+) "
    r"elapsed_ms=(?P<elapsed_ms>\d+(?:\.\d+)?) success=(?P<success>true|false)"
)
_GRAPH_REPLAY_RE = re.compile(
    r"(?:cudaGraphLaunch|CUDA graph (?:replay|replaying|launch|launched)|"
    r"(?:replay|replaying|launch|launched) (?:the )?CUDA graph)",
    re.IGNORECASE,
)
_LOG_TIMESTAMP_RE = re.compile(
    r"^\[(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}(?:\.\d+)?)\]"
)


def summarize_log(path: Path | str) -> dict[str, Any]:
    """从单份日志提取可审计的 runtime 事实，不推断实际 tactic 选择。"""
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    load_attempts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    aux_streams: list[int] = []
    worker_streams: list[int] = []
    profile_switches: list[dict[str, int]] = []
    profile_switch_times_ms: list[float | None] = []
    profile_switch_calls: list[dict[str, Any]] = []
    graph_replay_count = 0
    forced_tiled_head128_count = 0

    for line in text.splitlines():
        if _GRAPH_REPLAY_RE.search(line) is not None:
            graph_replay_count += 1
        fmha_match = _FMHA_RE.search(line)
        if fmha_match is not None:
            entry = {
                "index": int(fmha_match.group("index")),
                "sm": int(fmha_match.group("sm")),
                "size_bytes": int(fmha_match.group("size")),
                "function": fmha_match.group("function"),
            }
            load_attempts.append(entry)
            if "Skipping FMHA cubin rejected" in line:
                rejected.append(entry)
        if _FMHA_FORCED_TILED_RE.search(line) is not None:
            forced_tiled_head128_count += 1

        config_match = _CONFIG_RE.search(line)
        if config_match is not None:
            configs.append(_parse_config(config_match.group("body")))

        aux_match = _AUX_STREAM_RE.search(line)
        if aux_match is not None:
            aux_streams.append(int(aux_match.group("count")))
        worker_match = _WORKER_STREAM_RE.search(line)
        if worker_match is not None:
            worker_streams.append(int(worker_match.group("count")))
        profile_switch_match = _PROFILE_SWITCH_RE.search(line)
        if profile_switch_match is not None:
            profile_switches.append(
                {
                    "from": int(profile_switch_match.group("from")),
                    "to": int(profile_switch_match.group("to")),
                }
            )
            profile_switch_times_ms.append(_parse_log_timestamp_ms(line))
        profile_switch_call_match = _PROFILE_SWITCH_CALL_RE.search(line)
        if profile_switch_call_match is not None:
            profile_switch_calls.append(
                {
                    "kind": profile_switch_call_match.group("kind"),
                    "profile": int(profile_switch_call_match.group("profile")),
                    "elapsed_ms": float(profile_switch_call_match.group("elapsed_ms")),
                    "success": profile_switch_call_match.group("success") == "true",
                }
            )

    profile_transition_intervals_ms = _elapsed_intervals_ms(profile_switch_times_ms)

    graph_disabled = (
        "CUDA graph capture disabled by EDGELLM_DISABLE_CUDA_GRAPH=1." in text
    )
    graph_requested = "CUDA graph enabled" in text and not graph_disabled
    graph_capture_success = "CUDA graph captured successfully" in text
    decoder_graph_capture_success = (
        "Successfully captured decoding CUDA graphs" in text
    )
    rejected_keys = {_fmha_key(item) for item in rejected}
    loaded = [
        item for item in load_attempts if _fmha_key(item) not in rejected_keys
    ]
    engine_config = configs[-1] if configs else None
    attention = _summarize_attention_capabilities(engine_config, text)
    return {
        "schema_version": "parksight_edgellm_runtime_log_v2",
        "log_path": str(log_path),
        "line_count": len(text.splitlines()),
        "engine_config": engine_config,
        "attention": attention,
        "fmha": {
            "load_attempts": load_attempts,
            "loaded": loaded,
            "rejected": rejected,
            "loaded_count": len(loaded),
            "rejected_count": len(rejected),
            "forced_tiled_head128_count": forced_tiled_head128_count,
            "forced_tiled_head128_observed": forced_tiled_head128_count > 0,
            "evidence_boundary": (
                "loaded means the log did not subsequently reject the cubin; the forced-tiled "
                "marker confirms the candidate selection branch, while kernel timing still "
                "requires Nsight Systems/Compute"
            ),
        },
        "cuda_graph": {
            "requested": graph_requested,
            "disabled_by_config": graph_disabled,
            "capture_success": graph_capture_success and not graph_disabled,
            "decoder_capture_success": (
                decoder_graph_capture_success and not graph_disabled
            ),
            "replay_count": graph_replay_count,
            "replay_observed": graph_replay_count > 0 and not graph_disabled,
            "replay_evidence_boundary": (
                "replay_observed requires an explicit CUDA graph replay/launch marker in the log; "
                "it does not establish replay timing or rule out fallback execution, which requires Nsight Systems"
            ),
        },
        "streams": {
            "aux_stream_counts": aux_streams,
            "worker_stream_counts": worker_streams,
            "profile_switch_count": len(profile_switches),
            "profile_transitions": profile_switches,
            "same_profile_transition_count": sum(
                1 for item in profile_switches if item["from"] == item["to"]
            ),
            "timed_profile_transition_count": sum(
                timestamp is not None for timestamp in profile_switch_times_ms
            ),
            "profile_transition_message_intervals_ms": profile_transition_intervals_ms,
            "profile_transition_message_interval_summary": _numeric_summary(
                profile_transition_intervals_ms
            ),
            "profile_transition_interval_evidence_boundary": (
                "the interval between consecutive profile-switch log messages; it is not "
                "the duration of setOptimizationProfileAsync and includes work between messages"
            ),
            "profile_switch_api_calls": profile_switch_calls,
            "profile_switch_api_call_summary": _numeric_summary(
                [item["elapsed_ms"] for item in profile_switch_calls if item["success"]]
            ),
            "profile_switch_api_failure_count": sum(
                1 for item in profile_switch_calls if not item["success"]
            ),
            "profile_switch_api_evidence_boundary": (
                "elapsed_ms is the host-side API call interval emitted by the runtime; it is "
                "not GPU completion time and does not include later enqueue or synchronization"
            ),
        },
    }


def compare_summaries(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """比较两份日志的 attention 候选和 graph 事实。"""
    left_fmha = left["fmha"]
    right_fmha = right["fmha"]
    left_loaded = sorted(_fmha_key(item) for item in left_fmha["loaded"])
    right_loaded = sorted(_fmha_key(item) for item in right_fmha["loaded"])
    left_rejected = sorted(_fmha_key(item) for item in left_fmha["rejected"])
    right_rejected = sorted(_fmha_key(item) for item in right_fmha["rejected"])
    return {
        "same_engine_config": left.get("engine_config") == right.get("engine_config"),
        "same_attention_capabilities": left.get("attention") == right.get("attention"),
        "same_loaded_fmha_candidates": left_loaded == right_loaded,
        "same_rejected_fmha_candidates": left_rejected == right_rejected,
        "left_graph": left["cuda_graph"],
        "right_graph": right["cuda_graph"],
        "left_profile_switch_api": left.get("streams", {}).get(
            "profile_switch_api_call_summary"
        ),
        "right_profile_switch_api": right.get("streams", {}).get(
            "profile_switch_api_call_summary"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        type=Path,
        help="Edge-LLM runtime log; may be passed twice for an A/B comparison",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    summaries = [summarize_log(path) for path in args.log]
    result: dict[str, Any] = {
        "schema_version": "parksight_edgellm_runtime_log_comparison_v2",
        "summaries": summaries,
    }
    if len(summaries) == 2:
        result["comparison"] = compare_summaries(summaries[0], summaries[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


def _parse_config(body: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, raw_value in re.findall(r"(?P<name>\w+)=(?P<value>[^ ]+)", body):
        if raw_value.isdigit():
            values[name] = int(raw_value)
        elif raw_value in {"true", "false"}:
            values[name] = raw_value == "true"
        else:
            values[name] = raw_value
    return values


def _summarize_attention_capabilities(
    engine_config: dict[str, Any] | None, text: str
) -> dict[str, Any]:
    """Expose KV/attention routing facts without claiming kernel execution."""
    config = engine_config or {}
    paged_value = config.get("usePagedKVCache")
    if not isinstance(paged_value, bool):
        paged_match = _PAGED_KV_RE.search(text)
        paged_value = (
            paged_match.group("value").lower() == "true"
            if paged_match is not None
            else None
        )
    return {
        "num_kv_heads": config.get("numKVHeads"),
        "head_dim": config.get("headDim"),
        "kv_cache_dtype": config.get("kvCacheDtype"),
        "use_paged_kv_cache": paged_value,
        "max_kv_pool_pages": config.get("maxKVPoolPages", config.get("max_kv_pool_pages")),
        "tokens_per_page": config.get("tokensPerPage", config.get("tokens_per_page")),
        "spec_decode_type": config.get("specDecodeType"),
        "evidence_boundary": (
            "these are runtime/configuration facts; they do not prove that a specific "
            "attention kernel, tactic, paged layout, or speculative path executed"
        ),
    }


def _fmha_key(item: dict[str, Any]) -> tuple[int, int, str]:
    return item["index"], item["sm"], item["function"]


def _parse_log_timestamp_ms(line: str) -> float | None:
    match = _LOG_TIMESTAMP_RE.match(line)
    if match is None:
        return None
    hours = int(match.group("hour"))
    minutes = int(match.group("minute"))
    seconds = float(match.group("second"))
    return (hours * 3600 + minutes * 60 + seconds) * 1000.0


def _elapsed_intervals_ms(timestamps_ms: list[float | None]) -> list[float]:
    """Return intervals only for adjacent transitions with parseable timestamps."""
    intervals: list[float] = []
    for previous, current in zip(timestamps_ms, timestamps_ms[1:]):
        if previous is None or current is None:
            continue
        interval = current - previous
        if interval < 0:
            interval += 24 * 60 * 60 * 1000
        intervals.append(round(interval, 3))
    return intervals


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * weight, 3)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


if __name__ == "__main__":
    raise SystemExit(main())
