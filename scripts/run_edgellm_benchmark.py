"""在 Jetson 上对冻结 manifest 运行 Edge-LLM 低层 HTTP benchmark。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from parksight_vlm.casebook import DatasetSplit, ParkingCase
from parksight_vlm.inference.edge_llm import EdgeLlmHttpBackend
from parksight_vlm.workload import FrozenWorkload


def build_http_backend(
    *,
    endpoint: str,
    model_name: str,
    timeout_seconds: float,
    stream_responses: bool,
    reuse_http_connection: bool,
) -> EdgeLlmHttpBackend:
    """按板端 backend 实际接口构造客户端，兼容旧版 Jetson checkout。"""
    parameters = inspect.signature(EdgeLlmHttpBackend).parameters
    kwargs: dict[str, object] = {
        "base_url": endpoint,
        "model_name": model_name,
        "timeout_seconds": timeout_seconds,
    }
    optional_kwargs = {
        "stream_responses": stream_responses,
        "reuse_http_connection": reuse_http_connection,
    }
    kwargs.update(
        {
            name: value
            for name, value in optional_kwargs.items()
            if name in parameters
        }
    )
    return EdgeLlmHttpBackend(**kwargs)


def load_benchmark_cases(
    manifest_path: Path,
    *,
    split: str,
    data_root: Path,
    limit: int | None,
) -> list[ParkingCase]:
    """从冻结 manifest 读取样本，不要求加载标注，避免把质量评测混入性能测试。"""
    try:
        selected_split = DatasetSplit(split)
    except ValueError as error:
        raise ValueError(f"unsupported split: {split!r}") from error

    cases: list[ParkingCase] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"manifest line {line_number} must be an object")
        required = {"case_id", "image_ref", "source_group_id", "split"}
        if set(payload) != required:
            raise ValueError(
                f"manifest line {line_number} fields must be {sorted(required)}"
            )
        if payload["split"] != selected_split.value:
            continue
        image_ref = payload["image_ref"]
        if not isinstance(image_ref, str) or not image_ref.strip():
            raise ValueError(f"manifest line {line_number} image_ref is invalid")
        pure_ref = PurePosixPath(image_ref)
        if pure_ref.is_absolute() or ".." in pure_ref.parts:
            raise ValueError(f"manifest line {line_number} image_ref escapes data root")
        case = ParkingCase(
            case_id=str(payload["case_id"]),
            image_ref=pure_ref,
            source_group_id=str(payload["source_group_id"]),
            split=selected_split,
            reference_assessment=None,
        )
        case.resolve_image(data_root, require_exists=True)
        cases.append(case)
        if limit is not None and len(cases) >= limit:
            break
    if not cases:
        raise ValueError(f"no manifest records found for split {split!r}")
    return cases


def run_sample(
    *,
    backend: EdgeLlmHttpBackend,
    case: ParkingCase,
    workload: FrozenWorkload,
    data_root: Path,
    repetition: int,
) -> dict[str, Any]:
    """执行一次请求，保留客户端 E2E、HTTP RTT 和服务端阶段字段的区别。"""
    started_clock = time.perf_counter()
    try:
        image_path = case.resolve_image(data_root, require_exists=True)
        generation = backend.generate(image_path=image_path, workload=workload)
        timings = {
            name: float(value)
            for name, value in generation.stage_timings.to_mapping().items()
            if value is not None
        }
        # end_to_end_ms 是客户端从样本进入请求到响应结束的完整链路；
        # http_round_trip_ms 单独保留，不能写入 decode_ms。
        timings["end_to_end_ms"] = (time.perf_counter() - started_clock) * 1000.0
        return {
            "sample_id": case.case_id,
            "repetition": repetition,
            "status": "completed",
            "output_tokens": generation.output_tokens,
            "timings_ms": timings,
        }
    except Exception as error:  # benchmark 需保留失败样本，不因单样本退出
        return {
            "sample_id": case.case_id,
            "repetition": repetition,
            "status": "failed",
            "output_tokens": None,
            "timings_ms": {
                "end_to_end_ms": (time.perf_counter() - started_clock) * 1000.0
            },
            "failure": {
                "category": _failure_category(error),
                "message": str(error) or error.__class__.__name__,
                "exception_type": error.__class__.__name__,
            },
        }


def benchmark(
    *,
    cases: list[ParkingCase],
    workload: FrozenWorkload,
    backend: EdgeLlmHttpBackend,
    data_root: Path,
    repetitions: int,
    warmup: int,
    output_jsonl: Path,
    warmup_output_jsonl: Path | None,
    concurrency: int = 1,
    backend_factory: Callable[[], EdgeLlmHttpBackend] | None = None,
) -> dict[str, Any]:
    """先 warm-up，再按固定顺序输出样本；可选并发压测服务端动态 batch。"""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if concurrency > 1 and backend_factory is None:
        raise ValueError("backend_factory is required when concurrency is greater than one")
    if warmup > 0 and warmup_output_jsonl is None:
        raise ValueError(
            "warmup_output_jsonl is required when warmup is greater than zero"
        )
    warmup_rows: list[dict[str, Any]] = []
    for warmup_index in range(1, warmup + 1):
        row = run_sample(
            backend=backend,
            case=cases[(warmup_index - 1) % len(cases)],
            workload=workload,
            data_root=data_root,
            repetition=warmup_index,
        )
        row["sample_id"] = f"warmup-{warmup_index}-{row['sample_id']}"
        warmup_rows.append(row)
        if row["status"] != "completed":
            # Preserve the failure as evidence even though the steady-state run
            # must not start after an unsuccessful warm-up.
            _write_jsonl(warmup_output_jsonl, warmup_rows)
            raise RuntimeError(f"warm-up failed: {row['failure']}")

    if warmup_output_jsonl is not None:
        _write_jsonl(warmup_output_jsonl, warmup_rows)

    steady_state_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if concurrency == 1:
        for repetition in range(1, repetitions + 1):
            for case in cases:
                rows.append(
                    run_sample(
                        backend=backend,
                        case=case,
                        workload=workload,
                        data_root=data_root,
                        repetition=repetition,
                    )
                )
    else:
        rows = _run_concurrent_samples(
            cases=cases,
            workload=workload,
            data_root=data_root,
            repetitions=repetitions,
            concurrency=concurrency,
            backend_factory=backend_factory,
        )

    _write_jsonl(output_jsonl, rows)
    steady_state_wall_clock_ms = (time.perf_counter() - steady_state_started) * 1000.0
    completed_rows = [row for row in rows if row["status"] == "completed"]
    output_tokens = sum(
        int(row["output_tokens"])
        for row in completed_rows
        if isinstance(row.get("output_tokens"), int)
    )
    wall_clock_seconds = steady_state_wall_clock_ms / 1000.0
    return {
        "steady_state_wall_clock_ms": steady_state_wall_clock_ms,
        "steady_state_requests": len(rows),
        "steady_state_completed_requests": len(completed_rows),
        "steady_state_output_tokens": output_tokens,
        "aggregate_output_tokens_per_second": (
            output_tokens / wall_clock_seconds if wall_clock_seconds > 0 else None
        ),
    }


def _run_concurrent_samples(
    *,
    cases: list[ParkingCase],
    workload: FrozenWorkload,
    data_root: Path,
    repetitions: int,
    concurrency: int,
    backend_factory: Callable[[], EdgeLlmHttpBackend] | None,
) -> list[dict[str, Any]]:
    """以固定顺序收集并发请求，每个线程懒创建一个独立 backend。"""
    if backend_factory is None:
        raise ValueError("backend_factory is required for concurrent samples")

    thread_state = threading.local()
    created_backends: list[EdgeLlmHttpBackend] = []
    created_backends_lock = threading.Lock()

    def worker(order: int, case: ParkingCase, repetition: int) -> tuple[int, dict[str, Any]]:
        backend = getattr(thread_state, "backend", None)
        if backend is None:
            backend = backend_factory()
            thread_state.backend = backend
            with created_backends_lock:
                created_backends.append(backend)
        return order, run_sample(
            backend=backend,
            case=case,
            workload=workload,
            data_root=data_root,
            repetition=repetition,
        )

    ordered_rows: dict[int, dict[str, Any]] = {}
    items = [
        (repetition, case)
        for repetition in range(1, repetitions + 1)
        for case in cases
    ]
    try:
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="edgellm-benchmark",
        ) as executor:
            futures = {
                executor.submit(worker, order, case, repetition): order
                for order, (repetition, case) in enumerate(items)
            }
            for future in as_completed(futures):
                order = futures[future]
                try:
                    completed_order, row = future.result()
                except Exception as error:
                    repetition, case = _concurrent_item_at(
                        order=order,
                        cases=cases,
                        repetitions=repetitions,
                    )
                    row = _failed_row(case=case, repetition=repetition, error=error)
                    completed_order = order
                if completed_order in ordered_rows:
                    raise RuntimeError(f"duplicate benchmark order: {completed_order}")
                ordered_rows[completed_order] = row
    finally:
        for backend in created_backends:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
    return [ordered_rows[index] for index in range(len(ordered_rows))]


def _concurrent_item_at(
    *, order: int, cases: list[ParkingCase], repetitions: int
) -> tuple[int, ParkingCase]:
    """根据提交序号恢复失败 future 对应的 repetition 和 case。"""
    if order < 0 or order >= len(cases) * repetitions:
        raise RuntimeError(f"invalid benchmark order: {order}")
    return order // len(cases) + 1, cases[order % len(cases)]


def _failed_row(*, case: ParkingCase, repetition: int, error: Exception) -> dict[str, Any]:
    return {
        "sample_id": case.case_id,
        "repetition": repetition,
        "status": "failed",
        "output_tokens": None,
        "timings_ms": {"end_to_end_ms": 0.0},
        "failure": {
            "category": _failure_category(error),
            "message": str(error) or error.__class__.__name__,
            "exception_type": error.__class__.__name__,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--warmup-output-jsonl", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发请求数；>1 用于验证服务端动态 batch，默认串行",
    )
    parser.add_argument(
        "--run-metadata-json",
        type=Path,
        help="可选运行元数据输出，记录并发度和请求链路开关",
    )
    parser.add_argument(
        "--engine-max-batch-size",
        type=int,
        help="可选 engine maxBatchSize；仅写入 provenance 元数据，不改变客户端并发度",
    )
    parser.add_argument(
        "--engine-max-kv-pool-pages",
        type=int,
        help="可选 engine maxKVPoolPages；仅写入 provenance 元数据，不改变客户端请求",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model-name", default="local")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--no-stream-responses",
        action="store_true",
        help="关闭 SSE；关闭后只有服务端显式返回 TTFT 时才有 TTFT",
    )
    parser.add_argument(
        "--reuse-http-connection",
        action="store_true",
        help="复用 HTTP/1.1 keep-alive 连接；只作为服务链路 A/B 变量",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if (
        args.repetitions <= 0
        or args.warmup < 0
        or args.concurrency <= 0
        or (args.engine_max_batch_size is not None and args.engine_max_batch_size <= 0)
        or (args.engine_max_kv_pool_pages is not None and args.engine_max_kv_pool_pages <= 0)
    ):
        parser.error(
            "--repetitions must be positive, --warmup non-negative, and --concurrency positive"
        )
    if args.warmup > 0 and args.warmup_output_jsonl is None:
        parser.error("--warmup-output-jsonl is required when --warmup is greater than zero")

    cases = load_benchmark_cases(
        args.manifest,
        split=args.split,
        data_root=args.data_root,
        limit=args.limit,
    )
    workload = FrozenWorkload.load(args.workload)
    backend = build_http_backend(
        endpoint=args.endpoint,
        model_name=args.model_name,
        timeout_seconds=args.timeout_seconds,
        stream_responses=not args.no_stream_responses,
        reuse_http_connection=args.reuse_http_connection,
    )

    def create_backend() -> EdgeLlmHttpBackend:
        return build_http_backend(
            endpoint=args.endpoint,
            model_name=args.model_name,
            timeout_seconds=args.timeout_seconds,
            stream_responses=not args.no_stream_responses,
            reuse_http_connection=args.reuse_http_connection,
        )

    run_stats = benchmark(
        cases=cases,
        workload=workload,
        backend=backend,
        data_root=args.data_root.resolve(),
        repetitions=args.repetitions,
        warmup=args.warmup,
        output_jsonl=args.output_jsonl,
        warmup_output_jsonl=args.warmup_output_jsonl,
        concurrency=args.concurrency,
        backend_factory=create_backend,
    )
    if args.run_metadata_json is not None:
        _write_json(
            args.run_metadata_json,
            {
                "schema_version": "parksight_tensorrt_benchmark_run_v1",
                "cases": len(cases),
                "repetitions": args.repetitions,
                "warmup": args.warmup,
                "concurrency": args.concurrency,
                "max_batch_size": args.engine_max_batch_size,
                "max_kv_pool_pages": args.engine_max_kv_pool_pages,
                "stream_responses": not args.no_stream_responses,
                "reuse_http_connection": args.reuse_http_connection,
                **run_stats,
            },
        )
    print(args.output_jsonl)
    print(
        f"cases={len(cases)} repetitions={args.repetitions} warmup={args.warmup} "
        f"concurrency={args.concurrency}"
    )
    return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _failure_category(error: Exception) -> str:
    lowered = str(error).lower()
    if isinstance(error, TimeoutError) or "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "out of memory" in lowered:
        return "out_of_memory"
    if isinstance(error, json.JSONDecodeError):
        return "json_parse_error"
    return "runtime_error"


if __name__ == "__main__":
    raise SystemExit(main())
