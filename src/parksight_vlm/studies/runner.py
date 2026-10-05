"""StudyRunner 从目录样本到单份报告的编排逻辑。"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, local
import time
from collections.abc import Callable, Mapping
from typing import Any

from parksight_vlm.casebook import ParkingCase, ParkingCaseCatalog
from parksight_vlm.inference import InferenceRecord, RiskRuntime

from .model import StudyDefinition, StudyReport, StudyValidationError
from .performance import compute_performance_metrics
from .quality import compute_quality_metrics


class StudyRunner:
    """运行冻结研究，且不依赖具体推理后端。"""

    def __init__(
        self,
        environment_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self._environment_provider = environment_provider or (lambda: {})

    def run(
        self,
        casebook: ParkingCaseCatalog,
        runtime: RiskRuntime,
        study: StudyDefinition,
        *,
        concurrency: int = 1,
        runtime_factory: Callable[[], RiskRuntime] | None = None,
    ) -> StudyReport:
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise StudyValidationError("concurrency must be a positive integer")
        if concurrency > 1 and runtime_factory is None:
            raise StudyValidationError("concurrent studies require an independent runtime per worker")
        selected_cases = casebook.cases_in_split(study.split)
        if not selected_cases:
            raise StudyValidationError(
                f"study split contains no cases: {study.split.value!r}"
            )

        unlabeled_case_ids = [
            parking_case.case_id
            for parking_case in selected_cases
            if parking_case.reference_assessment is None
        ]
        if unlabeled_case_ids:
            raise StudyValidationError(
                f"study cases require reference assessments: {unlabeled_case_ids}"
            )
        measurement_start = time.perf_counter()
        jobs = tuple(case for case in selected_cases for _ in range(study.repetitions))
        if concurrency == 1:
            records = tuple(runtime.analyze(case, study.workload) for case in jobs)
        else:
            worker_state = local()
            worker_runtimes: list[RiskRuntime] = []
            worker_lock = Lock()

            def analyze(case: ParkingCase) -> InferenceRecord:
                if not hasattr(worker_state, "runtime"):
                    assert runtime_factory is not None
                    worker_state.runtime = runtime_factory()
                    with worker_lock:
                        if worker_state.runtime is runtime or any(worker is worker_state.runtime for worker in worker_runtimes):
                            raise StudyValidationError("workers must not share a runtime instance")
                        worker_runtimes.append(worker_state.runtime)
                    if worker_state.runtime.identity != runtime.identity:
                        raise StudyValidationError("worker runtime identity differs from study runtime")
                return worker_state.runtime.analyze(case, study.workload)

            try:
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    # map 保持提交顺序；等待中的任务不计入单请求 TTFT。
                    records = tuple(executor.map(analyze, jobs))
            finally:
                for worker in worker_runtimes:
                    worker.close()
        measurement_wall_seconds = time.perf_counter() - measurement_start
        references = {
            parking_case.case_id: parking_case.reference_assessment
            for parking_case in selected_cases
            if parking_case.reference_assessment is not None
        }
        failure_summary = Counter(
            record.failure.category.value
            for record in records
            if record.failure is not None
        )
        return StudyReport(
            study_identity={
                **study.identity_mapping(runtime.identity),
                "client_concurrency": concurrency,
                "load_pattern": "closed_loop_workers",
            },
            environment_snapshot=dict(self._environment_provider()),
            quality_metrics=compute_quality_metrics(records, references),
            performance_metrics=compute_performance_metrics(records, measurement_wall_seconds=measurement_wall_seconds),
            failure_summary=dict(sorted(failure_summary.items())),
            records=records,
        )
