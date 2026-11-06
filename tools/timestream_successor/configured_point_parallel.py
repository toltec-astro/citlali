"""Reduction-owned array workers; no scientific calculations or global pool.

The native observation setup owns the effective CPU budget. Fresh spawn workers
inherit single-thread numerical-library limits before importing NumPy/SciPy.
Independent array jobs keep separate output paths; all jobs join before callers
advance a feedback cycle or publish a completed product set.
"""

from concurrent.futures import ProcessPoolExecutor, wait
from contextlib import contextmanager
import multiprocessing
import os
import resource
import sys
import time


THREAD_LIMITS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     VECLIB_MAXIMUM_THREADS="1", BLIS_NUM_THREADS="1", OMP_DYNAMIC="FALSE")


@contextmanager
def _spawn_environment():
    # The orchestrator is single-threaded. Limits are inherited during submit's
    # synchronous process spawn, before __main__ or scientific module imports.
    previous = {key: os.environ.get(key) for key in THREAD_LIMITS}
    os.environ.update(THREAD_LIMITS)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _array_job(kind, job):
    wall, cpu = time.monotonic(), time.process_time()
    if kind == "feedback":
        from configured_point_feedback import initialize, update

        path, destination, verbose, context = job
        if context is None:
            context, result = initialize(path, destination, verbose)
        else:
            _freeze_context(context)
            result = update(path, context, destination, verbose)
        value = context, result
    elif kind == "fit":
        from point_reference import measure_forward

        path, details = job
        value = measure_forward(path, details=details)
    else:
        raise ValueError("unknown POINT array task")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value, dict(pid=os.getpid(), wall_seconds=time.monotonic() - wall,
                      cpu_seconds=time.process_time() - cpu,
                      process_lifetime_peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024)


def _freeze_context(context):
    # Pickle transports independent arrays but does not retain their write flag.
    if context is not None:
        for name in ("scale", "psd", "base_scale", "noise_valid"):
            getattr(context, name).flags.writeable = False


class PointArrayExecutor:
    """At most three array processes, reused only within this reduction.

    Idle workers do no computation while a native stage owns the CPU budget.
    Even a one-worker run uses spawn, so inherited parent BLAS state cannot
    change the numerical-library thread limit or serial/parallel comparison.
    """

    def __init__(self, requested, native_execution, *, maximum=12):
        effective = native_execution.get("effective_workers")
        if (type(effective) is not int or not 1 <= effective <= requested <= maximum <= 32
                or native_execution.get("requested_workers") != requested):
            raise ValueError("native POINT execution budget is missing or inconsistent")
        self.budget = dict(native_execution)
        self.workers = min(effective, 3)
        self.reports = []
        self._pool = None

    def run(self, kind, jobs, *, phase):
        jobs = list(jobs)
        if not 1 <= len(jobs) <= 3:
            raise ValueError("POINT array phase requires one to three jobs")
        started = time.monotonic()
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers,
                                            mp_context=multiprocessing.get_context("spawn"))
        futures = []
        try:
            with _spawn_environment():
                for job in jobs:
                    futures.append(self._pool.submit(_array_job, kind, job))
        finally:
            # Also join submitted work if pool startup/submission fails.
            wait(futures)
        report = dict(phase=phase, jobs=len(jobs), worker_budget=self.workers,
                      wall_seconds=time.monotonic() - started, status="failed",
                      numerical_library_threads_requested=1,
                      process_start_method="spawn")
        self.reports.append(report)
        # Input order determines both output order and the first raised failure.
        # There is no partially successful return from a failed array phase.
        results = [future.result() for future in futures]
        if kind == "feedback":
            for (context, _), _metrics in results:
                _freeze_context(context)
        report.update(status="completed", jobs_detail=[item[1] for item in results],
                      workers_used=len({item[1]["pid"] for item in results}),
                      cpu_seconds=sum(item[1]["cpu_seconds"] for item in results))
        return [item[0] for item in results]

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
