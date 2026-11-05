"""Owner-run concurrency, failure and scientific equivalence regressions."""

import os
from pathlib import Path
import time

import numpy as np
import pytest
from astropy.io import fits

import configured_point_parallel as parallel
import configured_point_products as products
from test_configured_point_products import native_map


def executor(workers):
    return parallel.PointArrayExecutor(workers, dict(requested_workers=workers,
                                                    effective_workers=workers))


def synchronized_child(kind, job):
    """Importable spawn callback for scheduler tests, never installed runtime."""
    index, root, fail = job
    root = Path(root)
    (root / f"arrived-{index}").touch()
    deadline = time.monotonic() + 10
    while len(list(root.glob("arrived-*"))) != 3:
        if time.monotonic() >= deadline:
            raise RuntimeError("array jobs did not overlap")
        time.sleep(.01)
    limits = {key: os.environ.get(key) for key in parallel.THREAD_LIMITS}
    (root / f"finished-{index}").touch()
    if fail and index in (0, 2):
        raise RuntimeError(f"array-{index}")
    return (index, limits), dict(pid=os.getpid(), cpu_seconds=0.)


def test_spawn_budget_overlap_order_limits_and_parent_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "19")
    monkeypatch.setattr(parallel, "_array_job", synchronized_child)
    pool = executor(3)
    try:
        result = pool.run("test", [(i, tmp_path, False) for i in range(3)], phase="test")
        assert [item[0] for item in result] == [0, 1, 2]
        assert all(item[1] == parallel.THREAD_LIMITS for item in result)
        assert pool.reports[0]['workers_used'] == 3
        assert os.environ['OPENBLAS_NUM_THREADS'] == '19'
    finally:
        pool.close()


def test_worker_failure_joins_every_array_and_propagates_first_input(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "_array_job", synchronized_child)
    pool = executor(3)
    try:
        with pytest.raises(RuntimeError, match="array-0"):
            pool.run("test", [(i, tmp_path, True) for i in range(3)], phase="test-failure")
        assert len(list(tmp_path.glob('finished-*'))) == 3
        assert pool.reports[0]['status'] == 'failed'
    finally:
        pool.close()


@pytest.mark.parametrize('budget', [{}, {'requested_workers': 3, 'effective_workers': 0},
                                   {'requested_workers': 3, 'effective_workers': 4},
                                   {'requested_workers': 2, 'effective_workers': 2}])
def test_native_cpu_budget_is_required_and_cannot_be_expanded(budget):
    with pytest.raises(ValueError, match="budget"):
        parallel.PointArrayExecutor(3, budget)


def test_array_count_caps_large_cpu_budget():
    pool = executor(12)
    assert pool.workers == 3 and pool._pool is None
    with pytest.raises(ValueError, match="one to three"):
        pool.run('fit', [], phase='invalid')
    pool.close()


def test_spawned_feedback_and_final_fits_preserve_serial_results(tmp_path):
    paths = [native_map(tmp_path / 'native' / array, array) for array in products.ARRAYS]
    runs = []
    for workers in (1, 3):
        pool = executor(workers)
        try:
            initial = pool.run('feedback', [(p, tmp_path / f'w{workers}-initial-{i}', False, None)
                                           for i, p in enumerate(paths)], phase='M0')
            later = pool.run('feedback', [(p, tmp_path / f'w{workers}-later-{i}', False, initial[i][0])
                                         for i, p in enumerate(paths)], phase='M1')
            fits_result = pool.run('fit', [(p, None) for p in paths], phase='fits')
            runs.append((initial, later, fits_result))
            for context, _ in initial + later:
                assert context is not None
                assert all(not getattr(context, key).flags.writeable
                           for key in ('scale', 'psd', 'base_scale', 'noise_valid'))
        finally:
            pool.close()
    # Different publication paths/file checksums are incidental; compare the
    # actual model planes, full guard values and frozen numerical contexts.
    for left, right in zip(runs[0][:2], runs[1][:2], strict=True):
        for (lc, lr), (rc, rr) in zip(left, right, strict=True):
            assert lr['status'] == rr['status'] == 'ready'
            assert lr['guard'] == rr['guard']
            for key in ('scale', 'psd', 'base_scale', 'noise_valid'):
                assert np.array_equal(getattr(lc, key), getattr(rc, key), equal_nan=True)
            with fits.open(lr['model_path']) as a, fits.open(rr['model_path']) as b:
                assert [h.name for h in a] == [h.name for h in b]
                assert all(np.array_equal(x.data, y.data, equal_nan=True)
                           for x, y in zip(a, b, strict=True))
    assert runs[0][2] == runs[1][2]


def test_parallel_fit_failure_cannot_publish_completed_product_table(tmp_path):
    for array in products.ARRAYS:
        native_map(tmp_path / 'native' / array, array)
    class FailedFits:
        def run(self, kind, jobs, *, phase):
            assert kind == 'fit' and len(jobs) == 3
            assert all(path.is_file() for path, _ in jobs)
            raise RuntimeError('required fit worker failed')
    destination = tmp_path / '.hidden-publish/raw'
    with pytest.raises(RuntimeError, match='required fit'):
        products.publish_maps(tmp_path / 'native', destination, 152389, products.ARRAYS,
                              fit_diagnostics=True, verbose=False, array_executor=FailedFits())
    assert not (destination / 'point-table.json').exists()
    assert not (destination / 'products.json').exists()
