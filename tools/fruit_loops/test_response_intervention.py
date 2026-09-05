from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from tools.fruit_loops import analyze_response_intervention as analysis
from tools.fruit_loops import run_response_intervention as runner


def fixture_metrics():
    fit = {"amplitude": 100.0, "major_fwhm_arcsec": 10.0, "minor_fwhm_arcsec": 9.0,
           "x_arcsec": 0.0, "y_arcsec": 0.0}
    return {"central_recovery": 1.0, "whole_kernel_recovery": 1.0, "major_ratio": 1.0,
            "minor_ratio": 1.0, "centroid_error": 0.0, "neptune_fit": fit,
            "regions": {region: {metric: {"rms": 1.0} for metric in
             ("transfer", "residual", "uninjected_residual")} for region in ("full", "source", "annulus", "neptune")}}


def test_all_array_protections_do_not_exchange_flux_support_or_neptune_for_leakage():
    h = fixture_metrics()
    c = copy.deepcopy(h)
    assert all(analysis.protections(h, c, 6, 0).values())
    c["regions"]["annulus"]["transfer"]["rms"] = .1
    c["central_recovery"] = .94
    assert not analysis.protections(h, c, 6, 1)["central_recovery_terminal"]
    assert not analysis.protections(h, c, 6, 1)["support"]
    c["neptune_fit"]["amplitude"] = 102
    assert not analysis.protections(h, c, 6, 0)["neptune_amplitude"]


def test_signed_leakage_cannot_cancel_and_unavailable_pixels_cannot_disappear():
    values = np.array([[-10., 10.]])
    summary = analysis.signed_summary(values, np.ones_like(values, dtype=bool))
    assert summary["rms"] == 10
    assert summary["negative_sum"] == -10 and summary["positive_sum"] == 10
    with pytest.raises(ValueError):
        analysis.signed_summary(np.array([[np.nan]]), np.array([[True]]))
    with pytest.raises(ValueError):
        analysis.signed_summary(values, np.zeros_like(values, dtype=bool))


def test_signed_wcs_regions_do_not_flip_injection_or_neptune():
    g = {"shape": [241, 241], "header": {"CRPIX1": 121., "CRPIX2": 121.,
         "CDELT1": -1., "CDELT2": 1., "CRVAL1": 0., "CRVAL2": 0.}}
    masks = analysis.regions(g)
    assert masks["source"][60, 120]
    assert not masks["source"][180, 120]
    assert masks["neptune"][115, 107]
    assert not masks["annulus"][115, 107]


def test_existing_gaussian_fit_and_full_model_agree_on_rotated_source():
    y, x = np.indices((91, 91), dtype=float)
    x -= 45
    y -= 45
    expected = 100 * np.exp(-.5 * (((x - 3) + (y + 4)) ** 2 / 50 + ((y + 4) - (x - 3)) ** 2 / 18))
    values = expected + 2
    old = analysis.gaussian_fit(values, 1., (3., -4.), 25.)
    new, model = analysis.gaussian_source_model(values, 1., (3., -4.), 25.)
    assert old == new
    np.testing.assert_allclose(model, expected, atol=1e-7)


def test_learning_comparison_uses_exact_ordered_iteration_fields_for_both_schemas():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "learning.csv"
        path.write_text("record_type,iter,value\na,0,1\na,1,2\nb,1,3\n")
        assert analysis.iteration_rows(path, 1)[1] == [("a", "1", "2"), ("b", "1", "3")]
        path.write_text("schema_version,iteration,value\nv1,1,2\nv1,0,9\n")
        assert analysis.iteration_rows(path, 1)[1] == [("v1", "1", "2")]


def test_checkpoint_audit_normalization_rejects_scientific_or_unlisted_change():
    with tempfile.TemporaryDirectory() as temporary:
        left, right = (Path(temporary) / name for name in ("a.nc", "b.nc"))
        for path, arm in ((left, None), (right, "H")):
            with Dataset(path, "w") as f:
                policy = f.createVariable("learning_policy_yaml", str)
                policy[()] = "enabled: true\n" + ("fruit_response_arm: H\n" if arm else "")
                value = f.createVariable("penalty_factor", "f8")
                value[()] = 0.
                if arm:
                    state = f.createVariable("fruit_response_state", str)
                    state[()] = "SCI-FRUIT-EL-F12-STATE-R0.1+CAP-001\nH 0\n0\n0\n0\n0\n"
        analysis.require_checkpoint_identity(left, right, audit_added=True)
        with Dataset(right, "a") as f:
            f["penalty_factor"][()] = .5
        with pytest.raises(ValueError, match="scientific checkpoint"):
            analysis.require_checkpoint_identity(left, right, audit_added=True)


def test_resource_ceiling_and_lossless_spool_retention():
    assert runner.resource_violation(3600, 43200, 16 * runner.GIB, 64 * runner.GIB, 32 * runner.GIB) is None
    assert runner.resource_violation(3601, 0, 0, 0, 0)
    assert runner.resource_violation(0, 0, 0, 0, 32 * runner.GIB + 1)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "fruit_response_synthetic.bin"
        path.write_bytes(b"synthetic retained occurrence\x00" * 1000)
        before = analysis.file_record(path)
        records = runner.compress_completed_spools(root)
        assert records[0]["original"] == before
        assert not path.exists() and path.with_suffix(".bin.gz").exists()


def test_run_controller_has_exact_predeclared_order_and_requires_no_implicit_retry():
    assert runner.ORDER == ["H0/uninjected", "H0/injected", "H/uninjected", "H/injected",
                            "Half/uninjected", "Half/injected", "Hold/uninjected", "Hold/injected"]
    assert len(runner.RESTART_ORDER) == 4
