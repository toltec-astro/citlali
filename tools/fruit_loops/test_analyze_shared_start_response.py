from __future__ import annotations

import numpy as np

from tools.fruit_loops import analyze_shared_start_response as analysis


def test_exact_telescoping_closure_passes() -> None:
    control = np.arange(12, dtype=float).reshape(3, 4)
    probe = control + 2.0
    without_penalty = probe - 0.5
    adaptive = without_penalty + 3.0
    total = adaptive - control
    shared = probe - control
    history = without_penalty - probe
    penalty = adaptive - without_penalty

    result = analysis.closure_metrics(
        total,
        shared,
        history,
        penalty,
        np.ones(total.shape, dtype=bool),
        64.0,
    )

    assert result["closure_pass"] is True
    assert result["maximum_absolute_residual_mjy_beam"] == 0.0


def test_telescoping_closure_rejects_material_error() -> None:
    total = np.ones((2, 2), dtype=float)
    shared = np.full((2, 2), 0.25)
    history = np.full((2, 2), 0.25)
    penalty = np.full((2, 2), 0.25)

    result = analysis.closure_metrics(
        total,
        shared,
        history,
        penalty,
        np.ones(total.shape, dtype=bool),
        64.0,
    )

    assert result["closure_pass"] is False


def test_annulus_excludes_neptune_region() -> None:
    manifest = {
        "injected_source_radius_arcsec": 20.0,
        "neptune_radius_arcsec": 20.0,
        "annulus_inner_arcsec": 40.0,
        "annulus_outer_arcsec": 120.0,
        "annulus_neptune_exclusion_radius_arcsec": 25.0,
    }
    masks = analysis.make_region_masks(
        (301, 301), 1.0, (0.0, -60.0), (12.5, -5.3), manifest
    )
    xx, yy = analysis.coordinate_grid((301, 301), 1.0)
    near_neptune = np.hypot(xx - 12.5, yy + 5.3) <= 25.0

    assert not np.any(
        masks["annulus_r40_120_excluding_neptune_r25"] & near_neptune
    )
    assert np.any(masks["annulus_r40_120_excluding_neptune_r25"])


def test_cross_terms_retain_sign_and_cosine() -> None:
    base = np.array([[1.0, -2.0], [3.0, -4.0]])
    components = {
        "T5_total_adaptive": 2.0 * base,
        "S5_shared_start": base,
        "H5_other_history": -base,
        "D4460_5": 2.0 * base,
    }
    support = np.ones(base.shape, dtype=bool)
    regions = {name: support for name in analysis.REGIONS}

    rows = analysis.cross_term_rows("a1400", components, support, regions)
    selected = next(
        row for row in rows
        if row["left_component"] == "S5_shared_start"
        and row["right_component"] == "H5_other_history"
        and row["region"] == "complete_map"
    )

    assert selected["mean_product_mjy2_beam2"] < 0.0
    assert selected["cosine"] == -1.0


def test_roundoff_bound_scales_with_largest_map_value() -> None:
    small = np.ones((2, 2))
    large = np.full((2, 2), 1.0e6)

    assert analysis.roundoff_bound([large], 64.0) > analysis.roundoff_bound(
        [small], 64.0
    )
