from __future__ import annotations

import math

from tools.fruit_loops import analyze_population_stage as analysis


def transition(iteration: int, value: float) -> dict:
    return {
        "current_iteration": iteration,
        "interpretable": True,
        "kernel_normalized_amplitude_change_fraction": value,
        "maximum_fwhm_change_fraction": value,
        "centroid_step_arcsec": 0.05,
        "successive_map_relative_rms": value,
        "sig2noise_change_fraction": value,
    }


def test_first_two_transition_pass_requires_consecutive_pair() -> None:
    rows = [
        transition(1, 0.20),
        transition(2, 0.009),
        transition(3, 0.011),
        transition(4, 0.008),
        transition(5, 0.007),
    ]

    assert analysis.first_two_transition_pass(
        rows, field="successive_map_relative_rms", maximum=0.01
    ) == 5
    assert analysis.first_combined_pass(rows, tolerance=0.01) == 5


def test_combined_pass_keeps_fixed_centroid_limit() -> None:
    rows = [transition(1, 0.009), transition(2, 0.009)]
    rows[1]["centroid_step_arcsec"] = 0.11

    assert analysis.first_combined_pass(rows, tolerance=0.10) is None


def test_individual_diagnostic_uses_its_own_eligibility() -> None:
    rows = [transition(1, 0.009), transition(2, 0.008)]
    for row in rows:
        row["interpretable"] = False
        row["source_association_valid"] = True

    assert analysis.first_two_transition_pass(
        rows,
        field="kernel_normalized_amplitude_change_fraction",
        maximum=0.01,
        eligibility_field="source_association_valid",
    ) == 2
    assert analysis.first_combined_pass(rows, tolerance=0.01) is None


def test_nonfinite_fit_is_not_valid() -> None:
    row = {
        field: 1.0 for field in analysis.CORE_FIELDS
    }
    row.update(
        {
            "kernel_fit_amplitude": 1.0,
            "major_fwhm_arcsec": 1.0,
            "minor_fwhm_arcsec": 1.0,
            "kernel_major_fwhm_arcsec": 1.0,
            "kernel_minor_fwhm_arcsec": 1.0,
        }
    )
    assert analysis.fit_is_valid(row)
    row["sig2noise"] = math.nan
    assert not analysis.fit_is_valid(row)


def test_stratum_summary_preserves_failed_yield() -> None:
    rows = []
    for obsnum, passed_arrays in ((1, 3), (2, 1)):
        rows.append(
            {
                "obsnum": obsnum,
                "quality_stratum": "stress",
                "tolerance_percent": 2,
                "interpretable_arrays": 3,
                "source_associated_arrays": 3,
                "psf_interpretable_arrays": passed_arrays,
                "arrays_with_any_combined_pass": passed_arrays,
                "all_arrays_have_combined_pass": passed_arrays == 3,
                "arrays_passing_endpoint_window": passed_arrays,
                "all_arrays_pass_endpoint_window": passed_arrays == 3,
                "worst_endpoint_sig2noise_ratio_seed": 0.8,
                "maximum_endpoint_centroid_shift_from_seed_arcsec": 0.2,
            }
        )

    summary = analysis.stratum_summary(rows)

    assert summary[0]["observation_count"] == 2
    assert summary[0]["observations_all_arrays_ever_pass"] == 1
    assert summary[0]["array_pass_count"] == 4
