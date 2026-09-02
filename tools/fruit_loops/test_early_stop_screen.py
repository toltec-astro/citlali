from __future__ import annotations

import pytest

from tools.fruit_loops import analyze_early_stop_screen as analysis


def manifest() -> dict:
    return {
        "test_id": "test",
        "trajectory_start_iteration": 0,
        "methods": {
            "reference": {
                "alpha": 1.0,
                "terminal_iteration": 6,
                "stop_iteration_exclusive": 7,
            },
            "candidate": {
                "alpha": 1.25,
                "terminal_iteration": 5,
                "stop_iteration_exclusive": 6,
            },
        },
        "screen": {
            "max_absolute_recovery_error_degradation": 0.01,
            "max_width_fractional_error": 0.03,
            "max_centroid_error_arcsec": 0.1,
            "max_residual_ratio_to_reference": 1.10,
            "minimum_pair_mean_wall_time_improvement_fraction": 0.10,
        },
    }


def row(alpha: float, iteration: int, array: str) -> dict:
    return {
        "alpha": alpha,
        "iteration": iteration,
        "array": array,
        "kernel_normalized_central_recovery": 0.98,
        "major_fwhm_over_kernel": 1.01,
        "minor_fwhm_over_kernel": 0.99,
        "centroid_error_arcsec": 0.05,
        "annular_residual_over_truth": 0.02,
        "kernel_residual_relative_rms": 0.04,
    }


def rows() -> list[dict]:
    result = []
    for array in analysis.ARRAYS:
        # Decoy rows ensure classification selects the exact predeclared
        # reference iteration 6 and candidate iteration 5.
        result.extend(
            [
                row(1.0, 5, array),
                row(1.0, 6, array),
                row(1.25, 4, array),
                row(1.25, 5, array),
            ]
        )
        result[-4]["centroid_error_arcsec"] = 9.0
        result[-2]["centroid_error_arcsec"] = 9.0
    return result


def execution(candidate_wall: float = 80.0) -> list[dict]:
    result = []
    for alpha, count, wall in ((1.0, 7, 100.0), (1.25, 6, candidate_wall)):
        for injected in (False, True):
            result.append(
                {
                    "trajectory": f"{alpha}-{injected}",
                    "alpha": str(alpha),
                    "injection": str(injected).lower(),
                    "status": "completed",
                    "completed_iterations": str(count),
                    "wall_seconds": str(wall),
                    "error_or_critical_messages": "0",
                }
            )
    return result


def test_iteration_set_accepts_only_exact_predeclared_range() -> None:
    analysis.require_iteration_set(
        [0, 1, 2, 3, 4, 5],
        start_iteration=0,
        stop_iteration_exclusive=6,
        context="candidate",
    )
    with pytest.raises(ValueError, match="iterations differ"):
        analysis.require_iteration_set(
            [0, 1, 2, 3, 4],
            start_iteration=0,
            stop_iteration_exclusive=6,
            context="candidate",
        )
    with pytest.raises(ValueError, match="iterations differ"):
        analysis.require_iteration_set(
            [0, 1, 2, 3, 4, 5, 6],
            start_iteration=0,
            stop_iteration_exclusive=6,
            context="candidate",
        )


def test_classification_uses_exact_terminal_iterations() -> None:
    result = analysis.classify(rows(), execution(), manifest())

    assert result["classification"] == "promising_early_stop_result"
    assert result["reference"]["terminal_iteration"] == 6
    assert result["candidate"]["terminal_iteration"] == 5
    assert result["scientific_protections_pass"]
    assert result["performance_target_pass"]
    assert result["restart_required"]


def test_scientific_failure_blocks_promising_result() -> None:
    values = rows()
    failed = analysis.select_row(values, 1.25, 5, "a1400")
    failed["annular_residual_over_truth"] = 0.03

    result = analysis.classify(values, execution(), manifest())

    assert result["classification"] == "does_not_replicate"
    assert not result["restart_required"]


def test_performance_miss_is_separate_from_scientific_result() -> None:
    result = analysis.classify(rows(), execution(candidate_wall=95.0), manifest())

    assert result["scientific_protections_pass"]
    assert not result["performance_target_pass"]
    assert result["classification"] == (
        "scientifically_replicates_but_misses_performance_target"
    )
    assert not result["restart_required"]
