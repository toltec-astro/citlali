import numpy as np
from astropy.io import fits

from tools.fruit_loops.analyze_population_full import (
    first_two_pass,
    map_residual_metrics,
    valid_support_change_fraction,
)


def test_valid_support_change_fraction_uses_symmetric_difference():
    previous = np.array([[True, True], [False, False]])
    current = np.array([[True, False], [True, False]])

    assert valid_support_change_fraction(previous, current) == 2.0 / 3.0
    assert valid_support_change_fraction(previous, previous) == 0.0


def test_first_two_pass_enforces_minimum_and_consecutive_transitions():
    transitions = [
        {
            "current_iteration": iteration,
            "passed": iteration in {4, 5, 7, 8},
        }
        for iteration in range(1, 10)
    ]

    assert first_two_pass(transitions, field="passed") == 8


def test_map_residual_metrics_separates_source_aperture(tmp_path):
    final = np.ones((7, 7), dtype=float)
    stopped = final.copy()
    stopped[3, 3] = 2.0
    header = fits.Header(
        {
            "CRPIX1": 4.0,
            "CRPIX2": 4.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 1.0,
            "CDELT2": 1.0,
        }
    )
    final_path = tmp_path / "final.fits"
    stopped_path = tmp_path / "stopped.fits"
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(final, header=header, name="signal_I"),
    ]).writeto(final_path)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(stopped, header=header, name="signal_I"),
    ]).writeto(stopped_path)

    metrics = map_residual_metrics(
        stopped_path,
        final_path,
        center_x_arcsec=0.0,
        center_y_arcsec=0.0,
        aperture_radius_arcsec=0.1,
        background_sigma=0.5,
    )

    np.testing.assert_allclose(
        metrics["whole_map_relative_rms_to_iteration_9"], 1.0 / 7.0
    )
    assert metrics["source_aperture_relative_rms_to_iteration_9"] == 1.0
    assert (
        metrics["source_aperture_delta_rms_over_background_sigma"] == 2.0
    )
