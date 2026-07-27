from __future__ import annotations

import numpy as np

from tools.fruit_loops.source_morphology import (
    centered_template_samples,
    convolved_source_template,
    morphology_template_metrics,
    uniform_disk_pixel_kernel,
)
from tools.fruit_loops.compare_injected_source_pair import gaussian_fit


def test_uniform_disk_kernel_is_normalized_and_symmetric() -> None:
    disk = uniform_disk_pixel_kernel(3.7, 1.0)

    assert disk.shape[0] % 2 == 1
    assert disk.shape[1] % 2 == 1
    np.testing.assert_allclose(
        np.sum(disk), 1.0, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(disk, disk[::-1, ::-1])


def test_disk_convolution_broadens_realized_kernel() -> None:
    axis = np.arange(-40.0, 41.0)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / 2.0**2)

    point = convolved_source_template(
        kernel, diameter_arcsec=0.0, pixel_size_arcsec=1.0
    )
    planet = convolved_source_template(
        kernel, diameter_arcsec=3.7, pixel_size_arcsec=1.0
    )

    assert planet.max() < point.max()
    np.testing.assert_allclose(
        np.sum(planet), np.sum(point), rtol=1.0e-12
    )


def test_morphology_template_recovers_injected_disk_scale() -> None:
    axis = np.arange(-80.0, 81.0)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / 2.5**2)
    template = convolved_source_template(
        kernel, diameter_arcsec=3.7, pixel_size_arcsec=1.0
    )
    signal = 42.0 * template + 3.0
    weight = np.ones_like(signal)
    coverage = np.ones_like(signal, dtype=bool)

    metrics = morphology_template_metrics(
        signal,
        weight,
        coverage,
        kernel,
        axis,
        axis,
        center_x_arcsec=0.0,
        center_y_arcsec=0.0,
        source_angular_diameter_arcsec=3.7,
    )

    np.testing.assert_allclose(
        metrics["morphology_template_amplitude_scale"],
        42.0,
        rtol=1.0e-8,
    )
    assert (
        metrics["morphology_template_major_fwhm_arcsec"]
        > 2.355 * 2.5
    )


def test_morphology_template_recenters_offset_realized_kernel() -> None:
    axis = np.arange(-80.0, 81.0)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(
        -0.5 * ((xx - 2.0) ** 2 + (yy + 1.0) ** 2) / 2.5**2
    )
    template = convolved_source_template(
        kernel, diameter_arcsec=3.7, pixel_size_arcsec=1.0
    )
    template_fit = gaussian_fit(template, 1.0)
    samples = centered_template_samples(
        template,
        axis,
        axis,
        xx,
        yy,
        center_x_arcsec=5.0,
        center_y_arcsec=-4.0,
        template_center_x_arcsec=template_fit["x_arcsec"],
        template_center_y_arcsec=template_fit["y_arcsec"],
    )
    signal = 42.0 * samples + 3.0

    metrics = morphology_template_metrics(
        signal,
        np.ones_like(signal),
        np.ones_like(signal, dtype=bool),
        kernel,
        axis,
        axis,
        center_x_arcsec=5.0,
        center_y_arcsec=-4.0,
        source_angular_diameter_arcsec=3.7,
    )

    np.testing.assert_allclose(
        metrics["morphology_template_amplitude_scale"],
        42.0,
        rtol=1.0e-8,
    )
