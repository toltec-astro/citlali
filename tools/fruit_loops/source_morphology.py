"""Source-morphology-aware template metrics for fruit-loop map analysis."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from tools.fruit_loops.compare_injected_source_pair import gaussian_fit


MINIMUM_FIT_PIXELS = 12


def uniform_disk_pixel_kernel(
    diameter_arcsec: float,
    pixel_size_arcsec: float,
    *,
    oversample: int = 16,
) -> np.ndarray:
    """Return a pixel-integrated, unit-sum circular disk kernel."""
    if not math.isfinite(diameter_arcsec) or diameter_arcsec <= 0.0:
        return np.ones((1, 1), dtype=float)
    if not math.isfinite(pixel_size_arcsec) or pixel_size_arcsec <= 0.0:
        raise ValueError("pixel_size_arcsec must be finite and positive")
    if oversample < 2:
        raise ValueError("oversample must be at least 2")

    radius_pixels = 0.5 * diameter_arcsec / pixel_size_arcsec
    half_size = int(math.ceil(radius_pixels + 1.0))
    axis = np.arange(-half_size, half_size + 1, dtype=float)
    offsets = (np.arange(oversample, dtype=float) + 0.5) / oversample - 0.5
    yy, xx, sub_y, sub_x = np.meshgrid(
        axis, axis, offsets, offsets, indexing="ij"
    )
    inside = (
        np.square(xx + sub_x) + np.square(yy + sub_y)
        <= radius_pixels**2
    )
    kernel = np.mean(inside, axis=(2, 3), dtype=float)
    total = float(np.sum(kernel))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("disk pixel integration produced zero support")
    return kernel / total


def convolved_source_template(
    kernel: np.ndarray,
    *,
    diameter_arcsec: float,
    pixel_size_arcsec: float,
) -> np.ndarray:
    """Convolve a realized point-source kernel with a uniform source disk."""
    values = np.where(np.isfinite(kernel), kernel, 0.0)
    if not np.any(values):
        raise ValueError("realized kernel has no finite nonzero samples")
    disk = uniform_disk_pixel_kernel(
        diameter_arcsec, pixel_size_arcsec
    )
    if disk.shape == (1, 1):
        return values.copy()
    return fftconvolve(values, disk, mode="same")


def centered_template_samples(
    template: np.ndarray,
    x_axis_arcsec: np.ndarray,
    y_axis_arcsec: np.ndarray,
    xx_arcsec: np.ndarray,
    yy_arcsec: np.ndarray,
    *,
    center_x_arcsec: float,
    center_y_arcsec: float,
    template_center_x_arcsec: float = 0.0,
    template_center_y_arcsec: float = 0.0,
) -> np.ndarray:
    """Evaluate a zero-centered map template at a requested source center."""
    if len(x_axis_arcsec) < 2 or len(y_axis_arcsec) < 2:
        raise ValueError("template axes need at least two pixels")
    dx = float(x_axis_arcsec[1] - x_axis_arcsec[0])
    dy = float(y_axis_arcsec[1] - y_axis_arcsec[0])
    if dx == 0.0 or dy == 0.0:
        raise ValueError("template axes must be monotonic")
    relative_x = (
        xx_arcsec - center_x_arcsec + template_center_x_arcsec
    )
    relative_y = (
        yy_arcsec - center_y_arcsec + template_center_y_arcsec
    )
    column = (relative_x - float(x_axis_arcsec[0])) / dx
    row = (relative_y - float(y_axis_arcsec[0])) / dy
    return map_coordinates(
        template,
        (row, column),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def weighted_template_amplitude(
    signal: np.ndarray,
    weight: np.ndarray,
    valid: np.ndarray,
    template_samples: np.ndarray,
    selected: np.ndarray,
) -> tuple[float, float]:
    """Fit a supplied template scale plus a constant weighted background."""
    use = (
        valid
        & selected
        & np.isfinite(template_samples)
        & np.isfinite(signal)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    if int(np.count_nonzero(use)) < MINIMUM_FIT_PIXELS:
        return math.nan, math.nan
    template = template_samples[use]
    values = signal[use]
    weights = weight[use]
    sum_weight = float(np.sum(weights))
    sum_template = float(np.sum(weights * template))
    sum_template2 = float(np.sum(weights * template * template))
    sum_signal = float(np.sum(weights * values))
    sum_template_signal = float(np.sum(weights * template * values))
    determinant = (
        sum_template2 * sum_weight - sum_template * sum_template
    )
    if (
        not math.isfinite(determinant)
        or determinant <= 0.0
        or sum_weight <= 0.0
    ):
        return math.nan, math.nan
    amplitude = (
        sum_template_signal * sum_weight - sum_signal * sum_template
    ) / determinant
    uncertainty = math.sqrt(sum_weight / determinant)
    return float(amplitude), float(uncertainty)


def morphology_template_metrics(
    signal: np.ndarray,
    weight: np.ndarray,
    coverage: np.ndarray,
    kernel: np.ndarray,
    x_axis_arcsec: np.ndarray,
    y_axis_arcsec: np.ndarray,
    *,
    center_x_arcsec: float,
    center_y_arcsec: float,
    source_angular_diameter_arcsec: float,
) -> dict[str, float]:
    """Measure source scale against the realized kernel convolved with a disk."""
    pixel_size_arcsec = math.sqrt(
        abs(
            float(x_axis_arcsec[1] - x_axis_arcsec[0])
            * float(y_axis_arcsec[1] - y_axis_arcsec[0])
        )
    )
    template = convolved_source_template(
        kernel,
        diameter_arcsec=source_angular_diameter_arcsec,
        pixel_size_arcsec=pixel_size_arcsec,
    )
    template_fit = gaussian_fit(template, pixel_size_arcsec)
    xx_arcsec, yy_arcsec = np.meshgrid(x_axis_arcsec, y_axis_arcsec)
    samples = centered_template_samples(
        template,
        x_axis_arcsec,
        y_axis_arcsec,
        xx_arcsec,
        yy_arcsec,
        center_x_arcsec=center_x_arcsec,
        center_y_arcsec=center_y_arcsec,
        template_center_x_arcsec=float(template_fit["x_arcsec"]),
        template_center_y_arcsec=float(template_fit["y_arcsec"]),
    )
    radius = np.hypot(
        xx_arcsec - center_x_arcsec,
        yy_arcsec - center_y_arcsec,
    )
    fit_radius = 3.0 * float(template_fit["major_fwhm_arcsec"])
    valid = (
        coverage
        & np.isfinite(signal)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    amplitude, uncertainty = weighted_template_amplitude(
        signal,
        weight,
        valid,
        samples,
        radius <= fit_radius,
    )
    return {
        "morphology_template_amplitude_scale": amplitude,
        "morphology_template_amplitude_uncertainty": uncertainty,
        "morphology_template_formal_sig2noise": (
            amplitude / uncertainty
            if math.isfinite(amplitude)
            and math.isfinite(uncertainty)
            and uncertainty > 0.0
            else math.nan
        ),
        "morphology_template_major_fwhm_arcsec":
            float(template_fit["major_fwhm_arcsec"]),
        "morphology_template_minor_fwhm_arcsec":
            float(template_fit["minor_fwhm_arcsec"]),
        "morphology_template_center_x_arcsec":
            float(template_fit["x_arcsec"]),
        "morphology_template_center_y_arcsec":
            float(template_fit["y_arcsec"]),
        "morphology_template_fit_radius_arcsec": fit_radius,
    }
