"""Controls for width semantics and honest finite-support comparisons."""

import numpy as np
import pytest
from scipy import signal

from run_rtc_notch_recovery import finite_trial
from rtc_wide_notch_experiment import centered_band, feature_edges
from analyze_rtc_wide_notch import power_record, source_record


def test_default_width_is_exact_previous_construction_and_wider_cutoff_separation():
    dt = 0.008192062377929688
    center = 11.006255844577186
    for width in (0.5, 1.0, 2.0):
        for seconds in (1, 3, 6):
            half = int(np.floor(seconds / (2 * dt)))
            h = signal.firwin(
                2 * half + 1,
                [center - width / 2, center + width / 2],
                pass_zero="bandstop",
                window="hann",
                fs=1 / dt,
                scale=True,
            )
            h = (h + h[::-1]) / 2
            h /= sum(h)
            trial = finite_trial(
                dt, center, seconds, np.ones(307) / 307, full_width_hz=width
            )
            assert np.array_equal(h, trial["finite_notch"])
            assert trial["cumulative_half_seconds"] == (half + 153) * dt <= 5
            assert abs(sum(h) - 1) < 1e-12
    assert finite_trial(dt, center, 6, [1.0]) == finite_trial(
        dt, center, 6, [1.0], full_width_hz=0.5
    )


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf"), 23, 130])
def test_unphysical_cutoffs_are_rejected(width):
    with pytest.raises(ValueError, match="design domain"):
        finite_trial(1 / 128, 11, 1, [1], full_width_hz=width)


def test_realized_band_is_not_the_requested_band_or_disconnected_ripple():
    f = np.arange(7.0)
    assert centered_band(
        f, np.array([True, False, True, True, True, False, True]), 3
    ) == [2, 4]
    assert (
        centered_band(f, np.array([True, False, True, False, True, False, True]), 3)
        is None
    )
    assert feature_edges(
        {
            "audit_record": {
                "features_x": [
                    dict(
                        family="around11", first_hz=10.5, last_hz=11.5, bin_span_hz=1.25
                    )
                ]
            }
        }
    ) == [10.375, 11.625]


def test_fixed_band_total_power_and_declared_gaps_without_window_joining():
    t = np.arange(4096) / 128
    x = 3 * np.sin(2 * np.pi * 11 * t)
    mask = np.ones(len(t), bool)
    mask[1800:2100] = False
    x[~mask] = np.nan
    p = power_record(x, mask, 1 / 128, [10.375, 11.625])
    assert p["available"] and np.isclose(p["feature_total_power"], 4.5, rtol=2e-5)
    assert all(b <= 1800 or a >= 2100 for a, b in p["windows"])
    mask[1800] = True
    with pytest.raises(ValueError, match="unexpected nonfinite"):
        power_record(x, mask, 1 / 128, [10.375, 11.625])
    mask[:] = False
    mask[::2] = True
    assert not power_record(np.zeros(len(t)), mask, 1 / 128, [10.375, 11.625])[
        "available"
    ]


def test_complete_source_unity_and_missing_core_not_recovered_tail():
    t = np.arange(4096) / 128
    xy = np.column_stack([(t - 16) * 20, np.zeros(len(t))])
    source = np.exp(-0.5 * ((t - 16) / 0.15) ** 2)
    mask = np.ones(len(t), bool)
    c = dict(kind="compact", time=16, center_arcsec=[0, 0])
    r = source_record(source, source, mask, t, xy, c)
    assert (
        r["available"]
        and r["peak_ratio"] == 1
        and r["waveform_rms_error_fraction"] == 0
    )
    assert r["integrated_response_ratio"] is None
    mask[2048] = False
    r = source_record(source, source, mask, t, xy, c)
    assert not r["available"] and not r["complete_crossing"]


def test_extended_domain_cannot_be_truncated_by_local_search_or_observation_edge():
    t = np.arange(4096) / 128
    xy = np.column_stack([(t - 16) * 2, np.zeros(len(t))])
    source = np.exp(-0.5 * ((t - 16) / 5) ** 2)
    mask = np.ones(len(t), bool)
    c = dict(kind="extended", time=16, center_arcsec=[0, 0], apparent_FWHM_arcsec=25.45)
    r = source_record(source, source, mask, t, xy, c)
    assert not r["available"] and not r["complete_crossing"]
