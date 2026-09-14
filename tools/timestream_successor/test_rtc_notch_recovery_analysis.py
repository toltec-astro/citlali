"""Numerical controls for the finite conditioned-output diagnostic."""

import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "run_rtc_notch_recovery", ROOT / "run_rtc_notch_recovery.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sys.modules["run_rtc_notch_recovery"] = runner
spec = importlib.util.spec_from_file_location(
    "analyze_recovery", ROOT / "analyze_rtc_notch_recovery.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_known_sine_power_and_windows_do_not_bridge_invalid_support():
    t = np.arange(4096) / 128
    x = 3 * np.sin(2 * np.pi * 11 * t)
    mask = np.ones(len(t), bool)
    mask[1800:2000] = False
    x[~mask] = np.nan
    f, p, support = m.measured_psd(x, mask, 1 / 128)
    assert np.isclose(p.sum() * (f[1] - f[0]), 4.5, rtol=1e-5)
    assert all(hi <= 1800 or lo >= 2000 for lo, hi in support)
    assert abs(f[np.argmax(p)] - 11) < 1e-14


def test_short_fragments_are_not_combined_into_artificial_windows():
    mask = np.ones(2048, bool)
    mask[::200] = False
    with pytest.raises(ValueError, match="fewer than two"):
        m.measured_psd(np.arange(2048), mask, 1 / 128)


def test_paired_source_unity_transfer_and_complete_rejection():
    t = np.arange(4096) / 128
    source = np.exp(-0.5 * ((t - 16) / 0.1) ** 2)
    rows = np.arange(0, len(t), 2)
    r = m.source_metrics(source, source, t, rows, 16)
    assert r["template_amplitude_ratio"] == 1
    assert r["waveform_error_fraction"] == 0
    assert r["centroid_shift_ms"] == 0
    assert r["retained_local_source_energy_fraction"] == 1
    r = m.source_metrics(source, source, t, np.array([], dtype=int), 16)
    assert (
        not r["transfer_available"] and r["retained_local_source_energy_fraction"] == 0
    )


def test_overlap_uses_partial_native_cells_once():
    t = np.array([0.0, 1.0, 2.0])
    mask = np.array([True, False, True])
    assert m.overlap_seconds(t, 1, mask, [[-250000, 250000], [1750000, 2250000]]) == 1
