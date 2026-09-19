"""Focused numerical checks of the selected development artifacts."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from scipy import signal

from prepare_rtc_array_lowpass import FREQUENCIES, INTERVAL, optical, prepare, source_probes

ARTIFACTS = Path(__file__).resolve().parents[2] / "validation/rtc_development_lowpass_2026-09-19"


class ArrayLowpassTests(unittest.TestCase):
    def test_fixed_artifact_reproducibility_and_cadence(self):
        for array in FREQUENCIES:
            with self.subTest(array=array):
                plan = json.loads((ARTIFACTS / (array + '.json')).read_text())
                h = np.asarray(plan['coefficients'], dtype='<f8')
                self.assertEqual(hashlib.sha256(h.tobytes()).hexdigest(), plan['coefficients_sha256'])
                self.assertFalse(plan['production_certified'])
                self.assertFalse(plan['automatic_selection'])
                self.assertEqual(plan['factor'], 2 if array == 'a2000' else 1)
                if array != 'a2000':
                    rebuilt, _ = prepare(array)
                    np.testing.assert_array_equal(rebuilt['coefficients'], h)
                else:
                    self.assertEqual(len(h), 307)
                    self.assertEqual(plan['coefficients_sha256'],
                                     'e25377075b9b20147bfc11b9c4b9ab792dd60576166f6c25dbc8b9aaf75e8967')

    def test_response_at_both_allowed_cadence_endpoints(self):
        for array in FREQUENCIES:
            plan = json.loads((ARTIFACTS / (array + '.json')).read_text())
            h = np.array(plan['coefficients'])
            band, _ = optical(array)
            for relative in (-.0001, .0001):
                with self.subTest(array=array, relative=relative):
                    fs = 1 / (INTERVAL * (1 + relative))
                    f, response = signal.freqz(h, worN=262145, include_nyquist=True, fs=fs)
                    self.assertLess(np.max(np.abs(np.abs(response[f <= band]) - 1)), .01)
                    stop = fs / (2 * plan['factor'])
                    self.assertLess(np.max(np.abs(response[f >= stop * (1 - .0001)])), .001)
                    self.assertLess(abs(h.sum() - 1), 1e-12)
                    # No new decimation images at M=1. This is not a claim
                    # about contamination already aliased by the readout.
                    self.assertLess(band, stop)
                    for hz in (11., 29.):
                        if hz <= band:
                            _, r = signal.freqz(h, worN=[hz], fs=fs)
                            self.assertLess(abs(abs(r[0]) - 1), .01)

    def test_paired_source_transfer_on_identical_complete_support(self):
        for array in FREQUENCIES:
            plan = json.loads((ARTIFACTS / (array + '.json')).read_text())
            probes = source_probes(plan)
            self.assertEqual(len(probes), 16)
            self.assertTrue(any(not row['center_sampling_admitted'] for row in probes))
            for row in probes:
                with self.subTest(array=array, speed=row['speed_arcsec_per_sec'], phase=row['phase_native_samples']):
                    self.assertLess(row['maximum_residual_over_native_peak'], .01)
                    self.assertLess(abs(row['integrated_relative_change']), .01)


if __name__ == '__main__':
    unittest.main()
