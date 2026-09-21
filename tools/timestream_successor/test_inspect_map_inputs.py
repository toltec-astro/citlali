import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from inspect_map_inputs import inspect, sha256


class InspectMapInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='map-input-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ptcdir = self.root / 'donor-continuity/ptc'
        self.caldir = self.root / 'donor-continuity/cal'
        self.ptcdir.mkdir(parents=True)
        self.caldir.mkdir(parents=True)
        self.cal = dict(schema='citlali-cal-output-v1', unit='mJy/nominal-beam',
                        output_VAL_generation=2, AST_frame='J2000-radec-gnomonic-tangent',
                        detectors=[dict(channel=4, selected_row='APT:row=40', array=0)])
        self.write(self.caldir / 'receipt.yaml', self.cal)
        self.learn = dict(network=0, observation=123, rows=3,
                          detectors=[dict(channel=4, occurrence='exact-detector-occurrence:row=40')])
        self.write(self.root / 'learning-receipt.yaml', self.learn)
        files = {}
        for suffix, data, dtype in [('-input.f64', [1,2,3], '<f8'),
                                    ('-centered.f64', [-1,0,1], '<f8'),
                                    ('-mean.f64', [2], '<f8'),
                                    ('-basis.f64', [1], '<f8'),
                                    ('-cleaned.f64', [-2,0,0], '<f8'),
                                    ('-eligible.u8', [1,1,0], 'u1'),
                                    ('-causes.u8', [0,0,1], 'u1'),
                                    ('-slots.i64', [0,1,2], '<i8')]:
            p = self.ptcdir / ('segment-0'+suffix)
            np.asarray(data, dtype=dtype).tofile(p)
            files[p.name] = sha256(p)
        self.ptc = dict(schema='citlali-ptc-output-v2', source_CAL_receipt_sha256=sha256(self.caldir/'receipt.yaml'),
                        input_VAL_generation=2, rank=1, method='observed-als', unit='mJy/nominal-beam',
                        centering='per-detector-eligible-arithmetic-mean-not-restored', processing_generation='scan-gen',
                        available=2, failed_fits=0, segments=[dict(network=0, scan=0, native_interval=[0,3],
                        output_detector_indices=[0], output_detectors=1, scheduled_times=3, retained=2,
                        converged=True, stopping_reason='converged', files=files)])
        self.save()

    @staticmethod
    def write(path, value):
        path.write_text(yaml.safe_dump(value))

    def save(self):
        self.write(self.ptcdir/'receipt.yaml', self.ptc)

    def test_real_zero_retained_but_inspection_never_grants_map_use(self):
        r = inspect(self.root)
        self.assertEqual(r['retained'], 2)
        self.assertEqual(r['numerical_integrity'], 'pass')
        self.assertEqual(r['MAP_readiness'], 'blocked')
        self.assertFalse(r['MAP_executed'])
        self.assertNotIn('MAP_product_identity', r)
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False))['verified_files'], r['verified_files'])

    def test_report_is_not_ptc_input(self):
        self.ptc['schema'] = 'spectral-diagnostic'
        self.save()
        with self.assertRaisesRegex(ValueError, 'actual successor PTC'):
            inspect(self.root)

    def test_corrupt_array_and_missing_required_component(self):
        p = self.ptcdir/'segment-0-cleaned.f64'
        p.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            inspect(self.root)
        del self.ptc['segments'][0]['files']['segment-0-cleaned.f64']
        self.save()
        with self.assertRaisesRegex(ValueError, 'component inventory'):
            inspect(self.root)

    def test_parent_generation_and_cal_content_are_exact(self):
        self.ptc['input_VAL_generation'] = 3
        self.save()
        with self.assertRaisesRegex(ValueError, 'generation mismatch'):
            inspect(self.root)
        self.ptc['input_VAL_generation'] = 2
        self.save()
        self.cal['unit'] = 'unrelated'
        self.write(self.caldir/'receipt.yaml', self.cal)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            inspect(self.root)

    def test_duplicate_detector_and_overlapping_segments(self):
        self.ptc['segments'][0]['output_detector_indices'] = [0,0]
        self.save()
        with self.assertRaisesRegex(ValueError, 'detector identity'):
            inspect(self.root)
        self.ptc['segments'][0]['output_detector_indices'] = [0]
        self.ptc['segments'] *= 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            inspect(self.root)

    def test_nonfinite_retained_is_failure_not_cleaned_or_filtered(self):
        p = self.ptcdir/'segment-0-cleaned.f64'
        np.array([np.nan,0,0], dtype='<f8').tofile(p)
        self.ptc['segments'][0]['files'][p.name] = sha256(p)
        self.save()
        with self.assertRaisesRegex(ValueError, 'nonfinite retained'):
            inspect(self.root)

    def test_failed_fit_is_not_promoted(self):
        self.ptc['segments'][0]['converged'] = False
        self.save()
        with self.assertRaisesRegex(ValueError, 'failed fit retained'):
            inspect(self.root)
        p = self.ptcdir/'segment-0-causes.u8'
        np.array([2,2,2], dtype='u1').tofile(p)
        self.ptc['segments'][0]['files'][p.name] = sha256(p)
        self.ptc['segments'][0]['retained'] = self.ptc['available'] = 0
        self.ptc['failed_fits'] = 1
        self.save()
        self.assertEqual(inspect(self.root)['failed_fits'], 1)

    def test_forged_extra_fields_cannot_grant_map_readiness(self):
        self.ptc['MAP_admission'] = 'PASS'
        self.ptc['coefficient'] = 1
        self.save()
        self.assertEqual(inspect(self.root)['MAP_readiness'], 'blocked')


if __name__ == '__main__':
    unittest.main()
