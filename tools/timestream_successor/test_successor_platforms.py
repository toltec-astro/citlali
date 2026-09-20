"""Cross-platform reports must not confuse byte equality with scientific identity."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml
from compare_successor_platforms import compare
from ptc_spod import digest


class PlatformComparison(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.a=Path(self.tmp.name)/'a';self.b=Path(self.tmp.name)/'b'
        for root in (self.a,self.b):
            cal=root/'donor-continuity/cal';ptc=root/'donor-continuity/ptc'
            cal.mkdir(parents=True);ptc.mkdir()
            np.array([1.,2.],dtype='<f8').tofile(cal/'value.f64')
            np.array([1,1],dtype='u1').tofile(ptc/'eligible.u8')
            learning=dict(schema='test',observation=1,network=0,rows=2,native_runs=[[0,2]],native_time_sha256='time',
                VAL_generation=0,original_parent='raw-and-tune',source_protection='empty',source_protection_authority='owner',
                native_integration_seconds=.01,cadence_interval_seconds=.01,spectral_estimator='accepted',spectral_conventions='accepted',
                spectra=[dict(channel=0,windows=[[0,2]],available=True)],
                explicit_array_lowpass=dict(path='/relocated/filter',sha256='fixed',array=0,factor=1),
                detectors=[dict(detector=0,channel=0,samples_sha256='original',occurrence='fixed-row',factor_authority='selected')])
            self.write(root/'learning-receipt.yaml',learning)
            self.write(cal/'receipt.yaml',dict(schema='cal',unit='mJy/nominal-beam',input_VAL_generation=1,output_VAL_generation=2,
                detectors=[dict(channel=0,selected_row='APT-row-0',available=2,joint_cause_counts={0:2},files={'value.f64':digest(cal/'value.f64')})]))
            self.write(ptc/'receipt.yaml',dict(schema='ptc',input_VAL_generation=2,output_VAL_generation=3,
                source_CAL_receipt_sha256=digest(cal/'receipt.yaml'),CAL_classification=str(cal/'receipt.yaml'),
                segments=[dict(detector_indices=[0],output_detector_indices=[0],native_interval=[0,2],converged=True,
                    iterations=2,objective=[4.,1.],files={'eligible.u8':digest(ptc/'eligible.u8')})]))

    @staticmethod
    def write(path,d):path.write_text(yaml.safe_dump(d))

    def mutate(self,relative,fn,rebind=False):
        path=self.b/relative;d=yaml.safe_load(path.read_text());fn(d);self.write(path,d)
        if rebind:
            p=self.b/'donor-continuity/ptc/receipt.yaml';d=yaml.safe_load(p.read_text());d['source_CAL_receipt_sha256']=digest(path);self.write(p,d)

    def test_identical_and_relocated_reference(self):
        self.mutate('donor-continuity/ptc/receipt.yaml',lambda d:d.update(CAL_classification='/old/location/donor-continuity/cal/receipt.yaml'))
        self.assertTrue(compare(self.a,self.b)['exact_identity_timing_mask_support'])

    def test_changed_ptc_mapping_or_boundary_rejected_with_identical_arrays(self):
        p=self.b/'donor-continuity/ptc/receipt.yaml';original=yaml.safe_load(p.read_text())
        for key,value in [('detector_indices',[1]),('output_detector_indices',[1]),('native_interval',[1,3]),('converged',False)]:
            with self.subTest(key=key):
                d=copy.deepcopy(original);d['segments'][0][key]=value;self.write(p,d)
                with self.assertRaises(Exception):compare(self.a,self.b)
        self.write(p,original)

    def test_changed_cal_identity_and_val_rejected(self):
        p=self.b/'donor-continuity/cal/receipt.yaml';original=yaml.safe_load(p.read_text())
        for kind in ('row','VAL','unit'):
            with self.subTest(kind=kind):
                self.write(p,original)
                def change(d):
                    if kind=='row':d['detectors'][0]['selected_row']='APT-row-1'
                    elif kind=='VAL':d['output_VAL_generation']=99
                    else:d['unit']='K'
                self.mutate('donor-continuity/cal/receipt.yaml',change,True)
                with self.assertRaises(Exception):compare(self.a,self.b)

    def test_stale_parent_digest_rejected(self):
        self.mutate('donor-continuity/ptc/receipt.yaml',lambda d:d.update(source_CAL_receipt_sha256='stale'))
        with self.assertRaises(Exception):compare(self.a,self.b)

    def test_stale_artifact_digest_rejected(self):
        np.array([2.,3.],dtype='<f8').tofile(self.b/'donor-continuity/cal/value.f64')
        with self.assertRaises(Exception):compare(self.a,self.b)

    def test_float_differences_reported_without_new_tolerance(self):
        p=self.b/'donor-continuity/cal/value.f64';np.array([1.,2.000001],dtype='<f8').tofile(p)
        self.mutate('donor-continuity/cal/receipt.yaml',lambda d:d['detectors'][0]['files'].update({'value.f64':digest(p)}),True)
        r=compare(self.a,self.b);self.assertTrue(r['exact_identity_timing_mask_support']);self.assertEqual(r['numeric_files_different'],1)
        self.assertIn('require-review',r['numerical_disposition'])

    def test_original_numeric_binding_and_solver_progress_are_reported(self):
        self.mutate('learning-receipt.yaml',lambda d:d['detectors'][0].update(samples_sha256='platform-numerics'))
        self.mutate('donor-continuity/ptc/receipt.yaml',lambda d:d['segments'][0].update(iterations=3))
        r=compare(self.a,self.b);self.assertTrue(r['exact_identity_timing_mask_support'])
        self.assertFalse(r['original_pair_content_bindings_match']);self.assertTrue(r['solver_statistics_differences'])
        self.assertIn('require-review',r['numerical_disposition'])

    def test_changed_mask_is_failure_even_with_updated_digest(self):
        p=self.b/'donor-continuity/ptc/eligible.u8';np.array([0,1],dtype='u1').tofile(p)
        self.mutate('donor-continuity/ptc/receipt.yaml',lambda d:d['segments'][0]['files'].update({'eligible.u8':digest(p)}))
        self.assertFalse(compare(self.a,self.b)['exact_identity_timing_mask_support'])


if __name__=='__main__':unittest.main()
