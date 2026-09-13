"""Admission/failure and data-only usability checks; no PTC or saved-map solve."""
import unittest
from unittest.mock import patch
import numpy as np
import nominal
import evaluation

class Tests(unittest.TestCase):
    def test_projected_gradient(self):
        c = nominal.stopping.completion(np.array([0.,2.]), 3., np.array([4.,.001]), 10.)
        self.assertEqual(c['relative_projected_gradient'], 1e-4)
        self.assertTrue(c['finite'] and c['feasible'])
        self.assertFalse(nominal.stopping.completion(np.array([-.1]),0.,np.zeros(1),1.)['feasible'])
        self.assertFalse(nominal.stopping.completion(np.zeros(1),0.,np.array([np.nan]),1.)['finite'])
    def test_adaptor_does_not_apply_failed_or_negative_iterate(self):
        e = object.__new__(nominal.Estimator)
        e.sigmas = np.ones((5,2)); e.D = np.array([True,True,False])
        p = dict(omega=np.ones((5,3),bool), background=np.zeros(3), scale=2.,gradient_scale=1.)
        for available,v,expected in [(True,[1.,2.],True),(False,[1.,2.],False),(True,[-1.,2.],False),(True,[0.,0.],False)]:
            snap=dict(v=np.array(v), finite=True,feasible=True,iterations=3,function_evaluations=5,relative_projected_gradient=1e-5)
            solved=dict(snapshots={'declared':snap},final=snap,operational_stop_available=available,solver_options={},termination='test',trace=[],wall_seconds=0.)
            with patch.object(nominal.stopping,'problem',return_value=p), patch.object(nominal.stopping,'solve_path',return_value=solved) as solve:
                u,d,_=e.infer(np.zeros(3))
                solve.assert_called_once_with(p,qualify=False)
            self.assertEqual(d['available'],expected)
            if expected: np.testing.assert_array_equal(u,[2.,4.,0.])
            else: self.assertFalse(u.any())
        p['omega'][:]=False
        with patch.object(nominal.stopping,'problem',return_value=p), patch.object(nominal.stopping,'solve_path') as solve:
            u,d,_=e.infer(np.zeros(3)); solve.assert_not_called()
            self.assertTrue(d['available']); self.assertFalse(d['admitted']); self.assertFalse(u.any())
    def test_shape_and_truth_do_not_veto_usable_centroid(self):
        original=dict(source_present=True,fit=dict(parameters=[1.]*9,peak=10.,widths=[12.,8.],boundary_rejection=False),shape_core_pixels=20,boundary_warning=False,distortion_warning=True,empirical_peak_score=4.)
        probe=dict(available=True,centroid_difference_arcsec=.1,peak_relative_difference=.001)
        a=evaluation.repaired.judgments(original,True,probe)
        original.update(injected_centroid=[999.,999.],centroid_error_arcsec=10000.,arm='C')
        self.assertEqual(a,evaluation.repaired.judgments(original,True,probe))
        self.assertTrue(a['centroid_usable'] and a['shape_warning'])
        self.assertFalse(a['peak_response_usable'])
        original['boundary_warning']=True
        self.assertFalse(evaluation.repaired.judgments(original,True,probe)['centroid_usable'])
if __name__=='__main__': unittest.main()
