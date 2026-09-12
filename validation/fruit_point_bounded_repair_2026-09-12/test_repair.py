"""Focused tests of scientific failure modes, without saved trial data."""
import copy
import unittest
import numpy as np
import reevaluate
import stopping

class RepairTests(unittest.TestCase):
    def test_projected_gradient_constraints_and_finiteness(self):
        c = stopping.completion(np.array([0., 2.]), 3., np.array([4., .001]), 10.)
        self.assertTrue(c['finite'] and c['feasible'])
        self.assertEqual(c['relative_projected_gradient'], 1e-4)
        self.assertFalse(stopping.completion(np.array([-.01]), 0., np.array([0.]), 1.)['feasible'])
        self.assertFalse(stopping.completion(np.array([0.]), 0., np.array([np.nan]), 1.)['finite'])
        self.assertEqual(stopping.completion(np.array([0.]), 1., np.array([-2.]), 2.)['relative_projected_gradient'], 1.)

    def test_same_path_operational_prefix_and_tight_comparison(self):
        diagonal = np.geomspace(.03, 2., 15)
        target = np.linspace(-.5, 2., 15)
        def objective(v):
            diff = v-target
            return .5*float(np.sum(diagonal*diff**2)), diagonal*diff
        p = dict(objective=objective, zero=np.zeros(15), gradient_scale=max(abs(diagonal*target)))
        both = stopping.solve_path(p)
        operational = stopping.solve_path(p, qualify=False)
        np.testing.assert_array_equal(both['snapshots']['declared']['v'], operational['snapshots']['declared']['v'])
        self.assertTrue(both['operational_stop_available'])
        self.assertLessEqual(both['snapshots']['tight']['relative_projected_gradient'], 1e-6)
        np.testing.assert_allclose(both['snapshots']['tight']['v'], np.maximum(target, 0), atol=1e-4)

    def test_truth_does_not_change_judgments_and_shape_is_separate(self):
        original = dict(source_present=True, fit=dict(parameters=[1.]*9, peak=10., widths=[12.,8.], boundary_rejection=False),
                        shape_core_pixels=20, boundary_warning=False, distortion_warning=True, empirical_peak_score=4.)
        probe = dict(available=True, centroid_difference_arcsec=.1, peak_relative_difference=.001)
        first = reevaluate.judgments(original, True, probe)
        second_input = copy.deepcopy(original)
        second_input.update(injected_centroid=[999., -999.], centroid_error_arcsec=10000., arm='C')
        self.assertEqual(first, reevaluate.judgments(second_input, True, probe))
        self.assertTrue(first['centroid_usable'] and first['shape_warning'])
        self.assertFalse(first['peak_response_usable'])
        for key in ['boundary_warning']:
            second_input[key] = True
            self.assertFalse(reevaluate.judgments(second_input, True, probe)['centroid_usable'])
        self.assertFalse(reevaluate.judgments(original, False, probe)['centroid_usable'])

if __name__ == '__main__':
    unittest.main()
