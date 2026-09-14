import unittest
import numpy as np
from analyze_rtc_real_data_audit import (independent_windows, cover_scans, incremental_cost,
                                       noise_factor, row_support, describe_coordinate)


class AuditTests(unittest.TestCase):
    def test_incremental_union_does_not_double_count_pair_or_baseline(self):
        self.assertEqual(incremental_cost([(0,100)],[(10,30)],[(20,50),(40,60)]),30)

    def test_gaps_and_invalid_support_do_not_enter_denominator(self):
        self.assertEqual(incremental_cost([(0,10),(90,100)],[],[(0,100)]),20)

    def test_scan_boundary_equality_is_half_open(self):
        self.assertEqual(cover_scans([(0,10)],10),[(0,10)])
        self.assertEqual(cover_scans([(9,11)],10),[(0,20)])
        self.assertEqual(cover_scans([(-1,1)],10),[(-10,10)])
        self.assertEqual(cover_scans([(0,10)],10,5),[(-5,15)])

    def test_independent_windows_exclude_end_anchor_overlap(self):
        w=np.zeros((5,14));w[:,2:4]=[(0,488),(244,732),(488,976),(732,1220),(734,1222)]
        self.assertEqual(independent_windows(w).tolist(),[0,2])

    def test_native_support_preserves_gap(self):
        cells=np.array([[0,0,-5,5],[1,10,5,15],[2,30,25,35]])
        self.assertEqual(row_support([(0,3)],cells),[(-5,15),(25,35)])
        with self.assertRaises(ValueError):row_support([(0,4)],cells)

    def test_fixed_weight_sensitivity_scaling(self):
        self.assertAlmostEqual(noise_factor(100,25),np.sqrt(4/3))
        self.assertIsNone(noise_factor(0,0));self.assertIsNone(noise_factor(100,100))

    def test_low_frequency_power_never_becomes_narrow_class(self):
        p=np.array([100,1,1,1,1.]);meta=dict(cause=0,regions_1=[],regions_2=[[0,2,0,100,.95,100,1,1]],regions_4=[])
        d,active=describe_coordinate(meta,np.array([p]*4),np.empty((0,14)),np.arange(5.))
        self.assertEqual(d['narrow_fraction'],0);self.assertEqual(active[.1],[])
        self.assertGreater(d['low_frequency_fraction'],.9)

    def test_descriptor_target_mismatch_is_unavailable_not_fake_recurrence(self):
        meta=dict(cause=0,regions_1=[],regions_2=[[3,4,3,10,.2,20,0,0],[5,8,6,20,.4,2,0,0]],regions_4=[])
        d,active=describe_coordinate(meta,np.array([[1.]*10]*4),np.empty((0,14)),np.arange(10.))
        self.assertEqual(d['narrow_fraction'],.2);self.assertIsNone(d['recurrence'])
        self.assertEqual(active[.1],[]);self.assertIn('recurrence_unavailable_reason',d)


if __name__=='__main__':unittest.main()
