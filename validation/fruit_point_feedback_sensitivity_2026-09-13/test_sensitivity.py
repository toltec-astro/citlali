"""Tests of state selection and sign/rotation-invariant subspace comparison."""
import unittest
import numpy as np
from diagnostics import subspace_distance,select_states
from common import b,candidate

class SensitivityTests(unittest.TestCase):
    def test_subspace_invariant_to_signs_and_rotations(self):
        A=np.eye(8)[:,:5]
        q,_=np.linalg.qr(np.arange(25).reshape(5,5)+np.eye(5))
        result=subspace_distance(A,b.mm(A,q))
        self.assertLess(result['normalized_projector_distance'],1e-7)
        B=A.copy();B[:,4]=np.eye(8)[:,7]
        result=subspace_distance(A,B)
        self.assertAlmostEqual(result['maximum_principal_angle_degrees'],90.)
        self.assertAlmostEqual(result['normalized_projector_distance'],np.sqrt(1/5))

    def test_selection_uses_only_registered_projected_core_metric(self):
        rows=[]
        for case,k,score,complete in [('H_20260911',0,1000,True),('D_20260912',0,9,True),
            ('H-shift_20260911',0,9,True),('T_20260911',1,999,False),('real123424',0,2000,True)]:
            for a in range(3):
                rows.append(dict(case=case,pass_index=k,available=complete or a<2,
                    projection=dict(source_core_difference_energy=score if a==1 else 0),
                    injected_centroid=[500,-500],next_map_error=1e6))
        selected,_=select_states(rows)
        self.assertEqual([(r['case'],r['pass_index']) for r in selected],
            [('H_20260911',0),('D_20260912',0),('real123424',0)])

    def test_feedback_optimization_and_inference_disabled(self):
        with self.assertRaises(RuntimeError):candidate.minimize(None)
        with self.assertRaises(RuntimeError):b.inference(None)

if __name__=='__main__':
    unittest.main()
