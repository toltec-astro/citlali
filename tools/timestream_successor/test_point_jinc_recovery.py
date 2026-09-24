import unittest
import numpy as np
from recover_point_jinc_inputs import rejoined_rows,MATRIX
class SavedJincRecovery(unittest.TestCase):
    def test_full_output_order_missing_and_exact_operand(self):
        clean=np.array([[1.,99.,2.],[3.,88.,4.]])
        causes=np.array([[0,17,0],[0,17,5]])
        operand=np.array([[10.,20.],[30.,40.]])
        z,v=rejoined_rows(clean,operand,causes,[2,0])
        np.testing.assert_array_equal(z,[1,2,3]);np.testing.assert_array_equal(v,[20,10,40])
        np.testing.assert_array_equal(z+v,[21,12,43])
        with self.assertRaises(ValueError):rejoined_rows(clean,operand,causes,[0,0])
        clean[0,0]=np.nan
        with self.assertRaises(ValueError):rejoined_rows(clean,operand,causes,[2,0])
    def test_exact_bounded_matrix(self):
        self.assertEqual(MATRIX,(('REF',0),('REF',10),('C-core-plus',10),('C-core-minus',10),('C-shoulder',10)))
