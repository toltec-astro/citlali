import unittest
import numpy as np
import experiment_r02 as e
from test_experiment import Invariants
import test_experiment

class Revision(Invariants):
    def setUp(self):
        self.old=test_experiment.e;test_experiment.e=e
    def tearDown(self):test_experiment.e=self.old
    def test_coherence_start_ignores_isolated_largest_pixel(self):
        yy,xx=np.mgrid[-90:91:2,-90:91:2];x=xx.ravel();y=yy.ravel();D=x*x+y*y<=90**2
        p=np.array([100.,17.,-11.,np.log(12.),np.log(8.),.4,0,0,0]);z=e.gaussian(p,x,y)
        i=np.argmin((x+78)**2+(y+18)**2);z[i]=200.
        f=e.fit_source(z,x,y,D)
        np.testing.assert_allclose(f['centroid'],[17,-11],atol=.1)
        self.assertFalse(f['boundary_rejection'])
if __name__=='__main__':unittest.main()
