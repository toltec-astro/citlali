import unittest
import numpy as np
from point_admission import admit
from point_fruit_diagnostics import replacement

class FruitReplacement(unittest.TestCase):
    def test_replacement_discards_previous_pixels(self):
        h=dict(CRPIX1=21.,CRPIX2=21.,CDELT1=-2/3600,CDELT2=2/3600)
        p=np.array([10.,0,0,2.,2.,0,0.]);s=np.zeros((41,41));s[20,20]=10;s[20,21]=5
        valid=np.ones_like(s,dtype=bool);scale=np.ones_like(s)
        old=admit(s,valid,scale,valid,h,p,3.,1.)
        s[20,21]=0
        new=admit(s,valid,scale,valid,h,p,3.,1.)
        self.assertTrue(old['selected'][20,21]);self.assertFalse(new['selected'][20,21]);self.assertEqual(new['candidate'][20,21],0)
        self.assertTrue(old['selected'][20,20] and new['selected'][20,20])

    def test_missing_fit_footprint_withholds_replacement(self):
        h=dict(CRPIX1=21.,CRPIX2=21.,CDELT1=-2/3600,CDELT2=2/3600)
        s=np.zeros((41,41));v=np.ones_like(s,dtype=bool);v[20,20]=False
        fit,candidate=replacement(s,v,np.ones_like(s),v,h,5)
        self.assertEqual(fit['status'],'unavailable');self.assertIsNone(candidate)

if __name__=='__main__':unittest.main()
