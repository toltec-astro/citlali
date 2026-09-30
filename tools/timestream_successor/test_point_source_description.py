"""Known source with holes checks measurement signs, widths and offset convention."""
import unittest
import numpy as np
from astropy.io import fits
from inspect_point_source import measure, solve_gaussian, gaussian, gaussian_jacobian


class PointSourceDescription(unittest.TestCase):
    def test_offset_ellipse_is_measured_with_holes_and_signed_background(self):
        yy,xx=np.indices((65,65));x=(33-xx-1)*2.;y=(yy+1-33)*2.
        # Independent axis-aligned source: north FWHM 10, east FWHM 6.
        signal=-7+125*np.exp(-4*np.log(2)*(((x-4)/6)**2+((y+6)/10)**2))
        support=np.ones(signal.shape,dtype=bool);support[::7,::5]=False;signal[~support]=np.nan
        header=fits.Header(dict(CRPIX1=33.,CRPIX2=33.,CDELT1=-2/3600,CDELT2=2/3600,
            CRVAL1=180.,CRVAL2=40.,CTYPE1='RA---TAN',CTYPE2='DEC--TAN',RADESYS='FK5',EQUINOX=2000.))
        r,_,_,_=measure(signal,support,header,7.5)
        np.testing.assert_allclose(r['centroid_tangent_east_north_arcsec'],[4,-6],atol=1e-7)
        np.testing.assert_allclose(r['measured_fwhm_major_minor_arcsec'],[10,6],atol=1e-7)
        self.assertAlmostEqual(r['fitted_amplitude_above_constant_mJy_per_nominal_beam'],125,places=7)
        self.assertAlmostEqual(r['fitted_constant_mJy_per_nominal_beam'],-7,places=7)
        self.assertFalse(r['fit_constraint_active'])

    def test_shared_analytic_jacobian_matches_finite_differences(self):
        y,x=np.mgrid[-8:9:2.,-8:9:2.];x=x.ravel();y=y.ravel()
        p=np.array([170.,2.,-3.,4.1,2.3,.71,-6.])
        numerical=[]
        for i in range(7):
            h=1e-5;plus=p.copy();minus=p.copy();plus[i]+=h;minus[i]-=h
            numerical.append((gaussian(plus,x,y)-gaussian(minus,x,y))/(2*h))
        np.testing.assert_allclose(gaussian_jacobian(p,x,y),np.array(numerical).T,rtol=1e-6,atol=1e-7)
        p[4]=p[3]
        np.testing.assert_array_equal(gaussian_jacobian(p,x,y)[:,5],np.zeros_like(x))

    def test_periodic_and_axis_swapped_initializations_have_same_image_objective(self):
        y,x=np.mgrid[-40:41:2.,-40:41:2.];roi=np.hypot(x,y)<=40
        truth=np.array([210.,4.,7.,4.6,2.1,-.25,-2.])
        signal=gaussian(truth,x,y)
        initial=np.array([180.,3.,6.,5.,2.5,-.4,-1.])
        periodic=initial.copy();periodic[5]+=np.pi
        swapped=initial.copy();swapped[3:5]=initial[[4,3]];swapped[5]+=np.pi/2
        outputs=[solve_gaussian(signal,roi,x,y,p,sigma_floor=1.5) for p in (initial,periodic,swapped)]
        for f in outputs:
            self.assertTrue(f.success);self.assertEqual(f.active_mask[5],0)
            np.testing.assert_allclose(gaussian(f.x,x,y),signal,atol=1e-7)
            self.assertLess(f.cost,1e-12)
        np.testing.assert_allclose(outputs[0].fun,outputs[1].fun,atol=1e-7)
        np.testing.assert_allclose(outputs[0].fun,outputs[2].fun,atol=1e-7)

    def test_angle_can_cross_old_boundary_without_restarting(self):
        y,x=np.mgrid[-40:41:2.,-40:41:2.];roi=np.hypot(x,y)<=40
        truth=np.array([230.,4.7,7.5,4.6,2.1,-.25,-2.])
        initial=truth.copy();initial[5]=-np.pi
        f=solve_gaussian(gaussian(truth,x,y),roi,x,y,initial,sigma_floor=1.5)
        self.assertTrue(f.success);self.assertEqual(f.active_mask[5],0)
        self.assertLess(f.cost,1e-12)
        np.testing.assert_allclose(gaussian(f.x,x,y),gaussian(truth,x,y),atol=1e-7)

    def test_near_circular_image_without_identified_angle(self):
        y,x=np.mgrid[-40:41:2.,-40:41:2.];roi=np.hypot(x,y)<=40
        for epsilon in (0.,1e-7):
            truth=np.array([170.,2.,-3.,3.,3.+epsilon,.71,-6.])
            initial=np.array([150.,1.,-2.,3.1,3.,-.62,-5.])
            f=solve_gaussian(gaussian(truth,x,y),roi,x,y,initial,sigma_floor=1.5)
            self.assertTrue(f.success);self.assertLess(f.cost,1e-12)
            np.testing.assert_allclose(gaussian(f.x,x,y),gaussian(truth,x,y),atol=1e-7)
            # No assertion on angle: it has no unique value for a circular source.

    def test_missing_central_support_is_not_a_source(self):
        h={'CRPIX1':4,'CRPIX2':4,'CDELT1':-2/3600,'CDELT2':2/3600}
        with self.assertRaisesRegex(ValueError,'insufficient'):
            measure(np.zeros((7,7)),np.zeros((7,7),dtype=bool),h,5)

if __name__=='__main__':unittest.main()
