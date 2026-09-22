"""Known source with holes checks measurement signs, widths and offset convention."""
import unittest
import numpy as np
from astropy.io import fits
from inspect_point_source import measure


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

    def test_missing_central_support_is_not_a_source(self):
        h={'CRPIX1':4,'CRPIX2':4,'CDELT1':-2/3600,'CDELT2':2/3600}
        with self.assertRaisesRegex(ValueError,'insufficient'):
            measure(np.zeros((7,7)),np.zeros((7,7),dtype=bool),h,5)

if __name__=='__main__':unittest.main()
