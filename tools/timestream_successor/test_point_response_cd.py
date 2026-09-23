"""Independent numerical controls for the bounded C+D companion."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from point_response_cd import (shapes, seed_candidate, response_metrics, direct_raster,
                               ellipse, processing_guard)
from run_point_response_cd import campaign_request, BRANCHES
from point_fruit_trajectory import candidate_guard
from point_empirical import coordinates
from point_admission import admit
from point_empirical_io import write_fits
from point_response_cd import injection_reference
from inspect_forward_map import digest
import yaml

H=dict(CRPIX1=21.,CRPIX2=21.,CDELT1=-2/3600,CDELT2=2/3600)

class ResponseControls(unittest.TestCase):
    def setUp(self):
        self.x,self.y=coordinates((41,41),H);self.v=np.hypot(self.x,self.y)<=40
        self.p=np.array([100.,3.,-4.,2.,3.,.4,7.])

    def test_shapes_signed_taper_physical_east_and_unavailable(self):
        r,info=shapes(self.p,7.,H,self.v)
        np.testing.assert_array_equal(r['C-core-plus'],-r['C-core-minus'])
        self.assertEqual(info['C-shoulder']['parameters'][1],6.5)
        self.assertEqual(info['C-shoulder']['parameters'][3:5],[4.,6.])
        for values in r.values():
            self.assertTrue(np.all(values[self.v & (np.hypot(self.x,self.y)>=36)]==0))
        v=self.v.copy();v[20,20]=False
        with self.assertRaisesRegex(ValueError,'unavailable'):shapes(self.p,7.,H,v)
        self.assertGreater(info['C-shoulder']['taper_loss_area_sum_arcsec2'],0)

    def test_signed_noop_and_finite_attenuating_recurrence(self):
        # Independent closed form: T0=a*y, T_k=a*y+(1-a)T_(k-1).
        # Thus response at k is [1-(1-a)^(k+1)] I, approaching unity;
        # first-pass attenuation is not an invariant of feedback.
        y=np.array([9.,10.,11.]);inc=np.array([.3,1.,.2])
        for alpha in (1.,.35):
            for sign in (0,1,-1):
                b=y+sign*inc;m=alpha*y;n=alpha*b
                for k in range(11):
                    np.testing.assert_allclose(n-m,(1-(1-alpha)**(k+1))*sign*inc,atol=1e-13)
                    if sign:
                        q=response_metrics(n-m,sign*inc,np.ones(3,bool))
                        self.assertAlmostEqual(q['g'],1-(1-alpha)**(k+1))
                    m=alpha*(y-m)+m;n=alpha*(b-n)+n

    def test_seed_only_feature_removed_by_replacement(self):
        # A feature excluded by ordinary replacement has no permanent exemption.
        y=np.array([10.,0.,0.]);seed=np.array([10.,.2,.1]);alpha=.5
        total=alpha*(y-seed)+seed
        self.assertGreater(total[1],0)
        replacement=np.where(total>=1,total,0)
        total2=alpha*(y-replacement)+replacement
        np.testing.assert_array_equal(total2[1:],[0,0])

    def test_seed_low_high_support_and_shoulder_exception(self):
        original=dict(CANDIDATE=np.where(self.v,np.where(np.hypot(self.x,self.y)<5,10.,0.),np.nan),
            VALID=self.v.astype('u1'),ADMITTED=(np.hypot(self.x,self.y)<5).astype('u1'),RATIO=np.zeros(self.v.shape))
        r,_=shapes(self.p,7,H,self.v)
        for b,factor in [('D-seed-low',.8),('D-seed-high',1.2)]:
            c=seed_candidate(original,b,r['C-shoulder']);candidate_guard(c,H)
            np.testing.assert_array_equal(c['ADMITTED'],original['ADMITTED'])
            np.testing.assert_array_equal(c['CANDIDATE'],original['CANDIDATE']*factor)
        shoulder=seed_candidate(original,'D-seed-shoulder',r['C-shoulder']);candidate_guard(shoulder,H)
        self.assertGreater(shoulder['ADMITTED'].sum(),original['ADMITTED'].sum())
        np.testing.assert_array_equal(original['ADMITTED'],np.hypot(self.x,self.y)<5)

    def test_projection_half_ties_and_support(self):
        # Pixel p at east0 contains [-1,+1) in decreasing-column coordinates:
        # +1 east lies on lower-index tie, assigned index20; -1 goes index21.
        raster=np.zeros((41,41));raster[20,20]=5;raster[20,21]=9
        for east,value in [(0,5),(.999,5),(-1,9),(1,5),(1.001,0)]:
            oh=dict(H,CRPIX1=1-east/(H['CDELT1']*3600),CRPIX2=1.)
            self.assertEqual(direct_raster(raster,H,(1,1),oh)[0,0],value)
        bad=raster.copy();bad[20,20]=np.nan
        with self.assertRaises(ValueError):direct_raster(bad,H,(41,41),H)

    def test_gauge_safe_ellipse(self):
        p=self.p.copy();q=p.copy();q[3:5]=p[[4,3]];q[5]+=np.pi/2
        np.testing.assert_allclose(ellipse(p),ellipse(q),atol=1e-14)

    def test_request_bounds_and_injection_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_text('bound')
            owner=dict(parents={'a1100':dict(map_request={},parent_plan={})},injections={'a1100':{'C-core-plus':{'sha256':'I'}}})
            for branch in BRANCHES:
                if branch.startswith('C-') and branch!='C-core-plus':continue
                r=campaign_request(p,owner,'a1100',branch,1,p,p)
                self.assertEqual('injection' in r,branch.startswith('C-'))
            for k in (-1,11):
                with self.assertRaises(ValueError):campaign_request(p,owner,'a1100','REF',k,p,p)
            with self.assertRaises(ValueError):campaign_request(p,owner,'a1100','D-seed-low',0,p)

    def test_stopped_fit_without_candidate_keeps_other_assessment_available(self):
        from report_point_response_cd import candidate_comparison
        failed=dict(measurement={'fit':{'status':'failed'}},total={'path':'preserved-total'},stop_reason='unusable fit/candidate')
        result=candidate_comparison(failed,{},np.ones((2,2)))
        self.assertEqual(result['candidate_status'],'unavailable')
        self.assertIsNone(result['replacement_shoulder_projection'])
        # Candidate unavailability does not prevent independent total-map response.
        response=response_metrics(np.ones((2,2)),np.ones((2,2)),np.ones((2,2),bool))
        self.assertEqual(response['g'],1)
        self.assertEqual(failed['total']['path'],'preserved-total')

    def test_shoulder_projection_with_supported_stamp_corners(self):
        from point_response_cd import shape_projection
        shape=np.where(self.v,np.exp(-(self.x**2+self.y**2)/100),np.nan)
        signal=np.nan_to_num(shape)
        p=shape_projection(signal,np.ones(shape.shape,bool),shape)
        self.assertAlmostEqual(p['value'],1)
        self.assertEqual(p['pixels'],self.v.sum())

    def test_ideal_map_addition_separates_estimator_response(self):
        from point_response_cd import measurement_response
        from point_fruit_trajectory import FixedMeasurements
        from inspect_point_source import gaussian
        fixed=object.__new__(FixedMeasurements)
        from astropy.io import fits
        fixed.header=fits.Header(dict(H,CRVAL1=100.,CRVAL2=20.,CTYPE1='RA---TAN',CTYPE2='DEC--TAN',RADESYS='FK5',EQUINOX=2000.))
        fixed.beam=7.5;fixed.x=self.x;fixed.y=self.y;fixed.disk=self.v;fixed.parameters=self.p
        fixed.radius=np.hypot(self.x-self.p[1],self.y-self.p[2]);fixed.major=7.
        fixed.background=self.v&(fixed.radius>21);fixed.edges=np.arange(0,42,2)
        template=gaussian([1,*self.p[1:6],0],self.x,self.y);template[~self.v]=0
        fixed.weights=template/np.sum(template**2)
        ref=gaussian(self.p,self.x,self.y);rasters,_=shapes(self.p,7,fixed.header,self.v)
        known=np.nan_to_num(rasters['C-shoulder']);result=measurement_response(fixed,ref,ref+known,self.v,known)
        for row in result['comparisons']:self.assertEqual(row['absolute_discrepancy'],0)
        raw=result['raw_aperture_and_background_increments'][-1]
        self.assertAlmostEqual(raw['actual_raw_increment'],raw['ideal_raw_increment'],places=10)
        zero=measurement_response(fixed,ref,ref,self.v,np.zeros(ref.shape))
        self.assertTrue(all(r['ratio'] is None for r in zero['comparisons']))

    def test_union_alignment_preserves_unavailable_support(self):
        from report_point_response_cd import aligned
        ha=dict(H,CRVAL1=100.,CRVAL2=20.,RADESYS='FK5',EQUINOX=2000.,CRPIX1=2.,CRPIX2=2.)
        hb=dict(ha,CRPIX1=3.,CRPIX2=3.)
        a,b,h,place=aligned(np.ones((3,3)),ha,np.ones((5,5))*2,hb)
        self.assertEqual(a.shape,(5,5));self.assertEqual(np.isfinite(a).sum(),9)
        self.assertTrue(np.isnan(a[0,0]));self.assertEqual(b[0,0],2)

    def test_fixed_ref_operator_normalization_missing_and_excluded_entries(self):
        # Three eligible samples, only two MAP contributors. Third injection
        # must remain visible in the pre-PTC audit. Counts2 give the mean, not sum.
        from astropy.io import fits
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ref=root/'ref';inj=root/'inj';source=root/'source'
            for p in (ref,inj,source):p.mkdir()
            name='network-0-segment-0';(ref/name).mkdir();(inj/name).mkdir()
            h=fits.Header(H);raster=np.zeros((41,41));raster[20,20]=7
            images=[('SIGNAL',np.zeros((41,41)),'x'),('SCI_SUPPORT',np.eye(41,dtype='u1')*0,'1'),
                ('HITS',np.zeros((41,41),np.int64),'1'),('NORMALIZATION',np.zeros((41,41)),'1')]
            images[1][1][20,20]=1;images[2][1][20,20]=2;images[3][1][20,20]=2
            write_fits(ref/'total.fits',h,images)
            # load() uses PRIMARY for native products; omit custom primary EXTNAME.
            with fits.open(ref/'total.fits',mode='update') as f:del f[0].header['EXTNAME']
            np.array([840,840],dtype='<i4').tofile(ref/'operator-pixel.i32')
            np.array([0,0,1],dtype='u1').tofile(ref/name/'causes.u8')
            np.array([1,1,1],dtype='u1').tofile(source/'segment-0-eligible.u8')
            np.array([7,7,7],dtype='<f8').tofile(inj/name/'injection.f64')
            s=dict(network=0,files={'segment-0-input.f64':'y'},scheduled_times=1,detectors=3,output_detectors=3,fit_columns_in_output=[0,1,2])
            rs=dict(source_segment=s,source_root=str(source),retained=2,MAP_first_input_row=0)
            ins=dict(source_segment=s,files={'injection.f64':digest(inj/name/'injection.f64')})
            (ref/'receipt.yaml').write_text(yaml.safe_dump(dict(segments=[rs])))
            (inj/'receipt.yaml').write_text(yaml.safe_dump(dict(segments=[ins])))
            image,valid,_,report=injection_reference(ref,inj,raster,h)
            self.assertEqual(image[20,20],7);self.assertEqual(report['non_MAP_nonzero_injected_entries'],1)
            self.assertEqual(valid.sum(),1);self.assertTrue(np.isnan(image[0,0]))

if __name__=='__main__':unittest.main()
