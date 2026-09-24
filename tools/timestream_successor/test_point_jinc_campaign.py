import numpy as np
from astropy.io import fits
from point_jinc_campaign import ordinary

def test_ordinary_same_order_and_rejoin_bound():
 h=fits.Header(dict(NAXIS1=5,NAXIS2=4));rng=np.random.default_rng(817)
 g=np.column_stack((rng.uniform(.5,5.5,500),rng.uniform(.5,4.5,500),np.ones(500),np.ones(500)))
 z=rng.normal(size=500);v=rng.normal(size=500)*100
 maps=[ordinary(g,t,h) for t in (z+v,z,v)];m,a,b=maps;use=m['SCI_SUPPORT'].astype(bool)
 assert m['HITS'].sum()==500
 delta=abs(m['PRIMARY']-a['PRIMARY']-b['PRIMARY']);bound=sum(x['ERROR'] for x in maps)+32*np.finfo(float).eps*sum(abs(x['PRIMARY']) for x in maps)
 assert np.all(delta[use]<=bound[use])

def test_ordinary_missing_support_and_constant():
 h=fits.Header(dict(NAXIS1=3,NAXIS2=3));g=np.array([[1.,1.,1.,1.],[2.,2.,1.,1.]])
 m=ordinary(g,np.array([17.,17.]),h);s=m['SCI_SUPPORT'].astype(bool)
 assert s.sum()==2 and np.isnan(m['PRIMARY'][~s]).all()
 np.testing.assert_array_equal(m['PRIMARY'][s],17.)

def test_failed_fit_withholds_background_dependent_measurements():
 from point_jinc_campaign import measure_full
 class Fixed:
  radius=np.zeros((41,41));major=3.
  def measure(self,s,v):return dict(fit=dict(status='unavailable'),frozen_template_projection=99.,apertures=[dict(value_mJy_per_beam_arcsec2=88.) for _ in range(3)])
 h=fits.Header(dict(NAXIS1=45,NAXIS2=45,CRPIX1=23.,CRPIX2=23.,CDELT1=-2/3600,CDELT2=2/3600));v=dict(PRIMARY=np.ones((45,45)),SCI_SUPPORT=np.ones((45,45),'u1'),HITS=np.ones((45,45)))
 m=measure_full(v,h,Fixed());assert m['frozen_template_projection'] is None
 assert all(x['value_mJy_per_beam_arcsec2'] is None for x in m['apertures'])
 assert all(x['available'] for x in m['raw_apertures'])

def test_response_full_domain_is_not_silently_shrunk():
 from point_jinc_science import response
 h=np.array([1.,2.,0.]);d=1.2*h;full=np.ones(3,bool);partial=np.array([True,True,False])
 assert response(d,h,full,partial)['status']=='unavailable'
 q=response(d,h,partial,partial);assert abs(q['g']-1.2)<1e-14 and abs(q['rho'])<1e-14
 assert response(d,h,np.zeros(3,bool),partial)['status']=='unavailable'
