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
