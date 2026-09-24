import numpy as np
import pytest
from types import SimpleNamespace
from point_corr_admission import Neighborhood,select,threshold,binomial_upper,trial_seed,base_from_fit
from point_admission import admit
from point_additive import draw_noise
from scipy.ndimage import label


def fixture(beam=5,correlated=False):
 shape=(41,41);h=dict(CRPIX1=21.,CRPIX2=21.,CDELT1=-2/3600,CDELT2=2/3600)
 scale=np.ones(shape);k=np.zeros(shape,bool);k[20,20]=True
 psd=np.ones((100,100))
 if correlated:
  rho=np.zeros((100,100));rho[0,0]=1;rho[0,1]=rho[0,-1]=.2;psd=np.fft.fft2(rho).real
 n=Neighborhood(h,shape,beam,scale,psd,k);data=np.ones(shape)*.7;data[20,20]=10
 f=SimpleNamespace(header=h,beam=beam,scale=scale,noise_valid=np.ones(shape,bool))
 fit=dict(status='success',parameters=[10,0,0,2,2,0,0],constrained=False)
 b=base_from_fit(data,np.ones(shape,bool),f,fit)
 return n,data,f,fit,b,psd


def test_exclusion_negative_neighbors_and_same_brightness():
 n,data,f,fit,b,psd=fixture();fields=n.scores(b)
 # Anchor east of core: center and core changes cannot affect its evidence
 anchor=(20,22);b2={**b,'brightness':b['brightness'].copy()};b2['brightness'][20,20]=1e12;b2['brightness'][anchor]=1e12
 assert n.scores(b2)['evidence'][anchor]==fields['evidence'][anchor]
 b2['brightness'][20,23]=-10
 assert n.scores(b2)['evidence'][anchor]<fields['evidence'][anchor]
 c=select(b,fields,n.k0,'CORR',-1)
 assert np.all(c['selected'][b['selected']])
 assert np.array_equal(c['candidate'][c['selected']],b['brightness'][c['selected']])
 assert np.all(c['candidate'][c['valid']&~c['selected']]==0)
 assert np.isnan(c['candidate'][~c['valid']]).all()


def test_current_B_excluded_and_disconnected_no_growth():
 n,data,f,fit,b,psd=fixture();b['selected'][20,21]=True
 fields=n.scores(b);other={**b,'brightness':b['brightness'].copy()};other['brightness'][20,21]=1e9
 assert n.scores(other)['evidence'][20,22]==fields['evidence'][20,22]
 for pol in ('CORR','DIAG'):
  fields['policies'][pol]['score'][:]=-100;fields['policies'][pol]['score'][10,10]=100
 c=select(b,fields,n.k0,'CORR',3);assert not c['added'].any()
 assert c['proposed_extra'][10,10]


def test_full_footprint_missing_and_geometry():
 n,data,f,fit,b,psd=fixture(7.5)
 assert len(n.offsets)==45
 assert (n.offsets==0).all(axis=1).sum()==1
 fields=n.scores(b);assert not fields['policies']['CORR']['available'][0,20]
 before=fields['policies']['CORR']['available'][20,25];assert before
 b['valid'][20,26]=False
 assert not n.scores(b)['policies']['CORR']['available'][20,25]
 n.scale[20,26]=np.nan
 assert not n.scores(b)['policies']['CORR']['available'][20,25]


def test_diagonal_identity_and_direct_covariance():
 for correlated in (False,True):
  n,data,f,fit,b,psd=fixture(correlated=correlated);fields=n.scores(b);r,c=20,24;i=np.flatnonzero(n.anchor_indices==r*41+c)[0]
  w=n.weights*(~(n.k0|b['selected']).ravel()[n.indices[i]]);w[n.center]=0;q=w/w.sum()
  dense=sum(q[j]*q[k]*n.rho[j,k] for j in range(len(q)) for k in range(len(q)))
  assert fields['policies']['CORR']['variance'][r,c]==pytest.approx(dense,rel=1e-14)
  if not correlated:
   assert np.allclose(fields['policies']['CORR']['score'],fields['policies']['DIAG']['score'],equal_nan=True)
   assert np.array_equal(select(b,fields,n.k0,'CORR',3)['selected'],select(b,fields,n.k0,'DIAG',3)['selected'])
  weights=np.zeros((41,41));weights.ravel()[n.indices[i]]=q
  values=np.array([np.sum(weights*draw_noise(psd,np.ones((41,41)),(919,j))) for j in range(1024)])
  assert abs(values.var(ddof=1)/dense-1)<.15
  assert np.array_equal(draw_noise(psd,n.scale,123),draw_noise(psd,n.scale,123))


def test_concentration_empty_source_and_constrained():
 n,data,f,fit,b,psd=fixture();n.k0[:]=True;n.k0[20,24:26]=False
 fields=n.scores(b);assert not fields['policies']['CORR']['available'].any()
 b['selected'][:]=False;assert not select(b,fields,n.k0,'CORR',-1)['selected'].any()
 assert base_from_fit(data,np.ones((41,41),bool),f,{**fit,'constrained':True}) is None
 assert base_from_fit(data,np.ones((41,41),bool),f,dict(status='failed')) is None
 assert select(None,None,n.k0,'CORR',3) is None


def test_order_statistic_strict_and_seeds():
 assert threshold(list(range(1024)))==1014
 assert threshold([None]*1024)==3
 assert threshold([5]*1024)==5
 n,data,f,fit,b,psd=fixture();fields=n.scores(b)
 fields['policies']['CORR']['score'][:]=5
 assert not select(b,fields,n.k0,'CORR',5)['added'].any()
 seeds=[trial_seed(a,p,i) for a in ('a1100','a1400','a2000') for p in ('calibration','validation') for i in range(1024)]
 assert len(set(seeds))==6144
 import hashlib
 assert seeds[0]==int.from_bytes(hashlib.sha256(b'POINT152389-CORR-NEIGHBOR-v1|a1100|calibration|0000').digest()[:16],'big')
 assert binomial_upper(0)==pytest.approx(1-.05**(1/1024),rel=1e-12)
 assert binomial_upper(1024)==1
 assert binomial_upper(10)<.02<binomial_upper(20)


def test_nonzero_background_operand():
 n,data,f,fit,b,psd=fixture();offset=137.
 shifted=base_from_fit(data+offset,np.ones((41,41),bool),f,{**fit,'parameters':[10,0,0,2,2,0,offset]})
 assert np.array_equal(shifted['selected'],b['selected'])
 assert np.allclose(n.scores(shifted)['evidence'],n.scores(b)['evidence'],equal_nan=True)
