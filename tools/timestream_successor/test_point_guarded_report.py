"""Deterministic signed-response and missing-domain checks; no new random fields."""
import numpy as np
from report_point_guarded_feedback import metrics
from report_point_response_cd import aligned

def test_signed_response_preserves_zero_truth_background():
    known=np.array([[1.,-1.],[0.,0.]])
    residual=np.array([[0.,0.],[1.,-1.]])
    delta=2*known+residual
    q=metrics(delta,known,np.ones((2,2),bool))
    assert q['g']==2 and np.isclose(q['eta'],np.sqrt(2)) and q['rho']==1
    assert metrics(delta,known,np.array([[False,False],[True,True]])) is None

def test_archived_extent_is_not_silently_cropped():
    h=dict(CDELT1=-2/3600,CDELT2=2/3600,CRVAL1=0,CRVAL2=0,RADESYS='FK5',EQUINOX=2000.,CRPIX1=2,CRPIX2=2)
    big=h|dict(CRPIX1=3,CRPIX2=3)
    now,old,union,place=aligned(np.ones((3,3)),h,np.ones((5,5)),big)
    assert now.shape==(5,5) and np.isnan(now).sum()==16 and np.isfinite(old).sum()==25
    benchmark=old==1;native=np.isfinite(now)
    assert benchmark.sum()==25 and (benchmark&native).sum()==9
