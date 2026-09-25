"""Deterministic signed-response and missing-domain checks; no new random fields."""
import numpy as np
from report_point_guarded_feedback import metrics
from report_point_guarded_feedback import publication_directory
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

def test_publication_recovery_preserves_fit_budget_and_nonoverwriting(tmp_path):
    out=tmp_path/'report';d=out/'a1100/BASE-POINT/endpoint';d.mkdir(parents=True)
    for name in ('learned.json','learned.npz','FIT_ATTEMPT.json'):(d/name).write_bytes(name.encode())
    (d/'RESULTS.json').write_text('failed publication');(d/'partial.fits').write_bytes(b'partial')
    ideal=out/'a1100/BASE-POINT/C-core-plus/ideal-measurement.json';ideal.parent.mkdir();ideal.write_text('saved fit')
    recovered=publication_directory(tmp_path)
    assert (recovered/'a1100/BASE-POINT/endpoint/learned.npz').read_bytes()==b'learned.npz'
    assert (recovered/'a1100/BASE-POINT/C-core-plus/ideal-measurement.json').read_text()=='saved fit'
    assert not (d/'partial.fits').exists() and not (d/'RESULTS.json').exists()
    assert (tmp_path/'provenance/report-publication-attempt-1/a1100/BASE-POINT/endpoint/partial.fits').read_bytes()==b'partial'
