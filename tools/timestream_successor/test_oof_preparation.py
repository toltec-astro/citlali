from pathlib import Path
import copy,json
import numpy as np
import pytest
from unittest.mock import Mock
from observation_mode import validate_recipe
from oof_saved_interface import embed_altaz
from point_additive_diagnostics import save_json

D=Path(__file__).resolve().parents[2]/'validation/oof_preparation_2026-10-01/inputs'
def recipe():return json.loads((D/'reference_152385.json').read_text())
def test_explicit_oof_identity_and_no_point_policy():
    r=recipe();assert validate_recipe(r)
    for key,value in [('scan',2),('subobservation',1),('mode','POINT'),('observation',152420),('cycles',5),('source_gate','bright'),('threshold',3),('networks',[0,1,2,3,4]),('ptc',dict(method='observed-als',rank=10))]:
        q=copy.deepcopy(r);q[key]=value
        with pytest.raises(ValueError):validate_recipe(q)
def header():return dict(CTYPE1='AZOFFSET',CTYPE2='ELOFFSET',CUNIT1='arcsec',CUNIT2='arcsec',CDELT1=-2.,CDELT2=2.,CRPIX1=3.,CRPIX2=2.,CRVAL1=0.,CRVAL2=0.,BUNIT='mJy/beam')
def test_asymmetric_full_support_embedding_preserves_physical_axes_and_amplitude():
    z=np.arange(15.).reshape(3,5)-8;v=np.ones_like(z,bool);v[1,2]=False;z[2,4]=np.inf
    a,s,g=embed_altaz(z,v,header());c=len(a)//2
    assert s==2 and np.isfinite(a).sum()==13 and np.isnan(a[c,c])
    for j,i in zip(*np.nonzero(g)):assert a[c+j-1,c+2-i]==z[j,i]
@pytest.mark.parametrize('patch',[dict(CTYPE1='RA---TAN'),dict(CUNIT2='rad'),dict(CRPIX1=3.5),dict(CDELT2=1.),dict(BUNIT='arbs'),dict(PC1_2=.2)])
def test_unsupported_frame_units_center_fail_closed(patch):
    h=header();h.update(patch)
    with pytest.raises(ValueError):embed_altaz(np.ones((3,5)),np.ones((3,5),bool),h)
def test_cold_m0_stops_before_source_gate(tmp_path,monkeypatch):
    import point_processing as p
    (tmp_path/'freeze').mkdir();(tmp_path/'inputs').mkdir()
    save_json(tmp_path/'freeze/prospective.json',recipe());save_json(tmp_path/'freeze/budget.json',dict(shared_upstream_allowance=100))
    save_json(tmp_path/'inputs/observation.json',dict(networks=[]))
    for f in ['inputs/map-owner.json','M0/products/a1100/plan.yaml']:
        a=tmp_path/f;a.parent.mkdir(parents=True,exist_ok=True);a.write_text('{}')
    calls=[];monkeypatch.setattr(p,'call',lambda *a,**kw:calls.append(a[4]) or 0)
    monkeypatch.setattr(p,'initialize',Mock(side_effect=AssertionError('POINT source gate invoked')))
    p.upstream(tmp_path,Path('binary'))
    assert calls==['ptc','map'] and p.read(tmp_path/'UPSTREAM.json')['status']=='completed-cold-M0'
    assert not p.initialize.called


def test_oof_feedback_rejected_before_environment_or_processing(tmp_path,monkeypatch):
    import point_pipeline as p
    (tmp_path/'freeze').mkdir();save_json(tmp_path/'freeze/prospective.json',recipe())
    check=Mock(side_effect=AssertionError('execution reached'));monkeypatch.setattr(p,'validate_environment',check)
    with pytest.raises(ValueError,match='OOF.*forward'):p.execute('feedback',tmp_path,Path('binary'))
    assert not check.called and not (tmp_path/'EXECUTION.json').exists()


def test_joint_lm_sequence_rejects_unequal_support_without_cropping():
    from oof_saved_interface import require_lm_sequence
    records=[dict(shape=[512,512],angular_pixel_arcsec=1,wavelength_m=.0011,supported_pixels=n) for n in (89440,89377,88880)]
    original=copy.deepcopy(records)
    with pytest.raises(ValueError,match='unequal valid-pixel'):require_lm_sequence(records)
    assert records==original
    for r in records:r['supported_pixels']=10
    require_lm_sequence(records)
    records[0]['angular_pixel_arcsec']=2
    with pytest.raises(ValueError,match='scale/wavelength'):require_lm_sequence(records)
