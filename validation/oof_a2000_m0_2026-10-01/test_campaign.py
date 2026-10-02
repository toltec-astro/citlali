"""Campaign policy fixtures; subprocesses are stubs, never observation launches."""
import importlib.util,json,sys
from pathlib import Path
from unittest.mock import Mock
import pytest
P=Path(__file__).with_name('run_campaign.py');spec=importlib.util.spec_from_file_location('oof_campaign',P);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
def test_attempts_are_unique_and_R_cannot_consume_S_allowance(tmp_path):
    for name in c.ROLES:
        c.consume(tmp_path,'parent',dict(run=name));c.consume(tmp_path,'trajectory',dict(run=name,branch='REF',cycle=0))
    assert len((tmp_path/'attempts.jsonl').read_text().splitlines())==6
    for kind,identity in [('parent',dict(run=c.ROLES[0])),('parent',dict(run='R-152385')),('feedback',dict(run=c.ROLES[0])),('trajectory',dict(run=c.ROLES[0],branch='REF',cycle=1))]:
        with pytest.raises(ValueError):c.consume(tmp_path,kind,identity)
def test_storage_counts_prior_and_reserves_three_serial_S(tmp_path):
    p=tmp_path/'prior';p.mkdir();(p/'x').write_bytes(b'x'*20);root=tmp_path/'root';root.mkdir()
    x=c.capacity(7704622,root,[p]);assert x['peak_bound_bytes']==20+2*c.ROLE_CAP+304*7704622+2700000000+c.META
    assert x['final_bound_bytes']==20+3*c.ROLE_CAP+c.META

def test_failure_stops_remaining_roles(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'prelaunch',lambda r:('abc',{'storage':{},'binary':{'sha256':'hash'}},{}))
    monkeypatch.setattr(c.subprocess,'check_output',lambda args,**kw:'abc' if args[1]=='rev-parse' else '')
    monkeypatch.setattr(c,'digest',lambda p:'hash');monkeypatch.setattr(c,'bound',lambda p:dict(path=str(p),sha256='hash'))
    prepared=[];monkeypatch.setattr(c,'prepare_role',lambda name,f:prepared.append(name) or tmp_path/name)
    monkeypatch.setattr(c,'launch',lambda *a,**kw:dict(status='failed'))
    import psutil;monkeypatch.setattr(psutil,'Process',lambda:Mock(children=lambda **kw:[]))
    monkeypatch.setattr(sys,'argv',[str(P),'--review',str(tmp_path/'review.json')])
    assert c.main()==1 and prepared==[c.ROLES[0]]
    assert json.loads((tmp_path/'STATUS.json').read_text())['state']=='FAILED'

def test_check_only_never_prepares_or_launches(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'prelaunch',lambda r:('abc',{},{}))
    fail=Mock(side_effect=AssertionError('numerical preparation reached'));monkeypatch.setattr(c,'prepare_role',fail);monkeypatch.setattr(c,'launch',fail)
    monkeypatch.setattr(sys,'argv',[str(P),'--review',str(tmp_path/'review'),'--check-only'])
    assert c.main()==0 and not fail.called

def test_cleanup_preserves_maps_and_masks_and_rejects_failed_child(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);out=tmp_path/c.ROLES[0];out.mkdir()
    for n,r in [('LAUNCH',dict(status='failed')),('EXECUTION',dict(status='completed'))]:c.save_json(out/(n+'.json'),r)
    with pytest.raises(ValueError):c.compact(out)
    c.save_json(out/'LAUNCH.json',dict(status='completed'))
    donor=out/'upstream/products/network11/donor-continuity';(donor/'ptc').mkdir(parents=True)
    for n in ('x-conditioned.f64','ptc/segment-0-centered.f64','ptc/segment-0-cleaned.f64','ptc/segment-0-eligible.u8'):(donor/n).write_bytes(b'data')
    (out/'map.fits').write_bytes(b'map');c.compact(out)
    assert not (donor/'x-conditioned.f64').exists() and not (donor/'ptc/segment-0-centered.f64').exists()
    assert (donor/'ptc/segment-0-cleaned.f64').read_bytes()==b'data' and (out/'map.fits').read_bytes()==b'map'
