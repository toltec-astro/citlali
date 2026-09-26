"""Small contracts for rank wiring, immutable operands and retained-only reporting."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound
from point_processing import owner, request
from point_rank_experiment import consume, LIMITS
from point_reference import context
from test_point_guarded_feedback import fixture

@pytest.mark.parametrize('rank',[5,10,15])
def test_m0_and_every_feedback_request_use_same_rank_profile(tmp_path,rank):
    f=tmp_path/'operand';f.write_text('{}');(tmp_path/'freeze/shapes').mkdir(parents=True)
    for b in ('C-core-plus','C-core-minus','C-shoulder'):(tmp_path/f'freeze/shapes/{b}.fits').write_bytes(b'operand')
    save_json(tmp_path/'freeze/prospective.json',dict(schema='citlali-point-ptc-rank-v1',observation=152430,array='a2000',profile=f'point-152430-ptc-rank-{rank}',ptc=dict(method='observed-als',rank=rank)))
    save_json(tmp_path/'freeze/parents.json',dict(parent_plan=bound(f),map_request=bound(f)))
    for branch in ('REF','C-core-plus','C-core-minus','C-shoulder'):
        o=owner(tmp_path,'BASE-POINT',branch);assert o['identity']=='POINT-PTC-RANK-v1'
        assert json.loads(Path(o['prospective']['path']).read_text())['ptc']['rank']==rank
        for k in range(11):
            q=request(tmp_path,f,'BASE-POINT',branch,k,f,f)
            assert q['schema']=='citlali-point-ptc-rank-v1' and q['mode']==('zero' if k==0 else 'reference')
            assert ('injection' in q)==branch.startswith('C-')
            assert ('previous_state' in q)==bool(k)
    for pol,branch,k in [('CORR','REF',1),('BASE-POINT','D-seed-shoulder',1),('BASE-POINT','REF',11)]:
        with pytest.raises(ValueError):request(tmp_path,f,pol,branch,k,f,f)

def test_attempts_count_failure_and_forbid_implicit_retry(tmp_path):
    consume(tmp_path,'trajectory',dict(rank=5,branch='REF'))
    with pytest.raises(ValueError,match='duplicate'):consume(tmp_path,'trajectory',dict(rank=5,branch='REF'))
    for i in range(LIMITS['trajectory']-1):consume(tmp_path,'trajectory',dict(i=i))
    with pytest.raises(ValueError,match='ceiling'):consume(tmp_path,'trajectory',dict(i=99))
    assert len((tmp_path/'attempts.jsonl').read_text().splitlines())==12

def test_common_measurement_geometry_does_not_transplant_noise(tmp_path,monkeypatch):
    # Use a complete tiny saved context shaped like the shared 41x41 fixture.
    from point_reference import context
    c,s,v=fixture();c['header'].update(CRVAL1=0.,CRVAL2=0.);f=c['fixed'];p=f.parameters
    g=dict(beam=f.beam,major=f.major,header=dict(c['header']),guard=c.get('guard',{}),full_WCS={'CRPIX1':21,'CRPIX2':21})
    save_json(tmp_path/'geometry.json',g)
    np.savez(tmp_path/'context.npz',original=s,support=v,full_support=v,scale=c['scale'],psd=c['psd'],empirical=c['psd'],base_scale=f.scale,noise_valid=f.noise_valid,parameters=p,q=f.weights,k0=c['k0'])
    for name in ('BASE','C0'):
        save_json(tmp_path/(name+'.json'),dict(request=dict(high_coverage_usable=True)))
        np.savez(tmp_path/(name+'.npz'),coverage_extrapolated=np.zeros_like(v),source_mask=np.zeros_like(v))
    import point_reference
    monkeypatch.setattr(point_reference,'Neighborhood',lambda *a,**k:type('N',(),dict(lags_supported=True))())
    other=p.copy();other[1]+=1
    mg=dict(header=dict(c['header']),parameters=other.tolist(),major=f.major*1.1,weights=(f.weights*.9).tolist())
    changed=context(tmp_path,mg)
    np.testing.assert_array_equal(changed['parameters'],p)
    np.testing.assert_array_equal(changed['scale'],c['scale'])
    np.testing.assert_array_equal(changed['fixed'].parameters,other)
    np.testing.assert_array_equal(changed['fixed'].weights,f.weights*.9)

def test_cross_rank_basis_distance_uses_actual_dimensions(tmp_path):
    from point_fruit_diagnostics import basis_distance
    a=np.eye(20)[:,:10];b=np.eye(20)[:,:5]
    a.tofile(tmp_path/'a');b.tofile(tmp_path/'b')
    assert basis_distance(tmp_path/'a',tmp_path/'b',20,10,5)==pytest.approx(np.sqrt(5))

def test_equivalent_uncertainty_preserves_sign_and_distortion_warnings():
    from report_point_rank import equivalent
    assert equivalent(4.,.8,.1,'C-core-plus')[0]==5.
    for sigma,g,eta,branch in [(4.,-1.,.1,'C-core-minus'),(4.,1.,1.2,'C-core-plus'),(4.,0.,0.,'C-core-plus'),(None,1.,0.,'C-core-plus'),(4.,1.,0.,'C-shoulder')]:
        assert equivalent(sigma,g,eta,branch)[0] is None
