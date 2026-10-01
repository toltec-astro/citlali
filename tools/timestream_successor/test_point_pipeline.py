"""Consolidation contracts, not another scientific calibration campaign."""
import json
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
import point_pipeline as pipeline
from point_additive_diagnostics import save_json
from point_guarded_feedback import infer, infer_base, publish
from point_fruit_diagnostics import load
from test_point_guarded_feedback import fixture


def test_base_extraction_preserves_fit_and_all_candidate_planes(tmp_path):
    c,s,v=fixture();old,prepared=infer(c,s,v)
    c['neighborhood'].scores=Mock(side_effect=AssertionError('parked method invoked'))
    new,base=infer_base(c,s,v)
    assert new==old and base[2] is None
    a,_=publish(tmp_path/'old',c,s,prepared,'BASE-POINT',3.)
    b,_=publish(tmp_path/'new',c,s,base,'BASE-POINT',3.)
    aa,_=load(a);bb,_=load(b)
    for k in aa:np.testing.assert_array_equal(aa[k],bb[k])



def freeze_point_recipe(root):
    (root/'freeze').mkdir()
    save_json(root/'freeze/prospective.json',dict(schema='citlali-bright-point-reference-v1',mode='POINT',subobservation=0,scan=2,cycles=10,threshold=3,coverage_policy='high-coverage-informational'))

@pytest.mark.parametrize('mode',['forward','feedback'])
def test_forward_and_feedback_use_one_initial_implementation(tmp_path,monkeypatch,mode):
    freeze_point_recipe(tmp_path)
    calls=[]
    monkeypatch.setattr(pipeline,'validate_environment',lambda *a:{})
    def forward(out,binary):
        calls.append('M0');save_json(out/'UPSTREAM.json',dict(status='accepted'))
    monkeypatch.setattr(pipeline,'upstream',forward)
    def feedback(*a):calls.append('BASE');return dict(ready=True,completed_cycle=10,endpoint_status='completed')
    monkeypatch.setattr(pipeline,'execute_branch',feedback)
    assert pipeline.execute(mode,tmp_path,Path('binary'))==0
    assert calls==(['M0','BASE'] if mode=='feedback' else ['M0'])
    with pytest.raises(ValueError,match='already attempted'):pipeline.execute(mode,tmp_path,Path('binary'))


def test_rejected_initial_source_never_starts_feedback(tmp_path,monkeypatch):
    freeze_point_recipe(tmp_path)
    monkeypatch.setattr(pipeline,'validate_environment',lambda *a:{})
    monkeypatch.setattr(pipeline,'upstream',lambda o,b:save_json(o/'UPSTREAM.json',dict(status='source_or_covariance_failed',guard=dict(complete=True,message='source rejected'))))
    blocked=Mock(side_effect=AssertionError('feedback on failed source'));monkeypatch.setattr(pipeline,'execute_branch',blocked)
    assert pipeline.execute('feedback',tmp_path,Path('binary'))==0
    assert not blocked.called and json.loads((tmp_path/'EXECUTION.json').read_text())['status']=='source_rejected'


def test_saved_report_relocates_without_original_inputs_or_execution(tmp_path,monkeypatch):
    root=tmp_path/'relocated';root.mkdir();(root/'freeze').mkdir()
    save_json(root/'EXECUTION.json',dict(mode='forward',status='completed'))
    save_json(root/'UPSTREAM.json',dict(status='accepted'))
    save_json(root/'freeze/prospective.json',dict(array='a2000',observation=7,input_root='/unavailable',reference=dict(path='/absent')))
    f=root/'M0/products/a2000/map.fits.gz';f.parent.mkdir(parents=True);f.write_bytes(b'saved product')
    monkeypatch.setattr(pipeline,'upstream',Mock(side_effect=AssertionError('hidden science')))
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert pipeline.inspect_saved(root)['observation']==7
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    f.unlink()
    with pytest.raises(ValueError,match='never regenerates'):pipeline.inspect_saved(root)


def test_generic_request_all_supported_arrays_and_explicit_identity(tmp_path):
    from point_processing import request, owner
    from point_fruit_trajectory import bound
    f=tmp_path/'file';f.write_text('{}');(tmp_path/'freeze').mkdir()
    for array in ('a1100','a1400','a2000'):
        save_json(tmp_path/'freeze/prospective.json',dict(schema='citlali-bright-point-reference-v1',observation=456,array=array))
        save_json(tmp_path/'freeze/parents.json',dict(parent_plan=bound(f),map_request=bound(f)))
        o=owner(tmp_path,'BASE-POINT','REF');assert o['observation']==456 and array in o['parents']
        r=request(tmp_path,f,'BASE-POINT','REF',1,f,f);assert r['array']==array and r['schema']=='citlali-bright-point-reference-v1'
        for pol,branch,k in [('CORR','REF',1),('BASE-POINT','C-core-plus',1),('BASE-POINT','REF',11)]:
            with pytest.raises(ValueError):request(tmp_path,f,pol,branch,k,f,f)
