import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound
import point_processing as processing
import point_speed_experiment as experiment

@pytest.mark.parametrize('N',[0,1,3,5,10])
@pytest.mark.parametrize('failed_guard',[False,'guard','cleanup'])
def test_exact_short_termination_and_one_endpoint(tmp_path,monkeypatch,N,failed_guard):
    out=tmp_path/'run';(out/'freeze/shapes').mkdir(parents=True)
    p=out/'freeze/prospective.json';save_json(p,dict(schema='citlali-point-speed-pca-v1',observation=152430,array='a2000',feedback_count=N,ptc=dict(rank=10)))
    save_json(out/'freeze/budget.json',dict(campaign_root=str(tmp_path)))
    fake=out/'input';fake.write_text('{}');save_json(out/'freeze/parents.json',dict(map_request=bound(fake),parent_plan=bound(fake)))
    for b in processing.BRANCHES[1:4]:(out/f'freeze/shapes/{b}.fits').write_bytes(b'fits')
    maps=out/'M0/products/a2000';maps.mkdir(parents=True);(maps/'map.fits').write_bytes(b'fits');(maps.parent/'attempt.yaml').write_text('{}')
    m=dict(guard=dict(accepted=True));save_json(out/'freeze/M0-measurement.json',m)
    c=dict(fixed=SimpleNamespace(),header={},original=np.zeros((2,2)),support=np.ones((2,2),bool));calls=[];endpoints=[]
    monkeypatch.setattr(processing,'context',lambda *a:c)
    monkeypatch.setattr(processing,'patch_map',lambda *a:(c['original'],c['support'],None,{}))
    monkeypatch.setattr(processing,'candidate',lambda *a:{})
    monkeypatch.setattr(processing,'continuation_reason',lambda *a:(None,None))
    monkeypatch.setattr(processing,'load',lambda *a:({'VALID':c['support'],'CANDIDATE':c['original']},{}))
    def publish(d,*args):
        p=d/'candidate.fits';p.write_bytes(b'fits');return p,{}
    monkeypatch.setattr(processing,'publish',publish)
    monkeypatch.setattr(processing,'infer_base',lambda *a:(m,({},None,None)))
    monkeypatch.setattr(processing,'read_yaml',lambda *a:{})
    def check(*a,**k):
        if failed_guard=='guard':raise ValueError('forced native support failure')
        return {}
    monkeypatch.setattr(processing,'processing_guard',check)
    monkeypatch.setenv('CITLALI_POINT_TIMING',str(out/'timing.jsonl'))
    monkeypatch.setattr(processing,'learned_state',lambda *a:[])
    monkeypatch.setattr(processing,'selection_difference',lambda *a:{})
    def cleanup(*a,**k):
        if failed_guard=='cleanup':raise OSError('forced cleanup I/O failure')
    monkeypatch.setattr(processing,'cleanup_cycle',cleanup)
    def call(out,binary,q,d):
        calls.append(json.loads(q.read_text())['cycle_index']);d=d/'products';d.mkdir(parents=True)
        for n in ('total.fits','restored.fits','receipt.yaml'):(d/n).write_text('{}')
        return 0
    monkeypatch.setattr(processing,'call',call)
    monkeypatch.setattr(processing,'endpoint',lambda *a:endpoints.append(a[0]) or [])
    result=processing.execute_branch(out,Path('binary'),'BASE-POINT','REF')
    if failed_guard and N:
        assert calls==[1] and endpoints==[] and not result['ready']
        published=json.loads((out/'READY-01.json').read_text())
        failed=json.loads(Path(published['state']).read_text())
        assert failed['cycle_index']==1 and not failed['ready']
        assert published['status']=='computation_or_processing'
        return
    assert calls==list(range(1,N+1))
    assert len(endpoints)==1 and result['completed_cycle']==N and result['endpoint_status']=='completed'
    assert len(result['histories'])==N+1 and not result['m10_applied']
    with pytest.raises(ValueError):processing.request(out,fake,'BASE-POINT','REF',N+1,fake,fake)


def test_speed_attempts_cannot_retry_and_keep_absolute_counts(tmp_path):
    experiment.consume(tmp_path,'trajectory',{'run':'first'})
    with pytest.raises(ValueError,match='duplicate'):experiment.consume(tmp_path,'trajectory',{'run':'first'})
    for i in range(1,experiment.LIMITS['trajectory']):experiment.consume(tmp_path,'trajectory',{'run':str(i)})
    with pytest.raises(ValueError,match='ceiling'):experiment.consume(tmp_path,'trajectory',{'run':'overflow'})


def test_failed_child_does_not_report_completed(tmp_path,monkeypatch):
    import point_pipeline
    out=tmp_path/'failed';(out/'freeze').mkdir(parents=True)
    save_json(out/'freeze/prospective.json',dict(schema='citlali-point-speed-pca-v1',feedback_count=3,experimental_method='observed-als',ptc=dict(rank=10),experiment_operands=[]))
    monkeypatch.setattr(point_pipeline,'validate_environment',lambda *a:{})
    def fail(*a):raise RuntimeError('native failure')
    monkeypatch.setattr(experiment,'upstream',fail)
    assert experiment.once(out,Path('binary'),0)==1
    assert json.loads((out/'EXECUTION.json').read_text())['status']=='processing_failed'


def test_ready_status_is_atomic_current_failure(tmp_path,monkeypatch):
    from point_timing import ready
    timing=tmp_path/'timing.jsonl';monkeypatch.setenv('CITLALI_POINT_TIMING',str(timing))
    state=tmp_path/'state.json';save_json(state,dict(cycle_index=1,ready=False,stop_kind='processing_failed'))
    ready(tmp_path,1,tmp_path/'map.fits',dict(guard=dict(accepted=True)),state)
    r=json.loads((tmp_path/'READY-01.json').read_text())
    assert r['status']=='processing_failed' and json.loads(Path(r['state']).read_text())['cycle_index']==1
    assert not list(tmp_path.glob('*.pending'))


def test_initial_source_rejection_has_diagnostic_milestone(tmp_path,monkeypatch):
    import point_pipeline
    out=tmp_path/'rejected';(out/'freeze').mkdir(parents=True);(out/'M0/products/a2000').mkdir(parents=True)
    (out/'M0/products/a2000/map.fits').write_bytes(b'fits')
    save_json(out/'freeze/prospective.json',dict(schema='citlali-point-speed-pca-v1',feedback_count=0,experimental_method='observed-als',ptc=dict(rank=10),experiment_operands=[]))
    save_json(out/'UPSTREAM.json',dict(status='source_or_covariance_failed',guard=dict(complete=True,accepted=False,message='source rejected')))
    monkeypatch.setenv('CITLALI_POINT_TIMING',str(out/'timing.jsonl'))
    monkeypatch.setattr(point_pipeline,'validate_environment',lambda *a:{})
    monkeypatch.setattr(experiment,'upstream',lambda *a:None)
    assert experiment.once(out,Path('binary'),0)==1
    assert json.loads((out/'EXECUTION.json').read_text())['status']=='source_rejected'
    assert json.loads((out/'READY-00.json').read_text())['status']=='source_rejected'


def test_invalid_first_candidate_blocks_repeats_and_injections(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(experiment,'prepare_run',lambda root,name,*a:root/name)
    def launch(root,out,*a):calls.append(out);return {'status':'failed'}
    monkeypatch.setattr(experiment,'launch',launch)
    monkeypatch.setattr(experiment,'failure_details',lambda *a:dict(status='unsupported',native_reasons=['masked-covariance-unsupported-pair-overlap']))
    rows=experiment.run_candidate_family(tmp_path,Path('binary'),Path('review'),tmp_path,'B','masked-covariance-count-v1',10)
    assert len(calls)==1 and len(rows)==6 and rows[0]['status']=='unsupported'
    assert all(r['status']=='blocked' for r in rows[1:])


def test_candidate_failure_returns_nonzero_and_preserves_other_family(tmp_path,monkeypatch):
    save_json(tmp_path/'PHASE-A.json',dict(status='completed'));(tmp_path/'report').mkdir();(tmp_path/'report/PHASE-A.md').write_text('measured')
    def family(root,binary,review,input_root,phase,method,rank):
        if method=='warm-observed-als':raise RuntimeError('native route failed')
        return [dict(method=method,status='completed',cell=i) for i in range(6)]
    monkeypatch.setattr(experiment,'run_candidate_family',family)
    assert experiment.candidates(tmp_path,Path('binary'),Path('review'),tmp_path)==1
    record=json.loads((tmp_path/'PHASE-B.json').read_text());assert record['status']=='completed_with_failures'
    assert sum(x['status']=='completed' for x in record['matrix'])==6


def test_partial_native_receipt_retains_unsupported_reason(tmp_path):
    p=tmp_path/'run/BASE-POINT/REF/a2000/cycle-01/native/products/network11-segment0/receipt.yaml'
    p.parent.mkdir(parents=True);p.write_text('converged: false\nstopping_reason: masked-covariance-unsupported-pair-overlap\n')
    assert experiment.failure_details(tmp_path)['status']=='unsupported'
    p.write_text('converged: false\nstopping_reason: internal-normal-matrix-failure\n')
    assert experiment.failure_details(tmp_path)['status']=='solver_failed'
    p.unlink()
    assert experiment.failure_details(tmp_path)['status']=='processing_failed'
