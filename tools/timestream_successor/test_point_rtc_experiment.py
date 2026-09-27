import json
from pathlib import Path
from unittest.mock import patch
import pytest
from point_storage import RunStorage,tree_bytes
from point_rtc_experiment import consume,LIMITS


def test_storage_ignores_historical_file_count(tmp_path):
    historical=tmp_path/'historical';historical.mkdir()
    out=tmp_path/'new';out.mkdir();(out/'config').write_bytes(b'123')
    budget=dict(peak_ceiling_bytes=1_000_000,free_space_reserve_bytes=0,
                campaign_root=str(tmp_path),campaign_retained_bytes=5000,metadata_reserve_bytes=1000)
    visited=[]
    original=Path.rglob
    def walk(self,*args,**kw):
        visited.append(self);return original(self,*args,**kw)
    with patch.object(Path,'rglob',walk):
        a=RunStorage(out,budget);a.reserve(100);stage=out/'stage';stage.mkdir();(stage/'data').write_bytes(b'x'*20);a.finish(stage,100)
        for i in range(1000):(historical/str(i)).write_bytes(b'abcde')
        b=RunStorage(out,budget);b.reserve(100);stage=out/'stage2';stage.mkdir();(stage/'data').write_bytes(b'x'*20);b.finish(stage,100)
    assert visited==[out,out/'stage',out/'stage2']
    assert b.state['native_written_bytes']==40
    assert b.historical==5000


def test_storage_overrun_failure_and_no_delete_credit(tmp_path):
    budget=dict(peak_ceiling_bytes=10000,free_space_reserve_bytes=0,metadata_reserve_bytes=1000)
    s=RunStorage(tmp_path,budget);d=tmp_path/'native';d.mkdir();f=d/'data';f.write_bytes(b'x'*101)
    with pytest.raises(ValueError):s.finish(d,100)
    f.unlink();s=RunStorage(tmp_path,budget);assert s.state['charged_bytes']==101
    assert s.reconcile()['reconciled_actual_bytes']>0


def test_storage_symlink_rejected(tmp_path):
    (tmp_path/'alias').symlink_to('/private/tmp')
    with pytest.raises(ValueError):tree_bytes(tmp_path)


def test_rtc_limits_duplicates_and_defect_only_replacement(tmp_path):
    consume(tmp_path,'trajectory',dict(run='one'))
    with pytest.raises(ValueError):consume(tmp_path,'trajectory',dict(run='one'))
    for i in range(1,LIMITS['trajectory']):consume(tmp_path,'trajectory',dict(run=str(i)))
    with pytest.raises(ValueError):consume(tmp_path,'trajectory',dict(run='overflow'))
    (tmp_path/'DEFECT-REPLACEMENTS.json').write_text(json.dumps(dict(implementation_defect='routing bug',corrected_source='abc',maximum_replacements=1)))
    consume(tmp_path,'trajectory',dict(run='replacement'))
    with pytest.raises(ValueError):consume(tmp_path,'trajectory',dict(run='second'))


def test_rtc_only_cap_independent(tmp_path):
    for i in range(4):consume(tmp_path,'rtc_only',dict(run=i))
    with pytest.raises(ValueError):consume(tmp_path,'rtc_only',dict(run=4))


@pytest.mark.parametrize('exit_code,overrun',[(1,False),(0,True),(1,True)])
def test_diagnostic_accounts_writes_even_on_failure(tmp_path,exit_code,overrun):
    from point_rtc_experiment import accounted_capture
    budget=dict(peak_ceiling_bytes=10000,free_space_reserve_bytes=0,
                metadata_reserve_bytes=1000,shared_upstream_allowance=100)
    def invoke():
        d=tmp_path/'native';d.mkdir();(d/'data').write_bytes(b'x'*(101 if overrun else 20))
        return exit_code
    if overrun:
        with pytest.raises(ValueError,match='exceeded reserved'):accounted_capture(tmp_path,budget,invoke)
    else:assert accounted_capture(tmp_path,budget,invoke)==exit_code
    state=json.loads((tmp_path/'storage-ledger.json').read_text())
    assert state['native_written_bytes']==(101 if overrun else 20)
    assert state['completed_stages']==1


def test_diagnostic_reserves_free_space_before_invocation(tmp_path):
    from point_rtc_experiment import accounted_capture
    from types import SimpleNamespace
    budget=dict(peak_ceiling_bytes=10000,free_space_reserve_bytes=100,
                metadata_reserve_bytes=1000,shared_upstream_allowance=100)
    def invoke():raise AssertionError('must not run')
    with patch('point_storage.shutil.disk_usage',return_value=SimpleNamespace(free=199)):
        with pytest.raises(ValueError,match='free-space reserve'):accounted_capture(tmp_path,budget,invoke)
