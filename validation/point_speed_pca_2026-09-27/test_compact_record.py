"""Destructive cleanup guards exercised only in temporary synthetic directories."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('point_speed_compact',Path(__file__).with_name('compact_record.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def campaign(tmp_path,monkeypatch):
 root=tmp_path/'campaign';root.mkdir();dest=tmp_path/'record/compact';dest.mkdir(parents=True)
 monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'DEST',dest);monkeypatch.setattr(m,'inactive',lambda:None)
 monkeypatch.setattr(m.subprocess,'check_output',lambda a,**k:'reviewed\n' if 'rev-parse' in a else '')
 monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:None)
 native=root/'A-rank10-N10-rep1/upstream/input.f64';native.parent.mkdir(parents=True);native.write_bytes(b'native')
 small=dest/'summary.json';small.write_bytes(b'{}')
 m.save(root/'FINAL-REVIEW.json',dict(source_revision='reviewed',verdict='pass'))
 m.save(dest.parent/'PROTECTED_INPUTS.json',[])
 m.save(dest.parent/'COMPACT_MANIFEST.json',dict(files=[dict(path='summary.json',sha256=m.digest(small))]))
 m.save(dest.parent/'HASH_ONLY.json.gz',dict(files=[]))
 m.save(dest.parent/'NATIVE_REMOVALS.json.gz',dict(applied=False,bytes=6,files=[dict(path=str(native),bytes=6,sha256=m.digest(native))]))
 return root,dest,native,small

def test_only_enumerated_native_is_removed(campaign):
 root,dest,native,small=campaign;unrelated=root/'keep.f64';unrelated.write_bytes(b'keep')
 m.clean('reviewed');assert not native.exists() and unrelated.read_bytes()==b'keep' and small.exists()
 assert m.read(dest.parent/'CLEANUP.json')['protected_inputs_unchanged']

@pytest.mark.parametrize('which',['compact','native','review'])
def test_changed_evidence_aborts_before_delete(campaign,which):
 root,dest,native,small=campaign
 if which=='compact':small.write_bytes(b'changed')
 elif which=='native':native.write_bytes(b'changed')
 else:m.save(root/'FINAL-REVIEW.json',dict(source_revision='foreign',verdict='pass'))
 with pytest.raises(AssertionError):m.clean('reviewed')
 assert native.exists() and not (dest.parent/'CLEANUP.json').exists()

def test_outside_payload_and_dirty_tree_rejected(campaign,monkeypatch,tmp_path):
 root,dest,native,small=campaign;outside=tmp_path/'outside.f64';outside.write_bytes(b'keep')
 m.save(dest.parent/'NATIVE_REMOVALS.json.gz',dict(applied=False,bytes=4,files=[dict(path=str(outside),bytes=4,sha256=m.digest(outside))]))
 with pytest.raises(AssertionError):m.clean('reviewed')
 assert outside.exists() and native.exists()
 monkeypatch.setattr(m.subprocess,'check_output',lambda a,**k:'reviewed\n' if 'rev-parse' in a else ' M changed')
 with pytest.raises(AssertionError):m.clean('reviewed')
 assert native.exists()

def test_protected_dependency_change_aborts(campaign,tmp_path):
 root,dest,native,small=campaign;source=tmp_path/'original.nc';source.write_bytes(b'original')
 deps=root/'A-rank10-N10-rep1/freeze/dependencies.json';deps.parent.mkdir();m.save(deps,dict(inputs=[dict(path=str(source),sha256=m.digest(source))]))
 m.save(dest.parent/'PROTECTED_INPUTS.json',m.protect());source.write_bytes(b'changed')
 with pytest.raises(AssertionError):m.clean('reviewed')
 assert source.read_bytes()==b'changed' and native.exists()
