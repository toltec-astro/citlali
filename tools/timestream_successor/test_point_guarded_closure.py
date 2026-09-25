"""The parallel evidence seal must authenticate payloads, not merely copy ledgers."""
import pytest
from close_point_guarded_feedback import parallel_seal
from point_fruit_trajectory import manifest,verify_manifest


def test_parallel_seal_matches_existing_verifier_and_rejects_tampering(tmp_path):
    nested=tmp_path/'branch';nested.mkdir();(nested/'payload').write_bytes(b'original\x00\xff');manifest(nested)
    (nested/'payload').write_bytes(b'tampered')
    with pytest.raises(Exception,match='nested manifest mismatch'):parallel_seal(tmp_path,workers=2)
    assert not (tmp_path/'FILES_SHA256SUMS').exists()
    (nested/'payload').write_bytes(b'original\x00\xff')
    result=parallel_seal(tmp_path,workers=2)
    assert result['files']==2 and result['nested_entries_reconciled']==1
    assert verify_manifest(tmp_path)==2
    with pytest.raises(Exception,match='already sealed'):parallel_seal(tmp_path)


def test_parallel_seal_rejects_external_link(tmp_path):
    (tmp_path/'link').symlink_to('/etc/hosts')
    with pytest.raises(Exception,match='external/symlink'):parallel_seal(tmp_path)
