"""Bind source bytes and preservation evidence before saved-map execution."""
import datetime
import re
import subprocess
from pathlib import Path
from common import b, read, HERE, ROOT, PRIOR, BASE, SAVED, NULL

def preservation():
    previous = read(PRIOR/'PRESERVATION_START.json')
    manifests = [Path(p['path']) for p in previous['packets']]+[PRIOR/'RESULT_MANIFEST.json']
    packets = []
    for path in manifests:
        rows = read(path)['files']
        for r in rows:
            assert b.digest(path.parent/r['path']) == r['sha256'], r['path']
        packets.append(dict(path=str(path), sha256=b.digest(path), payloads=len(rows)))
    frozen = ROOT/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4'
    records = re.findall(r'^\| `([^`]+)` \| (\d+) \| `([a-f0-9]{64})` \|$', (frozen/'PACKET_MANIFEST.md').read_text(), re.M)
    assert len(records) == 57
    for name, _, sha in records:
        assert b.digest(frozen/name) == sha, name
    for r in previous['protected_worktrees']:
        assert subprocess.check_output(['git', '-C', r['path'], 'status', '--short'], text=True) == r['status']
    products = read(SAVED/'PRODUCT_MANIFEST.json')['files']
    for r in products:
        assert b.digest(SAVED/r['path']) == r['sha256'], r['path']
    return dict(packets=packets, frozen_ordinary_MAP_payloads=57,
                saved_reduction_payloads=len(products), saved_product_manifest_sha256=b.digest(SAVED/'PRODUCT_MANIFEST.json'),
                protected_worktrees=previous['protected_worktrees'],
                archive_policy='presence/status only; never read, hash or unpack')

def main():
    assert not (HERE/'FREEZE.json').exists()
    preserved = preservation()
    b.write(HERE/'PRESERVATION_START.json', preserved)
    paths = list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md', HERE/'PRESERVATION_START.json',
        BASE/'experiment_r02.py', PRIOR/'candidate.py', PRIOR/'evaluation.py', PRIOR/'CASES.json',
        PRIOR/'RESULT_MANIFEST.json', PRIOR/'DIAGNOSTIC_EVIDENCE.json',
        SAVED/'PRODUCT_MANIFEST.json', SAVED/'RECEIPTS.json', SAVED/'geometry.npz', NULL]
    paths += sorted(SAVED.glob('*/*/pass*_maps.npz'))
    b.write(HERE/'FREEZE.json', dict(identity='SCI-FRUIT-POINT-BOUNDED-REPAIR-RETEST@r0.1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        starting_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        files=[dict(path=str(p), bytes=p.stat().st_size, sha256=b.digest(p)) for p in sorted(set(paths))],
        saved_map_count=167, saved_candidate_array_problems=144, pre_PTC_input_reads=0,
        numerical_execution_started=False, cleaning_calls=0))
    (HERE/'FREEZE.json.sha256').write_text(b.digest(HERE/'FREEZE.json')+'  FREEZE.json\n')
    print('Bound', len(paths), 'files; preserved', preserved['saved_reduction_payloads'], 'saved reduction payloads')

if __name__ == '__main__':
    main()
