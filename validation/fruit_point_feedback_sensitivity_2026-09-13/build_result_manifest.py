"""Bind the complete diagnostic result and verify its local links and hashes."""
import datetime
import re
import shutil
import subprocess
import sys
from common import b,read,HERE,OUT,verify_freeze

def check():
    result=read(HERE/'RESULT_MANIFEST.json')
    assert (HERE/'RESULT_MANIFEST.json.sha256').read_text().split()[0]==b.digest(HERE/'RESULT_MANIFEST.json')
    for row in result['files']:
        assert b.digest(HERE/row['path'])==row['sha256'],row['path']
    expected={r['path'] for r in result['files']}
    actual={p.name for p in HERE.iterdir() if p.is_file() and p.name not in ['RESULT_MANIFEST.json','RESULT_MANIFEST.json.sha256']}
    assert actual==expected
    for p in HERE.glob('*.md'):
        for ref in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in ref and not ref.startswith('#'):
                assert (p.parent/ref.split('#')[0]).exists(),(p.name,ref)
    verify_freeze()
    print('Verified',len(result['files']),'result payloads; six branches, no new feedback optimization')

def main():
    if '--check' in sys.argv:
        check();return
    verify_freeze()
    decision=read(HERE/'DECISION_EVIDENCE.json')
    verification=read(HERE/'VERIFICATION.json')
    assert verification['cleaning_calls_in_experiment']==6 and verification['verification_cleaning_calls']==0
    for name in ['PRODUCT_MANIFEST.json','SELECTED_STATES.json','START.json','COMPLETE.json']:
        shutil.copyfile(OUT/name,HERE/('RUN_'+name))
    frozen=read(HERE/'FREEZE.json');started=read(OUT/'START.json')
    assert datetime.datetime.fromisoformat(frozen['utc'])<datetime.datetime.fromisoformat(started['utc'])
    assert started['source_freeze_sha256']==b.digest(HERE/'FREEZE.json')
    result=dict(identity='SCI-FRUIT-POINT-PAIRED-FEEDBACK-SENSITIVITY-RESULT@r0.1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_snapshot_commit=subprocess.check_output(['git','rev-parse','bece76cc4'],text=True).strip(),
        source_hash_freeze_utc=frozen['utc'],execution_started_utc=started['utc'],
        source_commit_ordering='Commit completed after run launch; exact source/input hash freeze preceded execution. See PROVENANCE_NOTE.md.',
        disposition=decision['disposition'],recommendation=decision['recommendation'],
        cleaning_calls=6,feedback_optimization_calls=0,old_qualification='failed_and_preserved',
        new_candidate_admitted=False,full_trajectory_stability=False,production=False,
        external_payloads=verification['external_payloads'],
        figures_visually_inspected=['NEXT_MAP_SENSITIVITY.png','FEEDBACK_AND_NEXT_MAP.png'],
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(HERE.iterdir())
            if p.is_file() and p.name not in ['RESULT_MANIFEST.json','RESULT_MANIFEST.json.sha256']])
    b.write(HERE/'RESULT_MANIFEST.json',result)
    (HERE/'RESULT_MANIFEST.json.sha256').write_text(b.digest(HERE/'RESULT_MANIFEST.json')+'  RESULT_MANIFEST.json\n')
    check()

if __name__=='__main__':main()
