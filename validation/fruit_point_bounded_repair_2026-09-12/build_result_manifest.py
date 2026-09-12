"""Package the completed saved-map result without changing source authority."""
import datetime
import re
import shutil
import subprocess
import sys
from common import b, read, HERE, OUT, verify_freeze

def check():
    result = read(HERE/'RESULT_MANIFEST.json')
    assert (HERE/'RESULT_MANIFEST.json.sha256').read_text().split()[0] == b.digest(HERE/'RESULT_MANIFEST.json')
    for row in result['files']:
        assert b.digest(HERE/row['path']) == row['sha256'], row['path']
    expected = {r['path'] for r in result['files']}
    actual = {p.name for p in HERE.iterdir() if p.is_file() and p.name not in ['RESULT_MANIFEST.json','RESULT_MANIFEST.json.sha256']}
    assert actual == expected
    for p in HERE.glob('*.md'):
        for ref in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in ref and not ref.startswith('#'):
                assert (p.parent/ref.split('#')[0]).exists(), (p.name,ref)
    verify_freeze()
    print('Verified',len(result['files']),'result payloads;',result['disposition'])

def main():
    if '--check' in sys.argv:
        check()
        return
    verify_freeze()
    decision = read(HERE/'DECISION_EVIDENCE.json')
    verification = read(HERE/'VERIFICATION.json')
    assert not decision['operational_rerun_admitted']
    assert decision['new_cleaning_calls'] == 0
    shutil.copyfile(OUT/'PRODUCT_MANIFEST.json',HERE/'RUN_PRODUCT_MANIFEST.json')
    result = dict(identity='SCI-FRUIT-POINT-BOUNDED-REPAIR-RETEST-RESULT@r0.1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        freeze_commit=subprocess.check_output(['git','rev-parse','377164001'],text=True).strip(),
        source_freeze_sha256=b.digest(HERE/'FREEZE.json'), disposition=decision['disposition'],
        evaluator_saved_stage_passed=decision['evaluator_saved_stage_passed'],
        stopping_qualification_passed=decision['stopping_qualification_passed'],
        operational_rerun_admitted=False,new_cleaning_calls=0,new_operational_trajectories=0,
        qualification=False,independent_pointing_replication=False,
        full_external_product_manifest=str(OUT/'PRODUCT_MANIFEST.json'),
        external_payloads=verification['new_external_payloads'],
        visual_inspection='Both standalone scientific figures inspected; labels, model support and legends readable.',
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(HERE.iterdir())
               if p.is_file() and p.name not in ['RESULT_MANIFEST.json','RESULT_MANIFEST.json.sha256']])
    b.write(HERE/'RESULT_MANIFEST.json',result)
    (HERE/'RESULT_MANIFEST.json.sha256').write_text(b.digest(HERE/'RESULT_MANIFEST.json')+'  RESULT_MANIFEST.json\n')
    check()

if __name__=='__main__':
    main()
