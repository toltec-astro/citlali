"""Bind completed review artifacts and the immutable retained run products."""
import datetime
import subprocess
from common import b, read, verify_freeze, HERE, OUT

def main():
    verify_freeze()
    target=HERE/'RESULT_MANIFEST.json'
    if target.exists():raise FileExistsError('preserve the existing result manifest')
    files=[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=b.digest(p))
           for p in sorted(HERE.iterdir()) if p.is_file() and not p.name.startswith('RESULT_MANIFEST.json')]
    run=read(HERE/'RUN_PRODUCT_MANIFEST.json')
    value=dict(identity='SCI-FRUIT-POINT-OPERATIONAL-GATE-TRIAL-RESULT@r0.1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        freeze_commit=subprocess.check_output(['git','rev-parse','7836bc8c3'],text=True).strip(),
        state='complete_candidate_not_accepted_recommend_bounded_completion_evaluator_revision',
        new_cleaning_calls=167,trajectories_registered=34,trajectories_complete=22,
        independent_pointing_replication=False,qualification=False,
        files=files,total_payload_bytes=sum(r['bytes'] for r in files),
        external_products=dict(root=str(OUT),manifest='RUN_PRODUCT_MANIFEST.json',
            manifest_sha256=b.digest(HERE/'RUN_PRODUCT_MANIFEST.json'),
            payload_count=len(run['files']),payload_bytes=sum(r['bytes'] for r in run['files'])),
        verification=read(HERE/'VERIFICATION.json'))
    b.write(target,value)
    (HERE/'RESULT_MANIFEST.json.sha256').write_text(b.digest(target)+'  RESULT_MANIFEST.json\n')
    print('Result payloads:',len(files),'bytes:',value['total_payload_bytes'])

if __name__=='__main__':main()
