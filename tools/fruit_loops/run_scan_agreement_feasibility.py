#!/usr/bin/env python3
"""EL-F13 registered execution/resource supervisor and separate post-freeze report.

This executes the offline Python/C++ analyzer only, never Citlali. Each worker
receives a separate current-boundary input capability. Reporting opens EL-F12
outcomes only after all six prediction records and their maps are frozen.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import subprocess
import sys
import time

from scan_agreement import file_record,write_json,check


def now():return datetime.now(timezone.utc)


def bytes_under(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def resource_state(root,process,started):
    text=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],text=True,capture_output=True).stdout.strip() if process.poll() is None else ''
    rss=int(text)*1024 if text else 0
    return dict(elapsed_seconds=(now()-started).total_seconds(),worker_rss_bytes=rss,output_bytes=bytes_under(root))


def report(root,binding_path):
    freeze=json.loads((root/'PREDICTION_FREEZE_R0.1.json').read_text())
    for expected in freeze['files']:check(file_record(expected['path'])==expected,'frozen prediction changed')
    bindings=json.loads(Path(binding_path).read_text())
    accessed=[]; outcomes={}
    for record in bindings['evaluation_only_after_prediction_freeze']:
        check(file_record(record['path'])==record,'evaluation input identity changed')
        accessed.append(dict(**record,opened_utc=now().isoformat()))
        if Path(record['path']).name in ('Half_RESULT.json','Hold_RESULT.json'):
            outcomes[Path(record['path']).name.split('_')[0]]=json.loads(Path(record['path']).read_text())
    predictions=[json.loads((root/f'boundary_{k}'/'PREDICTION_R0.1.json').read_text()) for k in range(6)]
    check(all(x['construction']=='valid' for x in predictions),'invalid construction cannot receive scientific report')
    primary=predictions[1]['keys']
    check(len(primary)==1,'exposed negative comparison requires exact first opportunity')
    challenge=[]
    for probe in primary[0]['probes']:
        arm='Half' if probe['coefficient']==.5 else 'Hold'
        check(not outcomes[arm]['all_protections_pass'],'bound exposed negative no longer negative')
        status=probe['status']
        label='negative_challenge_failed' if status=='pass' else ('negative_challenge_avoided' if status in ('no_gain','insufficient_room') else 'negative_challenge_unassessed')
        challenge.append(dict(arm=arm,coefficient=probe['coefficient'],prediction_status=status,challenge=label))
    scoreable=sum(p['status'] in ('pass','no_gain','insufficient_room') for b in predictions for key in b['keys'] for p in key['probes'])
    summary=dict(status='insufficient_evidence' if scoreable==0 else 'feasibility_only',construction='valid',boundaries=6,
                 eligible_keys=sum(len(b['keys']) for b in predictions),scoreable_probes=scoreable,
                 primary_negative_challenge=challenge,independent_negative_opportunities=1,
                 boundary_2_label='unavailable: later outcomes are confounded by earlier actions',
                 empirical_benefit_rate='unavailable',false_action_rate='unavailable',
                 next_decision='Separate owner review before any new method, input, gate, resource scope or intervention.',
                 limitation='Shared-reference agreement can favor a wrong sky feature. Fixed-sample probes do not reproduce pre-RTC/PTC actions or adaptive trajectories.',
                 evaluations=accessed,prediction_freeze=file_record(root/'PREDICTION_FREEZE_R0.1.json'))
    write_json(root/'PAIRED_REPORT_R0.1.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def run(root,repo,setup_path=None):
    root=Path(root);repo=Path(repo)
    setup=Path(setup_path) if setup_path else root/'setup'
    registration=json.loads((setup/'REGISTRATION_R0.1.json').read_text())
    for expected in registration['frozen_files']:check(file_record(expected['path'])==expected,'registered helper/fixture changed')
    started=datetime.fromisoformat(json.loads((root/'STARTED_R0.1.json').read_text())['started_utc'])
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    receipts=[]
    for k in range(6):
        check((now()-started).total_seconds()<7200,'aggregate time limit')
        output=root/f'boundary_{k}'
        command=[sys.executable,str(setup/'scan_agreement.py'),'--input',str(setup/f'boundary_{k}_input.json'),'--library',str(setup/'scan_agreement_stream.dylib'),'--output',str(output)]
        launch=now(); peak=0;samples=[]
        with (root/f'boundary_{k}.log').open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            try:
                while process.poll() is None:
                    state=resource_state(root,process,started);peak=max(peak,state['worker_rss_bytes']);samples.append(state)
                    if state['elapsed_seconds']>=7200 or state['worker_rss_bytes']>4*1024**3 or state['output_bytes']>4*1024**3:
                        process.terminate();process.wait(timeout=10)
                        raise RuntimeError('registered resource limit reached')
                    time.sleep(1)
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=10)
                receipt=dict(boundary=k,command=command,started_utc=launch.isoformat(),finished_utc=now().isoformat(),exit_code=process.returncode,peak_sampled_worker_rss_bytes=peak,resource_samples=samples)
                write_json(root/f'boundary_{k}_RESOURCE_R0.1.json',receipt);receipts.append(receipt)
        check(process.returncode==0,f'boundary {k} failed; retained attempt requires diagnosis')
        prediction=json.loads((output/'PREDICTION_R0.1.json').read_text())
        check(prediction['peak_rss_bytes']<=4*1024**3 and bytes_under(root)<=4*1024**3,'completed-worker resource limit')
        print(json.dumps(dict(boundary=k,construction=prediction['construction'],keys=prediction['keys'],elapsed_seconds=prediction['elapsed_seconds'],peak_rss_bytes=prediction['peak_rss_bytes'])),flush=True)
    files=[file_record(p) for k in range(6) for p in sorted((root/f'boundary_{k}').iterdir())]
    write_json(root/'PREDICTION_FREEZE_R0.1.json',dict(frozen_utc=now().isoformat(),files=files,all_six_complete=True,source_registration=file_record(setup/'REGISTRATION_R0.1.json')))
    report(root,repo/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F13_BOUND_INPUTS_R0.1.json')
    write_json(root/'EXECUTION_COMPLETION_R0.1.json',dict(completed_utc=now().isoformat(),aggregate_elapsed_seconds=(now()-started).total_seconds(),output_bytes=bytes_under(root),citlali_runs=0,threads=1,boundaries=6,peak_sampled_worker_rss_bytes=max(x['peak_sampled_worker_rss_bytes'] for x in receipts)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repo',required=True);p.add_argument('--setup');args=p.parse_args()
    run(args.root,args.repo,args.setup)
