"""Run the frozen paired analyzer, retaining bounded resource evidence."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

import psutil

REPO = Path('/Users/gwilson/.codex/worktrees/4c31/citlali-refactor')
ROOT = Path('/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f12-response-aware-intervention-r0.1')
sys.path.insert(0, str(REPO))
from tools.fruit_loops.analyze_response_intervention import analyze_candidate, file_record, write_json
from tools.fruit_loops.run_response_intervention import ORDER, parse_time_log, tree_bytes, resource_violation

arm = sys.argv[1]
assert arm in ('Half', 'Hold')
out = ROOT / 'analysis'
if len(sys.argv) > 2 and sys.argv[2] == '--child':
    result = analyze_candidate(ROOT, arm, out / arm)
    write_json(out / f'{arm}_RESULT.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))
    raise SystemExit(0)

# The two primary arms must both be retained and complete before measurements.
for name in ORDER:
    receipt = json.loads((ROOT / name / 'EXECUTION_RECEIPT.json').read_text())
    if name == 'H/uninjected':
        proof = ROOT / name / 'SCALAR_DIMENSION_REPAIR_R0.2.json'
        repair = json.loads(proof.read_text())
        assert repair['original_failed_receipt'] == file_record(ROOT / name / 'EXECUTION_RECEIPT.json')
        receipt = repair['execution_receipt_after_comparison_repair']
    assert receipt['status'] == 'pass', name
registration = ROOT / 'setup/REGISTRATION_R0.2.json'
records = json.loads(registration.read_text())['artifacts']
source = REPO / 'tools/fruit_loops/analyze_response_intervention.py'
assert file_record(source) in records
out.mkdir(exist_ok=True)
started = time.time()
first = min(json.loads((ROOT / name / 'STARTED.json').read_text())['unix_time'] for name in ORDER)
command = ['/usr/bin/time', '-l', '/Users/gwilson/tolteca/bin/python', str(Path(__file__).resolve()), arm, '--child']
write_json(out / f'{arm}_ANALYSIS_STARTED_R0.1.json', {'command': command, 'started_unix': started,
    'registration': file_record(registration), 'analyzer': file_record(source), 'wrapper': file_record(Path(__file__))})
env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR='/private/tmp/fruit-el-f12-prototype-20260905/mpl',
           XDG_CACHE_HOME='/private/tmp/fruit-el-f12-prototype-20260905/cache')
env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})
log = out / f'{arm}_ANALYSIS_R0.1.log'
peak = 0
failure = None
with log.open('x') as stream:
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    try:
        while process.poll() is None:
            try:
                children = psutil.Process(process.pid).children(recursive=True)
                for child in children:
                    try:
                        peak = max(peak, child.memory_info().rss)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
            except psutil.NoSuchProcess:
                pass
            retained, spool = tree_bytes(ROOT)
            # The analysis has the unchanged aggregate, memory and disk limits;
            # the per-trajectory clock does not start a new replay allowance.
            failure = resource_violation(0, time.time() - first, peak, retained, spool)
            if failure:
                for child in reversed(children):
                    try: child.terminate()
                    except psutil.NoSuchProcess: pass
                process.terminate()
                break
            time.sleep(.25)
        code = process.wait()
    except Exception:
        process.terminate()
        code = process.wait()
        failure = traceback.format_exc()
record = {'arm': arm, 'exit_code': code, 'status': 'pass' if code == 0 and failure is None else 'unavailable',
          'failure': failure, 'elapsed_seconds': time.time() - started, 'aggregate_elapsed_seconds': time.time() - first,
          'sampled_peak_process_rss_bytes': peak, 'log': file_record(log), 'retained_bytes': tree_bytes(ROOT)[0]}
if code == 0:
    record['resources'] = parse_time_log(log.read_text())
    record['result'] = file_record(out / f'{arm}_RESULT.json')
write_json(out / f'{arm}_ANALYSIS_RECEIPT_R0.1.json', record)
print(json.dumps(record, indent=2))
raise SystemExit(0 if record['status'] == 'pass' else 1)
