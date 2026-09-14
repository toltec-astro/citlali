"""Replay the fixed onset-support experiment and its preserved executable control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable', type=Path)
    p.add_argument('control_executable', type=Path)
    p.add_argument('control_diagnosis', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    if a.control_diagnosis.resolve() in a.output.resolve().parents:
        p.error('output must be outside the sealed control')
    a.output.mkdir(parents=True, exist_ok=False)
    binding = json.loads((a.control_diagnosis/'source-binding.json').read_text())
    assert sha(a.control_executable) == binding['executables']['citlali_rtc_jump_injection']
    prior = a.control_diagnosis/'replay-final-01'
    invocations = json.loads((prior/'invocations.json').read_text())['invocations']

    def run(i):
        source, identity = invocations[i][0]['command'][1:3]
        if source != 'synthetic':
            assert sha(Path(source)) == identity.split(':sha256:')[-1]
        records = []
        for role, executable in [('control', a.control_executable), ('trial', a.executable)]:
            for mode in ['legacy', 'diagnostic']:
                command = [str(executable.resolve()), source, identity]
                if mode == 'diagnostic':
                    command.append('--loss-diagnosis')
                name = f'{i:02d}-' + ('control-' if role == 'control' else '') + mode
                start = time.perf_counter()
                with (a.output/(name+'.out')).open('w') as out, (a.output/(name+'.err')).open('w') as err:
                    result = subprocess.run(command, stdout=out, stderr=err, check=False)
                record = dict(background=i, role=role, mode=mode, command=command,
                              exit_code=result.returncode, elapsed_seconds=time.perf_counter()-start)
                if role == 'control':
                    record['control_bytes_preserved'] = sha(a.output/(name+'.out')) == sha(prior/f'{i:02d}-{mode}.out')
                records.append(record)
        return records

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(run, range(3)))
    passed = all(r['exit_code'] == 0 and r.get('control_bytes_preserved', True) for group in records for r in group)
    passed &= all(p.stat().st_size == 0 for p in a.output.glob('*.err'))
    result = dict(status='PASS' if passed else 'FAIL',
                  executable_sha256=sha(a.executable), control_executable_sha256=sha(a.control_executable),
                  invocations=records, full_corpus_rerun=False,
                  timing_scope='Concurrent fixed-input diagnostic processes; not an incremental runtime performance benchmark')
    (a.output/'invocations.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
