#!/usr/bin/env python3
"""Owner-run Slurm scheduling for the fixed, saved-product PTC diagnostic."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
PRIOR = Path.home() / 'work_toltec/wilson/citlali_testing/citlali-successor-diagnostics-64621061'


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sealed(path, value=None):
    if value is None:
        obj = json.loads(path.read_text())
        require(obj['sha256'] == canonical(obj['record']), 'damaged checkpoint: ' + str(path))
        return obj['record']
    write(path, dict(record=value, sha256=canonical(value)))
    return value


@contextmanager
def lock(path, shared=False, wait_seconds=0):
    # Prior lock already exists and is opened read-only; no old packet is edited.
    if shared:
        require(path.exists(), 'required shared lock is absent: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('r' if shared else 'a') as stream:
        until = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= until:
                    raise RuntimeError('another writer is active: ' + str(path))
                time.sleep(2)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def manifest_check(root, filename):
    for line in (root / filename).read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(sha(root / name) == digest, 'control changed: ' + name)


def controls():
    meta = json.loads((ROOT / 'PREPARATION.json').read_text())
    require(meta['ready_for_owner_submission'], 'packet not reviewed')
    review = json.loads((ROOT / 'INDEPENDENT_REVIEW.json').read_text())
    core = {k: v for k, v in meta.items() if k != 'ready_for_owner_submission'}
    require(review['verdict'] == 'PASS' and review['reporting_revision'] == meta['reporting_revision'] and
            review['reporting_tree'] == meta['reporting_tree'] and
            review['reviewed_controls_sha256'] == sha(ROOT / 'REVIEWED_SHA256SUMS') and
            review['preparation_core_sha256'] == canonical(core), 'independent review binding failed')
    manifest_check(ROOT, 'REVIEWED_SHA256SUMS')
    return meta


def load_prior(meta):
    require(sha(PRIOR / 'SHA256SUMS') == meta['prior_manifest_sha256'], 'serial packet identity changed')
    manifest_check(PRIOR, 'SHA256SUMS')
    spec = importlib.util.spec_from_file_location('serial_diagnostic_packet', PRIOR / 'diagnostics.py')
    serial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serial)
    return serial


def environment():
    return dict(python=sys.version, packages=sorted(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()))


def binding(meta, env):
    return dict(reporting_revision=meta['reporting_revision'], controls=sha(ROOT / 'REVIEWED_SHA256SUMS'),
                prior_manifest=meta['prior_manifest_sha256'], environment=env, settings=meta['scientific_settings'])


def prepare(meta, serial, attempt):
    networks, snapshots, inventory, accounting = serial.preflight()
    env = environment()
    serial.stage('scheduling-tests', [sys.executable, '-W', 'error', '-m', 'unittest', 'discover',
        '-s', ROOT, '-p', 'test_ptc_spod_array.py'], attempt)
    serial.stage('diagnostic-tests', [sys.executable, '-W', 'error', '-m', 'unittest', 'discover',
        '-s', serial.TOOLS, '-p', 'test_ptc_spod*.py'], attempt)
    data = dict(binding=binding(meta, env), networks=networks, inventory=str(inventory), accounting=accounting,
        serial_binding=dict(expected_sha256=serial.META['expected_sha256'],
            reporting_revision=serial.META['reporting_revision'], environment=env,
            settings=serial.META['scientific_settings']))
    previous = ROOT / 'PREPARED.json'
    if previous.exists():
        require(sealed(previous) == data, 'shared preparation changed; cannot reuse completions')
    sealed(previous, data)
    write(attempt / 'source.json', serial.source_state())


def prepared(meta, serial):
    data = sealed(ROOT / 'PREPARED.json')
    require(data['binding'] == binding(meta, environment()), 'prepared environment or controls changed')
    serial.check_tools()
    serial.source_state()
    return data


def select_network(data, number):
    found = [n for n in data['networks'] if n['network'] == number]
    require(len(found) == 1, 'unbound network assignment')
    return found[0]


def snapshot_for(serial, number):
    path = serial.VERIFY / f'verification/snapshots/workers-1/network{number}/snapshot.json'
    require(sha(path) == serial.EXPECTED['baseline_snapshots'][str(number)], 'verified snapshot changed')
    return json.loads(path.read_text())


def verify_record(record, data, serial, network, snapshot):
    require(record['network'] == network['network'] and record['binding'] == data['binding'], 'completion binding changed')
    output = Path(record['output'])
    allowed = (ROOT / 'runs').resolve() if record['origin'] == 'parallel' else (PRIOR / 'runs').resolve()
    require(record['origin'] in ('parallel', 'serial-reused') and output.resolve().is_relative_to(allowed), 'invalid output location')
    require({str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()} == set(record['outputs_sha256']), 'output inventory changed')
    serial.check_inputs({str(output / p): digest for p, digest in record['outputs_sha256'].items()})
    serial.validate_result(json.loads((output / 'result.json').read_text()), network, snapshot)
    if record['origin'] == 'serial-reused':
        require(sha(PRIOR / f'completed/network{network["network"]}.json') == record['serial_completion_sha256'], 'serial completion changed')
    return record


def run_network(meta, serial, data, number, attempt, generation):
    network = select_network(data, number)
    snapshot = snapshot_for(serial, number)
    serial.require(snapshot['root'] == network['output'], 'publication binding changed')
    serial.require(sha(network['input']['path']) == network['input']['sha256'], 'config changed')
    done = ROOT / f'completed/network{number}.json'
    if done.exists():
        record = verify_record(sealed(done), data, serial, network, snapshot)
        return dict(network=number, origin='parallel-checkpoint-reused', output=record['output'])
    prior_done = PRIOR / f'completed/network{number}.json'
    if prior_done.exists():
        # This branch can only validate an existing serial completion. It cannot
        # invoke a missing serial diagnostic or modify that packet.
        record = serial.resume_or_run(network, snapshot, data['serial_binding'], attempt)
        record = dict(record, binding=data['binding'], origin='serial-reused', serial_completion_sha256=sha(prior_done))
    else:
        require(not list((ROOT / 'submissions' / generation / 'failures').glob('*.json')), 'another task failed; no new diagnostic launched')
        output = ROOT / 'runs' / generation / attempt.name / f'network{number}'
        output.parent.mkdir(parents=True, exist_ok=True)
        serial.stage(f'network{number}', [sys.executable, serial.TOOLS / 'run_ptc_spod.py',
            '--input', network['output'], '--config', network['input']['path'], '--output', output,
            '--fft-seconds', '2', '8', '--pool-seconds', '120'], attempt)
        serial.validate_result(json.loads((output / 'result.json').read_text()), network, snapshot)
        serial.check_tools()
        record = dict(network=number, binding=data['binding'], output=str(output), origin='parallel',
            outputs_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    verify_record(record, data, serial, network, snapshot)
    sealed(done, record)
    return dict(network=number, origin=record['origin'], output=record['output'])


def task_outcomes(meta, data, attempt, generation):
    jobs = json.loads((ROOT / 'submissions' / generation / 'JOBS.json').read_text())
    array = jobs['networks']
    expected = {n['network'] for n in data['networks']}
    finals = {}
    for path in (ROOT / 'attempts' / generation).glob('network-*/FINAL.json'):
        final = json.loads(path.read_text())
        n = final['network']
        require(n in expected and n not in finals, 'unexpected or duplicate current task completion')
        require(final['mode'] == 'network' and final['disposition'] == 'PASS' and
                final['array_job'] == array and final['reporting_revision'] == meta['reporting_revision'] and
                final['runtime_revision'] == meta['runtime_revision'], 'current task did not pass')
        finals[n] = final
    require(set(finals) == expected, 'missing current task success; checkpoints alone cannot establish campaign PASS')
    # Slurm termination may precede accounting publication. Allow a bounded
    # refresh, but never interpret missing/stale accounting as success.
    rows = {}
    for retry in range(6):
        text = subprocess.check_output(['sacct', '--array', '--allocations', '--noheader', '--parsable2',
            '--jobs=' + array, '--format=JobID,JobIDRaw,State,ExitCode'], text=True)
        with (attempt / 'array-accounting.log').open('a') as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + '\n' + text)
        rows = {parts[0]: parts[1:] for line in text.splitlines() if len(parts := line.strip().split('|')) == 4}
        if all(rows.get(f'{array}_{n}') == [finals[n]['job'], 'COMPLETED', '0:0'] for n in expected):
            write(attempt / 'array-outcomes.json', rows)
            return
        if retry < 5:
            time.sleep(10)
    raise RuntimeError('Slurm array task outcomes missing or not COMPLETED/0:0: ' + repr(rows))


def finalize(meta, serial, data, attempt, generation):
    task_outcomes(meta, data, attempt, generation)
    records = []
    for network in data['networks']:
        number = network['network']
        record = verify_record(sealed(ROOT / f'completed/network{number}.json'), data, serial,
            network, snapshot_for(serial, number))
        records.append(dict(network=number, output=record['output']))
    require(not list((ROOT / 'submissions' / generation / 'failures').glob('*.json')), 'task failures prevent campaign PASS')
    view = ROOT / 'accounting'
    view.mkdir(exist_ok=True)
    targets = {f'workers-{n}': serial.CAMPAIGN / f'workers-{n}' for n in (1, 2, 4, 8, 12)}
    targets.update({f'diagnostic-network{n["network"]}': Path(n['output']) for n in records})
    for name, target in targets.items():
        link = view / name
        if link.is_symlink():
            require(link.resolve() == target.resolve(), 'accounting target changed')
        else:
            require(not link.exists(), 'unexpected accounting path')
            link.symlink_to(target, target_is_directory=True)
    campaign = json.loads((serial.CAMPAIGN / 'CAMPAIGN.json').read_text())
    campaign.update(comparisons='all44 exact comparisons verified in job64656301', diagnostics=records,
        state='completed-equivalent-with-partial-results', repeat_unavailable='optional repeats not performed')
    write(view / 'CAMPAIGN.json', campaign)
    serial.check_inputs(data['accounting'])
    serial.stage('population-and-cost-summary', [sys.executable, serial.TOOLS / 'summarize_successor_campaign.py',
        '--campaign', view, '--inventory', data['inventory']], attempt)
    serial.check_inputs(data['accounting'])
    serial.check_tools()
    write(attempt / 'source-after.json', serial.source_state())
    write(ROOT / 'STATUS.json', dict(state='PASS', generation=generation, networks_completed=records))


def collect():
    # Compact publication only. Original data and large numerical mode arrays stay put.
    files = {}
    for pattern in ('PREPARATION.json', 'INDEPENDENT_REVIEW.json', 'LOCAL_CHECKS.json', 'SHA256SUMS',
                    'PREPARED.json', 'STATUS.json', 'completed/*.json', 'submissions/*/JOBS.json',
                    'submissions/*/failures/*.json', 'attempts/*/*/*.json', 'attempts/*/*/*.txt',
                    'attempts/*/*/*.log', 'accounting/SUMMARY.json', 'accounting/CAMPAIGN.json'):
        files.update({str(p.relative_to(ROOT)): p for p in ROOT.glob(pattern) if p.is_file()})
    for completion in ROOT.glob('completed/network*.json'):
        record = sealed(completion)
        output = Path(record['output'])
        require(any(output.resolve().is_relative_to((p / 'runs').resolve()) for p in (ROOT, PRIOR)), 'invalid collected output path')
        for pattern in ('result.json', '*.png', 'temporal-supplement/*.png'):
            for p in output.glob(pattern):
                files[f'diagnostics/network{record["network"]}/' + str(p.relative_to(output))] = p
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    archive = ROOT / f'results-parallel-{stamp}.tar.gz'
    temp = archive.with_suffix('.tmp')
    manifest = []
    with tarfile.open(temp, 'w:gz') as tar:
        for name, p in sorted(files.items()):
            require(p.stat().st_size <= 64 * 1024**2, 'oversized compact report')
            content = p.read_bytes()
            item = tarfile.TarInfo(name); item.size = len(content)
            tar.addfile(item, io.BytesIO(content))
            manifest.append(hashlib.sha256(content).hexdigest() + '  ./' + name + '\n')
        content = ''.join(manifest).encode(); item = tarfile.TarInfo('COMPACT_SHA256SUMS'); item.size = len(content)
        tar.addfile(item, io.BytesIO(content))
    digest = sha(temp)
    write(archive.with_suffix('.ready.json'), dict(archive=archive.name, sha256=digest,
        scope='partial or complete evidence; require archive, matching digest, STATUS and FINAL'))
    print('PUBLISHING ' + str(archive), flush=True)
    # The archive is discoverable only after every other publication operation
    # succeeds. A marker without its matching archive is incomplete evidence.
    temp.replace(archive)


def submit(meta):
    with lock(ROOT / 'submission.lock'):
        # Check all prior submissions rather than relying on one overwritten job ID.
        tracked = {str(v) for p in ROOT.glob('submissions/*/JOBS.json')
                   for v in json.loads(p.read_text()).values()}
        queued = subprocess.check_output(['squeue', '--noheader', '--user', str(os.getuid()), '--format=%i'], text=True)
        require(not tracked.intersection(x.split('_')[0] for x in queued.split()), 'a prior parallel submission is still active')
        generation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        directory = ROOT / 'submissions' / generation; directory.mkdir(parents=True)
        jobs = {}
        def queue(name, args, mode):
            command = ['sbatch', '--parsable', '--chdir=' + str(ROOT), '--job-name=ptc-' + name,
                '--output=' + str(directory / (name + '-%A_%a.out')),
                '--error=' + str(directory / (name + '-%A_%a.err'))] + args + [str(ROOT / 'run.sbatch'), mode, generation]
            job = subprocess.check_output(command, text=True).strip().split(';')[0]
            require(job.isdigit(), 'unexpected sbatch response')
            jobs[name] = job; write(directory / 'JOBS.json', jobs)
            print(name + ': ' + job, flush=True)
            return job
        try:
            prep = queue('prepare', ['--time=02:00:00'], 'prepare')
            array = queue('networks', ['--dependency=afterok:' + prep, '--kill-on-invalid-dep=yes',
                '--array=' + ','.join(map(str, meta['networks'])) + '%11'], 'network')
            queue('finalize', ['--time=02:00:00', '--dependency=afterany:' + array], 'finalize')
        except BaseException:
            # These are only job IDs created by this exact failed submission.
            if jobs:
                subprocess.run(['scancel', *jobs.values()], check=False)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('submit', 'prepare', 'network', 'finalize', 'collect'))
    parser.add_argument('generation', nargs='?')
    args = parser.parse_args()
    meta = controls()
    if args.mode == 'submit':
        submit(meta); return
    if args.mode == 'collect':
        collect(); return
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdigit(), 'requires a Slurm compute allocation')
    require(args.generation and args.generation.isalnum(), 'invalid submission identity')
    number = int(os.environ.get('SLURM_ARRAY_TASK_ID', '-1'))
    name = args.mode + (f'-network{number}' if args.mode == 'network' else '') + '-' + job
    attempt = ROOT / 'attempts' / args.generation / (name + '-' + datetime.now(timezone.utc).strftime('%H%M%S%f'))
    attempt.mkdir(parents=True)
    write(attempt / 'execution.json', dict(host=platform.node(), platform=platform.platform(),
        cpu_allocation=os.environ.get('SLURM_CPUS_PER_TASK'), memory_allocation=os.environ.get('SLURM_MEM_PER_NODE'),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        thread_limits={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}))
    def stop(*_):
        raise KeyboardInterrupt('task interrupted; completed checkpoints preserved')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    code = 1
    try:
        with lock(PRIOR / 'diagnostics.lock', shared=True, wait_seconds=90):
            serial = load_prior(meta)
            with lock(ROOT / 'campaign.lock', shared=args.mode == 'network'):
                if args.mode == 'prepare':
                    write(ROOT / 'STATUS.json', dict(state='preparing', generation=args.generation))
                    prepare(meta, serial, attempt)
                    write(ROOT / 'STATUS.json', dict(state='prepared', generation=args.generation))
                else:
                    data = prepared(meta, serial)
                    if args.mode == 'network':
                        with lock(ROOT / f'locks/network{number}.lock'):
                            result = run_network(meta, serial, data, number, attempt, args.generation)
                            write(attempt / 'result.json', result)
                    else:
                        finalize(meta, serial, data, attempt, args.generation)
        code = 0
    except BaseException:
        reason = traceback.format_exc(); (attempt / 'failure.txt').write_text(reason)
        write(ROOT / 'submissions' / args.generation / 'failures' / (name + '.json'), dict(reason=reason, attempt=str(attempt)))
        traceback.print_exc()
        if args.mode == 'finalize':
            write(ROOT / 'STATUS.json', dict(state='FAIL', generation=args.generation, reason=reason))
    finally:
        final = dict(disposition='PASS' if code == 0 else 'FAIL', mode=args.mode,
            network=number if args.mode == 'network' else None, job=job,
            array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),
            reporting_revision=meta['reporting_revision'], runtime_revision=meta['runtime_revision'])
        write(attempt / 'FINAL.json', final)
        if args.mode == 'finalize':
            try:
                collect()
            except BaseException:
                reason = traceback.format_exc()
                (attempt / 'collection-failure.txt').write_text(reason)
                final['disposition'] = 'FAIL'
                final['collection_failure'] = reason
                write(attempt / 'FINAL.json', final)
                write(ROOT / 'STATUS.json', dict(state='FAIL', generation=args.generation, reason=reason))
                raise
    raise SystemExit(code)


if __name__ == '__main__':
    main()
