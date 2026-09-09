#!/usr/bin/env python3
"""Local, inert RTC census driver over an explicit header-inventory JSON.

All science runs in the C++ executable. This program selects exact input copies,
records per-invocation resource/provenance/failure evidence, and reconciles counts.
It does not submit jobs, fetch inputs, issue APTs, change policy, or merge refs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trial-half-width-seconds', type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists: preserve prior campaigns')
    args.output.mkdir(parents=True)
    exe = args.executable.resolve(strict=True)
    executable_hash = digest(exe)
    records = json.loads(args.inventory.read_text())['records']
    candidates = [r for r in records if r.get('apt_manifests') and r.get('fitreports')]
    deferred = [r for r in records if r not in candidates]
    write_json(args.output/'input-inventory.json', {'records': records})
    write_json(args.output/'deferred-inputs.json', deferred)
    # Short observations first, then complete long observations. Serial network
    # processing bounds resident memory and preserves each full native run.
    candidates.sort(key=lambda r: (r['rows'] > 20000, r['observation'], r['network']))
    entries = []
    with (args.output/'invocations.jsonl').open('w') as ledger:
        for r in candidates:
            obs, nw = r['observation'], r['network']
            stem = f'{obs}-{nw:02d}'
            raw = Path(r['path'])
            tunes = {digest(p): Path(p) for p in r['fitreports']}
            if len(tunes) != 1:
                entry = dict(observation=obs, network=nw, status='input_unavailable', reason='conflicting Tune copies')
                entries.append(entry)
                ledger.write(json.dumps(entry)+'\n'); ledger.flush()
                continue
            tune = next(iter(tunes.values()))
            manifests = sorted(r['apt_manifests'], key=lambda p: (
                0 if '/citlali-validation/v2/' in p else 1,
                'fluxcal' in p, '/point/apts/' not in p, p))
            manifest = Path(manifests[0])
            target = args.output/stem
            log = args.output/f'{stem}.log'
            command = [str(exe), str(raw), str(tune), str(manifest), str(target), str(args.trial_half_width_seconds)]
            before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (raw, tune, manifest)}
            started = time.monotonic()
            with log.open('wb') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
            elapsed = time.monotonic()-started
            after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (raw, tune, manifest)}
            entry = dict(observation=obs, network=nw, command=command, exit_status=process.returncode,
                         wall_seconds=elapsed, max_rss_bytes=usage.ru_maxrss,
                         resource_units='macOS ru_maxrss bytes', executable_sha256=executable_hash,
                         inputs_metadata_unchanged=before == after, channels=r['channels'], rows=r['rows'])
            receipt_path = target/'receipt.json'
            if process.returncode == 0 and before == after and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                detectors = [json.loads(line) for line in (target/'detectors.jsonl').open()]
                fits_count = sum(1 for _ in (target/'fits.jsonl').open())
                assert len(detectors) == r['channels'] == receipt['channels']
                assert fits_count == receipt['candidate_edges'] == sum(sum(d['candidate_counts']) for d in detectors)
                assert sum(1 for _ in (target/'detector-bindings.jsonl').open()) == len(detectors)
                assert all(d['rows'] == r['rows'] and d['pair_screening_excluded_rows'] <= d['rows'] for d in detectors)
                entry.update(status='completed', receipt=receipt, fit_rows=fits_count)
                write_json(target/'SHA256.json', {p.name: digest(p) for p in sorted(target.iterdir()) if p.is_file()})
            else:
                entry.update(status='failed', failure_tail=log.read_text(errors='replace')[-3000:])
            entries.append(entry)
            ledger.write(json.dumps(entry, allow_nan=False)+'\n'); ledger.flush()
            print(f"{stem}: {entry['status']}; {elapsed:.1f}s; {usage.ru_maxrss/2**20:.0f} MiB", flush=True)
    assert digest(exe) == executable_hash, 'executable changed during census'
    complete = [e for e in entries if e['status'] == 'completed']
    summary = dict(status='complete_with_explicit_input_limits' if len(complete)==len(candidates) else 'incomplete_failures_recorded',
                   attempted_network_files=len(candidates), completed_network_files=len(complete),
                   completed_channel_timestreams=sum(e['channels'] for e in complete),
                   completed_detector_samples=sum(e['channels']*e['rows'] for e in complete),
                   candidate_edges=sum(e['fit_rows'] for e in complete),
                   deferred_network_files=len(deferred), deferred_channels=sum(e['channels'] for e in deferred),
                   elapsed_network_wall_seconds=sum(e.get('wall_seconds',0) for e in entries),
                   peak_child_rss_bytes=max((e.get('max_rss_bytes',0) for e in entries), default=0),
                   inventory_sha256=digest(args.inventory), executable_sha256=executable_hash,
                   trial_half_width_seconds=args.trial_half_width_seconds,
                   retained_limits=['No hard event classification or physical affected-data fraction',
                                    'Source membership unavailable; no optical rejection or Apply',
                                    'Input preflight defers missing canonical detector bindings',
                                    'Candidate edges may represent the same physical event; counts are not event rates'])
    write_json(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
