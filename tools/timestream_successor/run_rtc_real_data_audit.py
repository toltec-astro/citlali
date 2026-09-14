#!/usr/bin/env python3
"""Run inert native spectral Learn on the exact preserved transient census inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-campaign', type=Path, required=True)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, choices=(1,2), default=1)
    p.add_argument('--only', nargs='*', help='Explicit observation-network pilot stems')
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    exe = a.executable.resolve(strict=True)
    exehash = digest(exe)
    invocations = [json.loads(s) for s in (a.previous_campaign/'invocations.jsonl').read_text().splitlines()]
    invocations.sort(key=lambda r: (r['rows'] > 20000, r['observation'], r['network']))
    def run(old):
        stem = f"{old['observation']}-{old['network']:02d}"
        raw, tune, manifest = map(Path, old['command'][1:4])
        output = a.output/stem
        command = [str(exe), str(raw), str(tune), str(manifest), str(output), str(a.selection.resolve(strict=True))]
        before = [(x.stat().st_size, x.stat().st_mtime_ns) for x in (raw, tune, manifest)]
        started = time.monotonic()
        with (a.output/f'{stem}.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        after = [(x.stat().st_size, x.stat().st_mtime_ns) for x in (raw, tune, manifest)]
        receipt = json.loads((output/'receipt.json').read_text()) if (output/'receipt.json').exists() else None
        prior = json.loads((a.previous_campaign/stem/'receipt.json').read_text())
        binding = receipt is not None and all(receipt[k] == prior[k] for k in ('observation', 'network', 'rows', 'channels', 'raw_sha256', 'tune_sha256', 'manifest_sha256'))
        entry = dict(observation=old['observation'], network=old['network'], command=command,
                     executable_sha256=exehash, selection_sha256=digest(a.selection),
                     previous_receipt_sha256=digest(a.previous_campaign/stem/'receipt.json'),
                     wall_seconds=time.monotonic()-started, max_rss_bytes=usage.ru_maxrss,
                     resource_units='macOS ru_maxrss bytes', exit_status=process.returncode,
                     exact_prior_input_binding=binding, input_metadata_unchanged=before == after,
                     receipt=receipt)
        return entry

    selected = [r for r in invocations if not a.only or f"{r['observation']}-{r['network']:02d}" in a.only]
    entries = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool, (a.output/'invocations.jsonl').open('w') as ledger:
        for entry in pool.map(run, selected):
            entries.append(entry)
            ledger.write(json.dumps(entry, allow_nan=False)+'\n'); ledger.flush()
            good = entry['exact_prior_input_binding'] and entry['exit_status']==0 and entry['input_metadata_unchanged']
            print(entry['observation'], entry['network'], 'PASS' if good else 'FAIL', round(entry['wall_seconds'],2), flush=True)
            if not good:
                raise RuntimeError('Failed run or exact input binding; inspect invocation ledger')
    if not entries:
        raise RuntimeError('No selected invocations')
    (a.output/'summary.json').write_text(json.dumps(dict(files=len(entries), wall_seconds=sum(x['wall_seconds'] for x in entries), max_rss_bytes=max(x['max_rss_bytes'] for x in entries), executable_sha256=exehash), indent=2)+'\n')


if __name__ == '__main__':
    main()
