#!/usr/bin/env python3
"""Ordinary POINT application adapter; TolTECA owns input preparation and merging.

The native executable remains the scientific stage owner. This boundary reads
one fully merged request, binds inputs, runs PTC then MAP, and publishes only
complete requested products. It never invokes a historical reduction fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

from configured_point_config import load_request


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(*args, **kwargs):
    from configured_point_inputs import prepare as implementation
    return implementation(*args, **kwargs)


def publish_maps(*args, **kwargs):
    from configured_point_products import publish_maps as implementation
    return implementation(*args, **kwargs)

REPO = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def bind(path):
    return {'path': str(path.resolve()), 'sha256': digest(path)}


def reserve_reduction(output):
    """Create only a new child; never reuse or remove a prior reduction."""
    output.mkdir(parents=True, exist_ok=True)
    ids = [int(p.name[4:]) for p in output.iterdir()
           if re.fullmatch(r'redu\d+', p.name)]
    number = max(ids, default=-1) + 1
    while True:
        candidate = output / f'redu{number:02d}'
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            number += 1


def native_call(binary, request, dest, *, terminal, workers=1, method=None, verbose=False):
    dest.mkdir()
    config = {'schema': 'citlali-development-v1', 'input': bind(request),
              'output': str(dest / 'products'), 'terminal': terminal,
              'network_workers': workers}
    if method is not None:
        # Per-array ranks live in the effective YAML. A global rank would
        # override all arrays, so it is deliberately absent here.
        config['ptc'] = {'method': method}
    path = dest / 'development.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE')
    started = time.monotonic()
    log_path = dest / 'native.log'
    with log_path.open('w') as log:
        result = subprocess.run([str(binary), '-l', 'debug' if verbose else 'warning', str(path)],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    # No process sampler or repeated output-tree inventory in ordinary mode.
    # Preserve warnings/errors in the user-facing launch log as well as disk.
    tail = []
    error_lines = []
    with log_path.open(errors='replace') as log:
        for line in log:
            tail.append(line.rstrip())
            tail = tail[-20:]
            if re.search(r'\[(error|critical)\]', line, re.I):
                error_lines.append(line.rstrip())
            if verbose or re.search(r'\[(warning|warn|error|critical)\]', line, re.I):
                print(line.rstrip(), flush=True)
    record = {'terminal': terminal, 'seconds': elapsed, 'return_code': result.returncode,
              'request': bind(path), 'unexpected_error_lines': error_lines}
    write_json(dest / 'stage.json', record)
    if result.returncode or error_lines:
        raise RuntimeError(f'{terminal} failed; see {log_path}\n' + '\n'.join(tail))
    return record


def map_request(prepared, upstream, destination, owner, request):
    networks = []
    for nw in prepared['networks']:
        root = upstream / 'products' / f'network{nw}'
        receipt = root / 'donor-continuity/ptc/receipt.yaml'
        result = yaml.safe_load(receipt.read_text())
        if result['failed_fits'] or not result['available'] or any(
            not s['converged'] or s.get('application_failed_times', 0)
            for s in result['segments']
        ):
            raise RuntimeError(f'network {nw}: required PTC result unavailable')
        networks.append({'root': str(root),
                         'input': bind(prepared['effective_config'].parent / f'network{nw}-input.json'),
                         'receipt_sha256': digest(root / 'receipt.yaml'),
                         'ptc_receipt_sha256': digest(receipt)})
    value = {'schema': 'citlali-map-saved-ptc-v1', 'owner_binding': bind(owner),
             'pixel_size_arcsec': request.pixel_size_arcsec,
             'coverage_cut': request.coverage_cut, 'networks': networks}
    write_json(destination, value)
    return destination


def release_transport(work):
    """Drop only generated binary transport after all its consumers complete.

    Retain native receipts with their payload digests. They describe execution,
    not a promise that an optional replay payload remains on disk.
    """
    count = size = 0
    for root in (work / 'upstream' / 'products', work / 'map' / 'products'):
        for path in root.rglob('*'):
            if path.suffix not in {'.f64', '.u8', '.u64', '.i32', '.i64'}:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(work.resolve()):
                raise RuntimeError(f'unsafe generated transport path: {path}')
            if path.is_file():
                size += path.stat().st_size
                path.unlink()
                count += 1
    return {'files': count, 'bytes': size, 'native_receipts_retained': True,
            'replay_payloads_retained': False}


def execute(config_path, native, *, launch_dir=None):
    launch_dir = Path.cwd() if launch_dir is None else launch_dir
    config_path = config_path.resolve()
    native = native.resolve()
    received = yaml.safe_load(config_path.read_text())
    request = load_request(received, launch_dir)
    # Reject unsupported requests before creating any reduction output.
    version = subprocess.check_output([str(native), '--version'], text=True).strip()
    reduction = reserve_reduction(request.output_dir)
    work = reduction / '.point-work'
    work.mkdir()
    record = {'schema': 'citlali-configured-point-run-v1', 'status': 'running',
              'received_config': bind(config_path), 'executable': bind(native),
              'version': version, 'launch_directory': str(launch_dir.resolve()),
              'plan': asdict(request), 'stages': [], 'products': [],
              'calibration_eligible': False,
              'downstream_legacy_image_analysis_compatible': False}
    started = time.monotonic()
    shutil.copyfile(config_path, work / 'received.yaml')
    write_json(reduction / 'RUN.json', record)
    try:
        drop_enabled = request.effective_config['successor']['rtc'].get('disturbance_exclusion', 'disabled') != 'disabled'
        if drop_enabled:
            from configured_point_drops import collect, console_summary
            # Preparation can fail before the exact detector inventory exists.
            # Preserve unknown populations rather than manufacturing zero losses.
            summary = collect(work / 'upstream/products', None, [])
            summary['evaluation_state'] = 'not_evaluated'
            path = reduction / 'detector-drops.json'
            write_json(path, summary)
            record['detector_drops'] = bind(path)
        try:
            prepared = prepare(request.effective_config, work / 'inputs',
                               launch_dir=launch_dir, rtc_profiles=request.rtc_profiles)
        except Exception:
            if drop_enabled:
                print(console_summary(summary, path), flush=True)
            raise
        record['inputs'] = {k: v for k, v in prepared.items() if k != 'bindings'}
        write_json(work / 'input-bindings.json', prepared['bindings'])
        print(f"POINT {prepared['observation']}: {len(prepared['networks'])} networks, "
              f"all three arrays, {request.workers} workers", flush=True)
        try:
            record['stages'].append(native_call(native, prepared['observation_request'], work / 'upstream',
                terminal='ptc', workers=request.workers, method=request.cleaning_method, verbose=request.verbose))
        finally:
            if drop_enabled:
                from configured_point_drops import collect, console_summary
                summary = collect(work / 'upstream/products', prepared['observation'], prepared['networks'])
                path = reduction / 'detector-drops.json'
                write_json(path, summary)
                record['detector_drops'] = bind(path)
                print(console_summary(summary, path), flush=True)
        if drop_enabled and summary['evaluation_state'] != 'complete':
            raise RuntimeError('required detector-drop evaluation/application is incomplete')
        owner = REPO / 'data/development/bright_point_base_v1/map-owner.json'
        q = map_request(prepared, work / 'upstream', work / 'map-request.json', owner, request)
        record['stages'].append(native_call(native, q, work / 'map', terminal='map', verbose=request.verbose))
        # Publish into hidden staging first. A failed required product never
        # leaves a discoverable, apparently complete observation directory.
        staged = work / 'publish'
        raw = staged / 'raw'
        record['products'] = publish_maps(work / 'map' / 'products', raw,
            prepared['observation'], request.arrays,
            fit_diagnostics=request.fit_diagnostics, verbose=request.verbose)
        if not request.retain_intermediates:
            record['transport_retention'] = release_transport(work)
        else:
            record['transport_retention'] = {'replay_payloads_retained': True}
        published = reduction / f"{prepared['observation']:06d}"
        staged.rename(published)
        # Paths reported by publisher must follow the published directory.
        record['products'] = json.loads(json.dumps(record['products'], default=str).replace(str(staged), str(published)))
        record.update(status='completed', seconds=time.monotonic() - started,
                      scientific_scope='forward POINT maps and qualified diagnostics; no flux-calibration table')
        write_json(reduction / 'RUN.json', record)
        print(f'POINT complete: {published}', flush=True)
        return reduction
    except Exception as exc:
        record.update(status='failed', seconds=time.monotonic() - started, error=str(exc))
        write_json(reduction / 'RUN.json', record)
        raise


def default_config():
    # TolTECA supports a fully specified low_level dictionary. The explicit
    # profile migrates the old default tree into the new native request shape.
    result = yaml.safe_load((REPO / 'data/config.yaml').read_text())
    overlay = yaml.safe_load((REPO / 'data/configured_point_v1.yaml').read_text())
    def merge(dst, src):
        for k, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], value)
            else:
                dst[k] = value
    merge(result, overlay)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True, type=Path)
    parser.add_argument('-l', '--log-level', default='info')
    parser.add_argument('--version', action='store_true')
    parser.add_argument('--dump_config', action='store_true')
    parser.add_argument('config', type=Path, nargs='?')
    args, extra = parser.parse_known_args(argv)
    # Versioned APT admission remains the existing native protocol, including
    # stdin/stdout and nonzero exit status. No reduction is launched here.
    if extra and any(x.startswith('--canonical-apt-contract-') for x in extra):
        return subprocess.call([str(args.native), *extra])
    if extra:
        parser.error('unrecognized arguments: ' + ' '.join(extra))
    if args.version:
        return subprocess.call([str(args.native), '--version'])
    if args.dump_config:
        print(yaml.safe_dump(default_config(), sort_keys=False), end='')
        return 0
    if args.config is None:
        parser.error('a merged TolTECA YAML is required')
    try:
        execute(args.config, args.native)
    except Exception as exc:
        print(f'POINT failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
