"""Inspect saved successor PTC numerical inputs without issuing MAP admission.

This is a bounded v2-reader and prerequisite report, not a coefficient producer,
coordinate reconstruction, VAL evaluation, or MAP product. All arrays are read
one processing segment at a time. Original files remain immutable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect(root):
    root = Path(root).resolve()
    wall, cpu = time.perf_counter(), time.process_time()
    ledger = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        value = sha256(path)
        require(expected is None or value == expected, f'checksum mismatch: {path}')
        require(str(path) not in ledger or ledger[str(path)] == value, f'input changed: {path}')
        ledger[str(path)] = value
        return path

    def read_yaml(path, stop=None):
        path = bind(path)
        with path.open() as f:
            lines = []
            for line in f:
                if stop and line.startswith(stop + ':'):
                    break
                lines.append(line)
        return yaml.load(''.join(lines), Loader=yaml.CSafeLoader)

    ptcroot = root / 'donor-continuity/ptc'
    ptc = read_yaml(ptcroot / 'receipt.yaml')
    require(ptc.get('schema') == 'citlali-ptc-output-v2', 'actual successor PTC v2 required')
    calpath = root / 'donor-continuity/cal/receipt.yaml'
    bind(calpath, ptc['source_CAL_receipt_sha256'])
    cal = read_yaml(calpath)
    learn = read_yaml(root / 'learning-receipt.yaml', stop='spectra')
    require(cal.get('schema') == 'citlali-cal-output-v1', 'CAL parent schema')
    require(ptc['input_VAL_generation'] == cal['output_VAL_generation'], 'PTC/CAL VAL generation mismatch')
    require(ptc['rank'] > 0 and ptc['method'] in ('observed-als', 'pairwise-covariance'), 'PTC method/rank')
    require(ptc['unit'] == cal['unit'] == 'mJy/nominal-beam', 'PTC/CAL unit mismatch')
    require(ptc['centering'] == 'per-detector-eligible-arithmetic-mean-not-restored', 'unknown additive-reference state')
    require(ptc['processing_generation'], 'processing generation absent')
    detectors = learn['detectors']
    require(len(detectors) == len(cal['detectors']), 'detector domain mismatch')
    require(len({d['occurrence'] for d in detectors}) == len(detectors), 'duplicate detector occurrence')
    for d, c in zip(detectors, cal['detectors']):
        require(d['channel'] == c['channel'] and c['selected_row'].endswith(
            ':row=' + d['occurrence'].rsplit('row=', 1)[1]), 'exact selected detector mismatch')
    segments, retained, failed, max_bytes = [], 0, 0, 0
    prior_end = -1
    numerical_started = time.perf_counter()
    for index, segment in enumerate(ptc['segments']):
        require(segment['network'] == learn['network'], 'segment network mismatch')
        lo, hi = segment['native_interval']
        require(0 <= lo < hi <= learn['rows'] and lo >= prior_end, 'overlapping/invalid processing segments')
        prior_end = hi
        cols = segment['output_detector_indices']
        require(len(set(cols)) == len(cols) and all(type(d) is int and 0 <= d < len(detectors) for d in cols),
                'segment detector identity invalid')
        require(len(cols) == segment['output_detectors'], 'output detector shape mismatch')
        n = segment['scheduled_times']
        require(type(n) is int and n > 0, 'empty or invalid scheduled grid')
        stem = f'segment-{index}'
        arrays = {}
        # Every required component is checked, including fit/mean/input arrays;
        # only cleaned/causes/slots need materialization for this inspection.
        required = {stem + s for s in ('-input.f64', '-centered.f64', '-mean.f64', '-basis.f64',
                                      '-cleaned.f64', '-eligible.u8', '-causes.u8', '-slots.i64')}
        require(set(segment['files']) == required, 'PTC component inventory mismatch')
        for name, digest in segment['files'].items():
            require(Path(name).name == name, 'nonlocal component name')
            path = bind(ptcroot / name, digest)
            for suffix, dtype in (('-cleaned.f64', '<f8'), ('-causes.u8', 'u1'), ('-slots.i64', '<i8')):
                if name.endswith(suffix):
                    require(path.stat().st_size % np.dtype(dtype).itemsize == 0, 'partial numeric element')
                    arrays[suffix] = np.fromfile(path, dtype=dtype)
        values, causes, slots = (arrays[s] for s in ('-cleaned.f64', '-causes.u8', '-slots.i64'))
        require(values.size == causes.size == n * len(cols), 'PTC value/cause shape mismatch')
        require(slots.size == n and np.all(slots >= 0) and np.all(np.diff(slots) > 0), 'PTC slot domain invalid')
        require(np.all((causes & ~np.uint8(31)) == 0), 'unknown PTC cause')
        mask = causes == 0
        count = int(np.count_nonzero(mask))
        require(count == segment['retained'], 'PTC retained count mismatch')
        require(np.isfinite(values[mask]).all(), 'nonfinite retained PTC signal')
        require(segment['converged'] or count == 0, 'failed fit retained samples')
        retained += count
        failed += not segment['converged']
        max_bytes = max(max_bytes, sum(a.nbytes for a in arrays.values()) + mask.nbytes)
        segments.append(dict(segment=index, scan=segment['scan'], native_interval=[lo, hi],
                             scheduled=n * len(cols), retained=count, converged=segment['converged'],
                             stopping_reason=segment['stopping_reason']))
    require(retained == ptc['available'] and failed == ptc['failed_fits'], 'PTC aggregate mismatch')
    numerical_seconds = time.perf_counter() - numerical_started
    for path, digest in ledger.items():
        require(sha256(path) == digest, f'input changed during inspection: {path}')
    # v2 is an explicitly supported historical publication schema. Arbitrary
    # extra metadata cannot turn this inspector into an admission authority.
    blockers = [
        dict(requirement='002/044', cause='nominal_beam_identity_not_published', owner='CAL/source APT'),
        dict(requirement='005/044', cause='complete_source_frame_not_published', owner='AST',
             observed=cal.get('AST_frame')),
        dict(requirement='005/043', cause='same_RTC_sample_AST_coordinates_not_persisted', owner='AST'),
        dict(requirement='006', cause='selected_coefficient_publication_and_QC_not_present_in_v2', owner='PTC'),
        dict(requirement='004/010', cause='MAP_occurrence_admission_not_evaluated', owner='MAP/VAL'),
    ]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        peak *= 1024
    return dict(schema='citlali-map-input-inspection-v1', source_root=str(root),
                numerical_integrity='pass', MAP_admission='not_evaluated', MAP_readiness='blocked',
                verification_scope='declared PTC component hashes, cleaned/causes/slots and selected parent links; not full upstream or scientific conformance',
                upstream_build_identity=ptc.get('build_identity', 'not_published'),
                MAP_executed=False, observation=learn['observation'], network=learn['network'],
                arrays=sorted({d['array'] for d in cal['detectors']}), unit=ptc['unit'],
                requested_detectors=len(detectors), retained=retained, failed_fits=failed,
                method=ptc['method'], rank=ptc['rank'], segments=segments, blockers=blockers,
                optional_companions=dict(upstream_response='unavailable', covariance='unavailable',
                                         original_footprint_exposure='unavailable'),
                cost=dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                          numerical_inspection_seconds=numerical_seconds, peak_rss_bytes=peak,
                          segment_numeric_and_mask_bytes=max_bytes,
                          scope='input inspection only; no MAP or upstream processing cost'),
                verified_files=ledger, original_inputs_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='saved network output containing learning-receipt.yaml')
    parser.add_argument('--output', type=Path, required=True, help='new JSON evidence file (never a MAP product)')
    args = parser.parse_args()
    require(not args.output.exists(), 'inspection output exists')
    result = inspect(args.root)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(f"PTC integrity PASS; MAP BLOCKED: {result['retained']} retained samples, "
          f"{result['failed_fits']} failed fits. Evidence: {args.output}")
    return 2  # never promote this successful inspection into MAP readiness


if __name__ == '__main__':
    raise SystemExit(main())
