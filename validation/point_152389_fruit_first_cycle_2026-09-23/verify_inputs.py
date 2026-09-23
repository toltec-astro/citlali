"""Read-only POINT 152389 first-cycle input and model-domain inventory.

This does not define or execute a sky-to-timestream projection or PTC replay.
Counts refer to the existing MAP operator, not the larger pre-PTC domain.
"""
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO = Path('/private/tmp/citlali-point-additive-20260923')
sys.path.insert(0, str(REPO / 'tools/timestream_successor'))
from inspect_forward_map import digest, read_yaml, require
from point_empirical_io import load_parent
from point_additive_diagnostics import verify_companion, save_json

BASE = Path('/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs')
PARENT = BASE / 'successor-point-152389-network-rescue-20260922'
ADMISSION = BASE / 'successor-point-152389-admission-prototype-20260923'
ADDITIVE = BASE / 'successor-point-152389-additive-uncertainty-20260923/companion-r2'
OUT = Path(__file__).parent


def run():
    start = time.perf_counter()
    cpu = time.process_time()
    measurement = PARENT / 'diagnostics/source/source-measurements.json'
    saved = json.loads(measurement.read_text())
    hashes = {a['array']: read_yaml(PARENT / 'map-pass/products' / a['array'] / 'bundle.yaml')['files']['map.fits'] for a in saved['arrays']}
    admission = verify_companion(ADMISSION / 'companion-r2', hashes)
    additive = verify_companion(ADDITIVE, hashes)
    require(admission['manifest_sha256'] == '42e656d6fedefc1a47599333983bc5a7db31acb42e78be0ec3449a80fe577f71', 'foreign admission generation')
    require(additive['manifest_sha256'] == 'd650deddab4886d9ad386183a6425e5309643cab8effb82fd645106dde3f38c8', 'foreign additive generation')
    require(digest(ADMISSION / 'ASSESSMENT.md') == 'b3a779e1ee31ee92079fd293a9ab516b38e6a0e2c4d03089fa5c05cd917f6a47', 'foreign assessment')
    report = dict(scope='identity verification and existing-MAP support inventory only; no feedback or zero-model PTC replay', admission=admission['manifest_sha256'], additive=additive['manifest_sha256'], measurement_sha256=digest(measurement), arrays={})
    parents = {}
    for desc in saved['arrays']:
        array = desc['array']
        root = PARENT / 'map-pass/products' / array
        full, h, bundle, plan, measured, parameters, verification = load_parent(root, desc)
        candidate = ADMISSION / 'companion-r2' / array / 'candidate-reference.fits'
        with fits.open(candidate, checksum=True, memmap=False) as hdus:
            f = hdus['CANDIDATE'].data.copy()
            valid = hdus['VALID'].data.astype(bool)
            selected = hdus['ADMITTED'].data.astype(bool)
            ch = hdus[0].header.copy()
        rr, cc = np.indices(f.shape)
        xx = (cc + 1 - ch['CRPIX1']) * ch['CDELT1'] * 3600
        yy = (rr + 1 - ch['CRPIX2']) * ch['CDELT2'] * 3600
        domain = np.hypot(xx, yy) <= 40
        require(np.array_equal(valid, domain), 'actual candidate has unexpected missing interior support')
        require(np.all(np.isnan(f[~valid])) and np.all(f[valid & ~selected] == 0), 'candidate zero/unavailable distinction changed')
        shape = full['PRIMARY'].shape
        mr, mc = np.indices(shape)
        inside = np.hypot((mc + 1 - h['CRPIX1']) * h['CDELT1'] * 3600, (mr + 1 - h['CRPIX2']) * h['CDELT2'] * 3600) <= 40
        per_network = {}
        op = np.memmap(root / 'operator-pixel.i32', dtype='<i4', mode='r')
        for col in plan['ordered_column_domain']:
            pub = read_yaml(root / col['publication_file'])
            source = Path(pub['source_PTC_root'])
            if str(source) not in parents:
                receipt = read_yaml(source / 'receipt.yaml')
                require(digest(source / 'receipt.yaml') == pub['source_PTC_receipt_sha256'], 'PTC receipt changed')
                inventory = []
                for segment in receipt['segments']:
                    for name, sha in segment['files'].items():
                        require(digest(source / name) == sha, 'saved PTC component changed')
                    inventory.append({k: segment[k] for k in ('network', 'scan', 'scheduled_times', 'output_detectors', 'converged', 'retained')})
                parents[str(source)] = dict(receipt_sha256=pub['source_PTC_receipt_sha256'], method=receipt['method'], rank=receipt['rank'], centering=receipt['centering'], segments=inventory)
            pixels = op[col['first_input_row']:col['first_input_row'] + col['input_rows']]
            pixels = pixels[pixels >= 0]
            stats = per_network.setdefault(str(pub['network']), dict(map_members=0, inside_candidate_disk=0, outside_candidate_disk=0))
            within = int(np.count_nonzero(inside.ravel()[pixels]))
            stats['map_members'] += len(pixels)
            stats['inside_candidate_disk'] += within
            stats['outside_candidate_disk'] += len(pixels) - within
        report['arrays'][array] = dict(map_sha256=hashes[array], candidate_sha256=digest(candidate), shape=list(f.shape), valid_pixels=int(valid.sum()), nonzero_admitted_pixels=int(selected.sum()), intentional_zero_pixels=int((valid & ~selected).sum()), unavailable_pixels_in_stored_square=int((~valid).sum()), no_model_definition_beyond_stored_square=True, networks=per_network, parent_verification=verification, source_measurement=measured)
        save_json(OUT / 'verification.json', report)
        print(array, 'verified', flush=True)
    report['PTC_parents'] = parents
    report['runtime'] = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report['script_sha256'] = digest(__file__)
    save_json(OUT / 'verification.json', report)
    print(json.dumps(report['runtime']), flush=True)


if __name__ == '__main__':
    run()
