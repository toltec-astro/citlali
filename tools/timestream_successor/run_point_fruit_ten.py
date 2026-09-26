#!/usr/bin/env python3
"""Continue verified POINT152389 T1 through T10; preserve m10 without applying it."""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from point_fruit_trajectory import (bound, verified, verify_manifest, manifest, FixedMeasurements,
                                    load, candidate_guard, processing_guard, diagnose_cycle, compare_zero)
from point_additive_diagnostics import save_json, verify_companion
from inspect_forward_map import require, digest, read_yaml
from run_successor_observation import run as invoke

ARRAYS = ('a1100', 'a1400', 'a2000')


def make_request(array, cycle, candidate, previous, parents, owner, zero=False):
    require((zero and cycle == 0) or (not zero and 2 <= cycle <= 10), 'authorized cycle index is 2..10')
    return dict(schema='citlali-point-fruit-continuation-v1', method='point-152389-direct-ten-cycle-v1',
                mode='zero' if zero else 'reference', array=array, cycle_index=cycle,
                candidate=bound(candidate), previous_state=bound(previous),
                map_request=parents['map_request'], parent_plan=parents['parent_plan'], owner_binding=bound(owner))


def original_parents(first, array):
    """Recover immutable references from the already manifest-verified T1 request."""
    request = json.loads((first/f'cycle/reference-{array}.json').read_text())
    for key in ('map_request', 'parent_plan'):
        verified(request[key])
    # Follow the original native ledger, not just hashes of the ledger document.
    ledger = read_yaml(first/f'cycle/reference/{array}/products/input-manifest.yaml')
    for path, sha in ledger.items():
        verified(dict(path=path, sha256=sha))
    return {key:request[key] for key in ('map_request', 'parent_plan')}


def continue_array(seed, execute, first=2, last=10):
    """Explicit bounded sequencing: callback owns execution/guards, never implicit retry."""
    require(first == 2 and last == 10, 'this experiment requires nine additional cycles only')
    state = seed
    results = []
    for cycle in range(first, last+1):
        if not state['ready']:
            break
        state = execute(cycle, state)
        results.append(state)
    return results


from point_fruit_trajectory import ready_state


def run(args):
    started = time.perf_counter()
    repo = Path(__file__).resolve().parents[2]
    require(not args.output.exists(), 'fresh output required; never overwrite or resume implicitly')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip(), 'clean committed source required')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    short = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=repo, text=True).strip()
    version = subprocess.check_output([str(args.binary), '--version'], text=True)
    require(short in version, 'source/executable revision mismatch')
    owner = json.loads(args.owner.read_text())
    require(owner['nonzero_cycles'] == 10, 'ten-cycle owner binding required')
    verify_manifest(args.first/'cycle', owner['first_cycle_manifest_sha256'])
    verify_manifest(args.first/'diagnostics-r2', owner['corrected_manifest_sha256'])
    original = json.loads((args.first/'cycle/provenance.json').read_text())
    req = original['request']
    maps, admission, additive = (Path(req[n]) for n in ('maps', 'admission', 'additive'))
    hashes = {a:digest(maps/a/'map.fits') for a in ARRAYS}
    require(hashes == original['maps'], 'original maps changed')
    require(verify_companion(admission, hashes)['manifest_sha256'] == original['admission_manifest'], 'admission changed')
    require(verify_companion(additive, hashes)['manifest_sha256'] == original['additive_manifest'], 'noise scale changed')
    verified(original['measurement'])
    args.output.mkdir(parents=True)
    save_json(args.output/'provenance.json', dict(source_revision=revision, binary=bound(args.binary), version=version,
        owner=bound(args.owner), original_provenance=bound(args.first/'cycle/provenance.json'),
        first_cycle_manifest=bound(args.first/'cycle/FILES_SHA256SUMS'),
        corrected_manifest=bound(args.first/'diagnostics-r2/FILES_SHA256SUMS'),
        original_maps=hashes, input_verification_seconds=time.perf_counter()-started,
        request={k:str(v) for k,v in vars(args).items()}))
    contexts = {}
    for a in ARRAYS:
        root = args.output/a
        root.mkdir()
        fixed = FixedMeasurements(maps/a, admission/a, additive/a)
        parents = original_parents(args.first, a)
        require(Path(parents['parent_plan']['path']) == maps/a/'plan.yaml', 'original plan locator changed')
        candidate = args.first/f'diagnostics-r2/{a}/m1.fits'
        require(digest(candidate) == owner['seed_candidates'][a], 'saved m1 differs')
        c, h = load(candidate)
        candidate_guard(c, h)
        oldproducts = args.first/f'cycle/reference/{a}/products'
        processing_guard(maps/a/'map.fits', oldproducts, read_yaml(oldproducts/'receipt.yaml'))
        seed = ready_state(a, 1, candidate, oldproducts/'total.fits', oldproducts/'receipt.yaml', parents, True)
        save_json(root/'seed.json', seed)
        seed['state_path'] = str(root/'seed.json')
        from point_fruit_diagnostics import patch_map
        t1, v1, _, _ = patch_map(oldproducts/'total.fits')
        histories = {0:dict(cycle_index=0, total=bound(maps/a/'map.fits'), candidate=bound(admission/a/'candidate-reference.fits'),
                           measurement=fixed.measure(fixed.original, fixed.support)),
                     1:dict(cycle_index=1, total=seed['total'], candidate=seed['candidate'], measurement=fixed.measure(t1, v1))}
        # The new measurement implementation must reproduce the corrected first-cycle result.
        saved = json.loads((args.first/f'diagnostics-r2/{a}/measurements.json').read_text())
        import numpy as np
        for k, n in [(0, 'M0'), (1, 'T1')]:
            m, old = histories[k]['measurement'], saved['measurements'][n]
            require(np.array_equal(m['fit']['parameters'], old['fit']['parameters']), 'first-cycle fit definition changed')
            require(m['frozen_template_projection'] == old['frozen_template_projection'] and m['apertures'] == old['apertures'], 'fixed measurement definition changed')
        contexts[a] = (root, fixed, parents, seed, histories)

    def zero(a):
        root, fixed, parents, seed, histories = contexts[a]
        try:
            request = make_request(a, 0, verified(seed['candidate']), Path(seed['state_path']), parents, args.owner, True)
            save_json(root/'zero-request.json', request)
            require(invoke(args.binary, root/'zero-request.json', root/'zero', 1, 'point-fruit') == 0, 'zero execution failed')
            products = root/'zero/products'
            receipt = read_yaml(products/'receipt.yaml')
            for seg in receipt['segments']:
                require(seg['input_eligibility_changed'] == seg['output_cause_changed'] == 0 and
                        seg['zero_replay_maximum_cleaned_difference'] <= 1e-8, 'zero PTC mismatch')
            result = compare_zero(fixed.maps/'map.fits', products/'total.fits', fixed.beam)
            save_json(root/'zero-control.json', result)
            return True
        except (ValueError, OSError, KeyError) as e:
            save_json(root/'STOP.json', dict(stage='zero control', reason=str(e), last_completed=1))
            return False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        zero_pass = dict(zip(ARRAYS, pool.map(zero, ARRAYS)))

    def worker(a):
        root, fixed, parents, seed, histories = contexts[a]
        if not zero_pass[a]:
            return dict(array=a, completed_through=1, stop='zero control failed', histories=histories)
        def execute(index, prior):
            begin = time.perf_counter()
            directory = root/f'cycle-{index:02d}'
            directory.mkdir(exist_ok=False)
            try:
                previous = Path(prior['state_path'])
                request = make_request(a, index, verified(prior['candidate']), previous, parents, args.owner)
                save_json(directory/'request.json', request)
                require(invoke(args.binary, directory/'request.json', directory/'native', 1, 'point-fruit') == 0, 'native execution failed; preserved logs')
                products = directory/'native/products'
                diag_started = time.perf_counter()
                report = diagnose_cycle(a, index, fixed, products, directory/'diagnostics', previous, histories)
                report['diagnostic_seconds'] = time.perf_counter()-diag_started
                state = ready_state(a, index, directory/f'diagnostics/m{index}.fits', products/'total.fits', products/'receipt.yaml',
                                    parents, report['ready'], report.get('stop_reason'))
                save_json(directory/'state.json', state)
                state['state_path'] = str(directory/'state.json')
                report['wall_seconds'] = time.perf_counter()-begin
                save_json(directory/'cycle.json', report)
                histories[index] = report
                manifest(directory)
                print(f'{a} cycle {index}: '+('valid; next candidate saved' if state['ready'] else state['stop_reason']), flush=True)
                return state
            except (ValueError, OSError, KeyError) as e:
                state = dict(array=a, cycle_index=index, ready=False, stop_reason=str(e))
                save_json(directory/'STOP.json', state)
                manifest(directory)
                return state
        completed = continue_array(seed, execute)
        last = completed[-1]
        outcome = dict(array=a, completed_through=max(histories), ready=last['ready'], histories=histories,
                       stop='authorized ten total cycles completed; not convergence; m10 unapplied' if last['ready'] and last['cycle_index']==10 else last['stop_reason'])
        save_json(root/'trajectory.json', outcome)
        return outcome
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = dict(zip(ARRAYS, pool.map(worker, ARRAYS)))
    verify_manifest(args.first/'cycle', owner['first_cycle_manifest_sha256'])
    verify_manifest(args.first/'diagnostics-r2', owner['corrected_manifest_sha256'])
    require(hashes == {a:digest(maps/a/'map.fits') for a in ARRAYS}, 'original map changed')
    summary = dict(arrays=results, wall_seconds=time.perf_counter()-started, array_workers=args.workers,
                   numerical_threads_per_worker=1, m10_applied=False, source_revision=revision)
    save_json(args.output/'RUN.json', summary)
    manifest(args.output)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'first', 'owner', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    run(p.parse_args())
