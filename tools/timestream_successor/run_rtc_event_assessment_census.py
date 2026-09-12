#!/usr/bin/env python3
"""Local, inert RTC event-assessment census driver over an explicit header-inventory JSON.

All science runs in the C++ executable. This program selects exact input copies,
records per-invocation resource/provenance/failure evidence, and reconciles counts.
It does not submit jobs, fetch inputs, issue APTs, change policy, or merge refs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


GROUP_SPAN_POLICY = 'rtc-jump-transition-2026-09-10-v1'
ONSET_POLICY = 'rtc-jump-transition-onset-2026-09-12-v2'


def onset_cells(event, candidates):
    """Independent set closure; do not assume the producer's member ordering."""
    seed = candidates[event['seed']]
    component = set(range(seed['earlier_row'], seed['later_row'] + 1))
    while True:
        expanded = set(component)
        for index in event['candidates']:
            candidate = candidates[index]
            edge = set(range(candidate['earlier_row'], candidate['later_row'] + 1))
            if edge & component:
                expanded |= edge
        if expanded == component:
            return component
        component = expanded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trial-half-width-seconds', type=float, required=True)
    parser.add_argument('--previous-campaign', type=Path,
                        help='Require unchanged inputs and upstream outputs; only the named v1-to-v2 transition change may differ')
    parser.add_argument('--example-selection', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists: preserve prior campaigns')
    args.output.mkdir(parents=True)
    exe = args.executable.resolve(strict=True)
    executable_hash = digest(exe)
    selection_hash = digest(args.example_selection)
    (args.output/'selected-examples.tsv').write_bytes(args.example_selection.read_bytes())
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
            command = [str(exe), str(raw), str(tune), str(manifest), str(target), str(args.trial_half_width_seconds), str(args.example_selection.resolve(strict=True))]
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
                fits_count = sum(1 for _ in (target/'candidates.jsonl').open())
                events = [json.loads(line) for line in (target/'events.jsonl').open()]
                assert len(events) == receipt['assessed_events']
                members = [i for e in events for i in e['candidates']]
                assert sorted(members) == list(range(fits_count)), 'candidate membership is incomplete or duplicated'
                assert all(not e['hard_event_accepted'] and not e['apply_authorized'] and e['spectral_context_unavailable'] for e in events)
                assert len(detectors) == r['channels'] == receipt['channels']
                assert fits_count == receipt['candidate_edges'] == sum(sum(d['candidate_counts']) for d in detectors)
                assert len({d['occurrence'] for d in detectors}) == len(detectors)
                assert all(d['rows'] == r['rows'] and d['pair_screening_excluded_rows'] <= d['rows'] for d in detectors)
                timing = json.loads((target/'timing.json').read_text())
                transition_policy = timing['transition_policy']
                assert transition_policy in (GROUP_SPAN_POLICY, ONSET_POLICY), 'unreviewed transition policy'
                stage_values = list(timing['stages_seconds'].values())
                assert all(math.isfinite(v) and v >= 0 for v in stage_values)
                assert math.isclose(sum(stage_values), timing['measured_total_seconds'], rel_tol=1e-10, abs_tol=1e-7)
                seeds = {s['candidate']: s for s in map(json.loads, (target/'candidates.jsonl').open())}
                noise = {b['noise_block']: b for b in map(json.loads, (target/'health-blocks.jsonl').open())}
                amplitude_counts, consistency_counts = [0]*6, [0]*7
                joint_calls = pre_calls = short_available = without_recovery = jump_rows = 0
                for row in map(json.loads, (target/'jump-consistency.jsonl').open()):
                    assert row['event'] == jump_rows
                    event = events[jump_rows]
                    assert row['detector'] == event['detector']
                    assert not row['hard_event_accepted'] and not row['apply_authorized']
                    for c, check in enumerate(row['coordinates']):
                        amplitude_counts[check['amplitude_cause']] += 1
                        consistency_counts[check['consistency_cause']] += 1
                        first = next((i for i in event['candidates'] if seeds[i]['seed_coordinate'] == c), None)
                        assert check['candidate'] == first
                        if first is not None:
                            block = noise[check['noise_block']]
                            assert block['detector'] == event['detector']
                            # The original producer assigns an edge to its later
                            # endpoint's block, including a crossing at block start.
                            assert seeds[first]['earlier_row'] + 1 == seeds[first]['later_row']
                            assert block['first'] <= seeds[first]['later_row'] < block['end']
                            if check['sigma_delta'] is not None:
                                assert check['sigma_delta'] == block['coordinates'][c]['scale']
                        fit = check['short_fit']
                        if check['amplitude_cause'] != 5:
                            assert fit is None and check['short_fit_cause'] == 0
                        else:
                            assert fit is not None
                            pre_calls += fit['pre_scale_fit']['cause'] != 1
                            joint_calls += fit['with_offset']['cause'] != 1
                            short_available += fit['available']
                        if check['consistency_cause'] == 6:
                            assert fit['available'] and check['amplitude_cause'] == 5
                            a = event['coordinates'][c]['with_offset']['offset']
                            b = fit['with_offset']['offset']
                            sigma = check['sigma_delta']
                            assert abs(a) >= 5*sigma and abs(b) >= 5*sigma
                            assert (a > 0) == (b > 0) and abs(a-b) <= 2*sigma
                            without_recovery += not check['confirmed_recovery_excludes_persistent_shift']
                    jump_rows += 1
                assert jump_rows == len(events)
                assert amplitude_counts == timing['amplitude_cause_counts']
                assert consistency_counts == timing['consistency_cause_counts']
                assert amplitude_counts[5] == timing['requested_coordinates']
                assert pre_calls == timing['pre_fit_calls'] and joint_calls == timing['joint_fit_calls']
                assert short_available == timing['available_coordinates']
                assert without_recovery == timing['consistent_without_confirmed_recovery']
                transition_counts = [0]*10
                transition_requested = transition_examined = transition_outside = transition_rows = 0
                checks = map(json.loads, (target/'jump-consistency.jsonl').open())
                for bound, check in zip(map(json.loads, (target/'jump-transitions.jsonl').open()), checks, strict=True):
                    assert bound['event'] == check['event'] == transition_rows
                    assert bound['detector'] == check['detector']
                    assert not bound['hard_event_accepted'] and not bound['apply_authorized']
                    event = events[transition_rows]
                    for c, b in enumerate(bound['coordinates']):
                        d = check['coordinates'][c]
                        expected = 0 if d['consistency_cause'] != 6 else 1 if d['confirmed_recovery_excludes_persistent_shift'] else 2
                        assert b['request_cause'] == expected
                        transition_requested += expected == 2
                        transition_counts[b['cause']] += 1
                        transition_examined += b['examined_rows']
                        assert b['available'] == (b['cause'] == 1)
                        assert not b['timing_uncertainty_quantified']
                        assert not b['physical_event_identity_resolved']
                        if expected != 2:
                            assert b['cause'] == 0 and b['examined_rows'] == 0
                        else:
                            assert b['cause'] != 0
                            assert b['residual_scale'] == event['coordinates'][c]['pre']['scale']
                        if b['available']:
                            pre, post = b['confirmations']
                            assert pre['end']-pre['begin'] >= .05 and post['end']-post['begin'] >= .05
                            assert not pre['also_matches_other_reference'] and not post['also_matches_other_reference']
                            assert pre['rows'][1] == b['affected'][0] < b['affected'][1] == post['rows'][0]
                            assert pre['end'] == b['begin'] < b['end'] == post['begin']
                            edge_rows = {seeds[k]['earlier_row'] for k in event['candidates']}
                            assert b['multiple_candidate_edges'] == (len(edge_rows) > 1)
                            required = (onset_cells(event, seeds) if transition_policy == ONSET_POLICY
                                        else {min(edge_rows), max(edge_rows)+1})
                            assert b['affected'][0] <= min(required) and b['affected'][1] > max(required)
                            expected_outside = b['affected'][0] < event['trial_exclusion'][0] or b['affected'][1] > event['trial_exclusion'][1]
                            assert b['exceeds_fitting_exclusion'] == expected_outside
                            transition_outside += expected_outside
                    transition_rows += 1
                assert transition_rows == len(events)
                assert transition_counts == timing['transition_cause_counts']
                assert transition_requested == without_recovery == timing['transition_requested']
                assert transition_examined == timing['transition_examined_rows']
                assert transition_outside == timing['transition_outside_trial']
                if args.previous_campaign:
                    previous = args.previous_campaign/stem
                    old_receipt = json.loads((previous/'receipt.json').read_text())
                    for key in ('raw_sha256', 'tune_sha256', 'manifest_sha256', 'candidate_edges', 'assessed_events'):
                        assert receipt[key] == old_receipt[key], f'Prior identity/count changed: {key}'
                    preserved = ('candidates.jsonl', 'events.jsonl', 'health-blocks.jsonl',
                                 'detectors.jsonl', 'examples.jsonl', 'jump-consistency.jsonl')
                    previous_policy = json.loads((previous/'timing.json').read_text())['transition_policy']
                    if previous_policy == transition_policy:
                        preserved += ('jump-transitions.jsonl',)
                    else:
                        assert (previous_policy, transition_policy) == (GROUP_SPAN_POLICY, ONSET_POLICY)
                    entry['transition_policy_comparison'] = [previous_policy, transition_policy]
                    for name in preserved:
                        assert digest(target/name) == digest(previous/name), f'Prior output changed: {name}'
                    entry['preserved_previous_outputs'] = list(preserved)
                entry.update(status='completed', receipt=receipt, fit_rows=fits_count, assessed_events=len(events), jump_timing=timing)
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
                   assessed_events=sum(e['assessed_events'] for e in complete),
                   deferred_network_files=len(deferred), deferred_channels=sum(e['channels'] for e in deferred),
                   elapsed_network_wall_seconds=sum(e.get('wall_seconds',0) for e in entries),
                   peak_child_rss_bytes=max((e.get('max_rss_bytes',0) for e in entries), default=0),
                   inventory_sha256=digest(args.inventory), executable_sha256=executable_hash,
                   trial_half_width_seconds=args.trial_half_width_seconds,
                   retained_limits=['Measured trial support/recovery and review dispositions, no accepted hard-event classification',
                                    'Source membership unavailable; no optical rejection or Apply',
                                    'Spectral context remains owner-deferred and unavailable',
                                    'Input preflight defers missing canonical detector bindings',
                                    'Candidate edges may represent the same physical event; counts are not event rates'])
    stage_names = sorted({s for e in complete for s in e['jump_timing']['stages_seconds']})
    summary['timed_stages_seconds'] = {s: sum(e['jump_timing']['stages_seconds'][s] for e in complete) for s in stage_names}
    summary['jump_fit_counts'] = {k: sum(e['jump_timing'][k] for e in complete) for k in (
        'requested_coordinates', 'pre_fit_calls', 'joint_fit_calls', 'available_coordinates',
        'pre_reported_iterations', 'joint_reported_iterations', 'consistent_without_confirmed_recovery')}
    summary['peak_short_scratch_rows'] = max((e['jump_timing']['peak_short_scratch_rows'] for e in complete), default=0)
    summary['transition_cause_counts'] = [sum(e['jump_timing']['transition_cause_counts'][i] for e in complete) for i in range(10)]
    summary['transition_requested'] = sum(e['jump_timing']['transition_requested'] for e in complete)
    summary['transition_examined_rows'] = sum(e['jump_timing']['transition_examined_rows'] for e in complete)
    summary['transition_outside_trial'] = sum(e['jump_timing']['transition_outside_trial'] for e in complete)
    summary['peak_transition_index_entries'] = max((e['jump_timing']['transition_index_entries'] for e in complete), default=0)
    summary['peak_transition_neighbor_ranges'] = max((e['jump_timing']['transition_peak_neighbor_ranges'] for e in complete), default=0)
    summary['peak_transition_result_bytes'] = max((e['jump_timing']['transition_result_bytes'] for e in complete), default=0)
    summary['preserved_previous_outputs'] = sum(len(e.get('preserved_previous_outputs', [])) for e in complete)
    summary['timing_scope'] = 'Local inert native test driver; production RTC/PTC runtime not measured; iteration counts include successful and failed IRLS loop entries, zero before loop'
    assert digest(args.example_selection) == selection_hash, 'Example selection changed during census'
    summary['example_selection_sha256'] = selection_hash
    write_json(args.output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if len(complete) != len(candidates):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
