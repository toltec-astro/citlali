#!/usr/bin/env python3
"""Bounded validation reports for existing RTC health evidence; never emits flags."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE = Path('/private/tmp/citlali-rtc-common-mode-health-2026-09-16')
FOLLOW = (221, 445, 426, 236, 301, 492)
CAUSES = {0: 'available', 1: 'insufficient_support', 2: 'weak_reference',
          3: 'input_nonfinite', 4: 'fit_failed'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def bound(path):
    return {'path': str(path), 'sha256': digest(path)}


def load_yaml(path):
    return yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ecsv(path):
    lines = Path(path).read_text().splitlines()
    metadata = yaml.safe_load('\n'.join(x[2:] for x in lines[1:] if x.startswith('# ')))
    return metadata['meta']['canonical_apt_v2'], list(csv.DictReader(
        x for x in lines if not x.startswith('#')))


def accepted_relation(relation):
    # TolProj's is_good_match records the selected seed's flag == 0;
    # it does not annul a unique accepted matched identity.
    return (relation['disposition'] == 'matched'
            and bool(relation['seed_occurrence']) and bool(relation['seed_uid']))


def identity_inventory(manifest, native_receipt):
    """Report verified existing relations; do not implement a detector matcher."""
    manifest = Path(manifest)
    receipt = json.loads(Path(native_receipt).read_text())
    assert receipt['manifest_sha256'] == digest(manifest)
    meta, rows = ecsv(manifest)
    components = {}
    for row in rows:
        path = manifest.parent / row['relative_path']
        assert path.parent == manifest.parent and path.is_file()
        assert 'sha256:' + digest(path) == row['transport_sha256']
        components[row['role']] = ecsv(path) if row['role'] in {'apt', 'relation'} else row
    relation_meta, relations = components['relation']
    apt_meta, apt = components['apt']
    by_uid = {r['output_uid']: r for r in relations}
    assert len(by_uid) == len(relations)
    parent = {x['key']: x['value'] for x in meta['role_metadata'] if x['key'].startswith('baseline_parent.')}
    baseline_key = '|'.join(parent[k] for k in sorted(parent))
    result = []
    for row in apt:
        if row['nw'] != '0':
            continue
        relation = by_uid[row['uid']]
        qualified = accepted_relation(relation)
        result.append({'channel': int(row['kids_tone']), 'array': int(row['array']),
                       'apt_flag': row['flag'], 'apt_flag2': row['flag2'],
                       'tune_flag': row['kids_flag'], 'flxscale_bound': row['flxscale'],
                       'target_occurrence': relation['target_occurrence'],
                       'target_uid': relation['target_uid'], 'output_uid': row['uid'],
                       'relation_occurrence': relation_meta['occurrence'],
                       'relation_envelope': relation_meta['envelope_sha256'],
                       'apt_envelope': apt_meta['envelope_sha256'],
                       'match_disposition': relation['disposition'],
                       'matched_seed_good': relation['is_good_match'] == 'true',
                       'match_qualified': qualified, 'seed_occurrence': relation['seed_occurrence'],
                       'seed_uid': relation['seed_uid'], 'baseline_key': baseline_key,
                       'longitudinal_key': baseline_key + '|' + relation['seed_occurrence'] + '|' + relation['seed_uid'] if qualified else None})
    table = pd.DataFrame(result)
    # Duplicate endpoints are unavailable; a dictionary must never pick one.
    duplicates = table.longitudinal_key.notna() & table.longitudinal_key.duplicated(keep=False)
    table.loc[duplicates, 'longitudinal_key'] = None
    table['longitudinal_reason'] = np.where(duplicates, 'nonunique-qualified-seed',
                                           np.where(table.match_qualified, 'qualified-existing-relation', 'unqualified-or-unmatched'))
    assert table.channel.is_unique and (table.array == 0).all()
    return table


def weighted_median(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return float('nan')
    v, w = values[good], weights[good]
    order = np.argsort(v, kind='stable')
    v, w = v[order], w[order]
    return float(v[np.searchsorted(np.cumsum(w), .5 * w.sum(), side='left')])


def matched_channel(baseline, repeat, baseline_channel):
    row = baseline.loc[baseline.channel == baseline_channel]
    if len(row) != 1 or pd.isna(row.iloc[0].longitudinal_key):
        return None
    key = row.iloc[0].longitudinal_key
    if (baseline.longitudinal_key == key).sum() != 1:
        return None
    matches = repeat.loc[repeat.longitudinal_key == key]
    return int(matches.iloc[0].channel) if len(matches) == 1 else None


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')


def fit_summary(fits, dt):
    good = fits[fits.cause == 0]
    weights = (good.past_last - good['first']).to_numpy()
    result = {'fit_seconds': float(weights.sum() * dt),
              'available_intervals': int(good.interval.nunique()),
              'available_segments': len(good),
              'negative_fit_seconds': float(weights[good.gain.to_numpy() < 0].sum() * dt),
              'unavailable_fit_reasons': json.dumps({CAUSES[int(k)]: int(v) for k, v in fits.loc[fits.cause != 0, 'cause'].value_counts().items()}, sort_keys=True)}
    for field in ['gain', 'relative_gain', 'correlation', 'residual_scatter', 'reference_scatter']:
        result[field + '_duration_weighted_median'] = weighted_median(good[field], weights)
    interval_rows = []
    for interval, part in good.groupby('interval'):
        w = part.past_last - part['first']
        interval_rows.append({'interval': int(interval), 'fitted_seconds': float(w.sum()*dt),
                              'gain': weighted_median(part.gain, w),
                              'q': weighted_median(part.relative_gain, w)})
    gains = [r['gain'] for r in interval_rows]
    result['interval_gain_empirical_mad'] = mad(gains)
    result['negative_interval_count'] = sum(g < 0 for g in gains)
    longest = run = 0
    previous = None
    for r in interval_rows:
        run = run + 1 if r['gain'] < 0 and previous == r['interval'] - 1 else int(r['gain'] < 0)
        longest = max(longest, run)
        previous = r['interval']
    result['longest_consecutive_negative_interval_count'] = longest
    # Contiguous negative support is never joined across an unavailable gap or
    # a processing boundary. Consecutive interval labels above are separate.
    longest_rows = run_rows = 0
    previous_end = previous_interval = None
    for row in good.sort_values(['interval', 'first']).itertuples(index=False):
        length = row.past_last-row.first
        if row.gain < 0:
            run_rows = run_rows + length if previous_end == row.first and previous_interval == row.interval else length
        else:
            run_rows = 0
        longest_rows = max(longest_rows, run_rows)
        previous_end, previous_interval = row.past_last, row.interval
    result['longest_contiguous_negative_fit_seconds'] = float(longest_rows*dt)
    return result, interval_rows


def overlap_fits(a, b):
    """Intersect only the particular pair, retaining original fit coefficients."""
    a = list(a[a.cause == 0].sort_values('first').itertuples(index=False))
    b = list(b[b.cause == 0].sort_values('first').itertuples(index=False))
    i = j = 0
    while i < len(a) and j < len(b):
        x, y = a[i], b[j]
        lo, hi = max(x.first, y.first), min(x.past_last, y.past_last)
        if hi > lo:
            assert x.interval == y.interval
            yield int(lo), int(hi), x, y
        if x.past_last <= y.past_last:
            i += 1
        else:
            j += 1


def matched_comparison(target_fits, control_fits, target_x, control_x, reference, dt):
    pieces = []
    for lo, hi, target, control in overlap_fits(target_fits, control_fits):
        n = hi-lo
        row = {'interval': int(target.interval), 'first': lo, 'past_last': hi,
               'seconds': float(n*dt), 'gain_ratio': float(target.gain/control.gain) if control.gain != 0 else float('nan')}
        for name, fit, x in [('target', target, target_x), ('control', control, control_x)]:
            c, y = reference[lo:hi], x[lo:hi]
            assert np.isfinite(c).all() and np.isfinite(y).all()
            # Same frozen fits, evaluated on the exact common support. No new
            # fit or changed support eligibility is produced by this report.
            row[name+'_correlation'] = float(np.corrcoef(c, y)[0, 1]) if n >= 64 and np.std(c) > 0 and np.std(y) > 0 else float('nan')
            row[name+'_residual_mad'] = mad(y-fit.offset-fit.gain*c) if n >= 64 else float('nan')
        row['reference_mad'] = mad(reference[lo:hi]) if n >= 64 else float('nan')
        pieces.append(row)
    frame = pd.DataFrame(pieces)
    result = {'common_fit_seconds': float(frame.seconds.sum()) if len(frame) else 0.,
              'common_interval_count': int(frame.interval.nunique()) if len(frame) else 0,
              'common_piece_count': len(frame),
              'minimum_metric_samples': 64,
              'metric_seconds': float(frame.loc[(frame.past_last-frame['first']) >= 64, 'seconds'].sum()) if len(frame) else 0.}
    for field in ['gain_ratio', 'target_correlation', 'control_correlation', 'target_residual_mad', 'control_residual_mad', 'reference_mad']:
        result[field+'_duration_weighted_median'] = weighted_median(frame[field], frame.seconds) if len(frame) else float('nan')
    return result, frame


def prepare(selection_path, output):
    selection = json.loads(selection_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    configs, identities = [], {}
    base = json.loads((BASE/'prepared/network0.json').read_text())
    for entry in selection['observations']:
        obs = entry['observation']
        cfg = json.loads(json.dumps(base))
        for key in ['raw', 'tune', 'manifest', 'telescope']:
            assert digest(entry[key]['path']) == entry[key]['sha256']
            cfg[key] = entry[key]
        native = BASE/'native-network0' if obs == 152390 else output.parent/'native-152392-full'
        cfg['audit_receipt'] = bound(native/'receipt.json')
        cfg['detectors'] = [{'channel': c, 'samples': bound(native/f'samples-{c}.f64')} for c in range(entry['rows_channels'][1])]
        cfg['observation'] = obs
        cfg['common_mode_census'] = {'selection': bound(selection_path)}
        if obs != 152390:
            cfg['source_protection_authority'] = 'rtc-census-source-membership-unavailable'
            data = Path(entry['raw']['path']).parent
            for timing in cfg['decision_apply']['timing_inputs']:
                network = timing['network']
                timing.update(bound(next(data.glob(f'toltec{network}_{obs}_000_0002_*.nc'))))
            prior = Path(cfg['decision_apply']['processing_provenance']['path'])
            cfg['decision_apply']['processing_provenance'] = bound(prior.parent.parent/str(obs)/prior.name)
            cfg['ast_acceptance'] = bound(Path('/Users/gwilson/work_toltec/local_data/citlali-validation/wp7/rtc-filter-census/adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7')/f'science-{obs}.json')
        identities[obs] = identity_inventory(cfg['manifest']['path'], cfg['audit_receipt']['path'])
        identities[obs].to_csv(output/f'identity-{obs}.csv', index=False)
        path = output/f'{obs}.json'
        dump_json(path, cfg)
        configs.append(bound(path))
    for target in selection['follow_targets']:
        assert target['repeat_channel'] == matched_channel(identities[152390], identities[152392], target['baseline_channel'])
    dump_json(output/'manifest.json', {'selection': bound(selection_path), 'configs': configs})


def report(selection_path, evidence, output):
    selection = json.loads(selection_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    census, followed, pair_rows, pair_pieces, histories, reference_rows = [], [], [], [], [], []
    observations = []
    for item in selection['observations']:
        obs = item['observation']
        replay = evidence/f'replay-{obs}'
        cfg = json.loads((evidence/f'prepared/{obs}.json').read_text())
        receipt = load_yaml(replay/'receipt.yaml')
        health_receipt = load_yaml(replay/'health/receipt.yaml')
        assert receipt['original_pair_unchanged'] and not receipt['Apply_performed']
        assert health_receipt['configuration_sha256'] == digest(evidence/f'prepared/{obs}.json')
        dt = receipt['native_integration_seconds']
        ids = identity_inventory(cfg['manifest']['path'], cfg['audit_receipt']['path'])
        table = pd.read_csv(replay/'health/detectors.csv')
        fits = pd.read_csv(replay/'health/fits.csv')
        intervals = load_yaml(replay/'health/intervals.yaml')
        table = table.merge(ids, on='channel', validate='one_to_one')
        assert len(table) == item['rows_channels'][1]
        by_detector = {d: group for d, group in fits.groupby('detector')}
        empty = fits.iloc[:0]
        summaries = []
        for d in table.detector:
            summary, _ = fit_summary(by_detector.get(d, empty), dt)
            summary['detector'] = d
            summaries.append(summary)
        table = table.merge(pd.DataFrame(summaries), on='detector', suffixes=('', '_accounted'))
        table['observation'] = obs
        table['original_paired_seconds'] = np.sum([s['original_paired_rows'] for s in intervals], axis=0)*dt
        table['adaptive_eligible_seconds'] = np.sum([s['eligible_target_rows'] for s in intervals], axis=0)*dt
        table['no_fit_state'] = np.where(table.fit_seconds == 0, 'inconclusive-no-available-fit;inspect-interval-support', 'has-available-fits-not-a-health-verdict')
        census.append(table)
        denominator = float(table.loc[table.reference_population == 1, 'adaptive_eligible_seconds'].sum())
        target_map = {}
        for target in selection['follow_targets']:
            base_channel = target['baseline_channel']
            channel = base_channel if obs == 152390 else target['repeat_channel']
            if channel is None:
                followed.append({'observation': obs, 'baseline_channel': base_channel,
                                 'channel': None, 'longitudinal_state': target['longitudinal_state'],
                                 'interpretation_state': 'unresolved-identity-unavailable'})
                continue
            row = table.loc[table.channel == channel].iloc[0].to_dict()
            row.update(baseline_channel=base_channel, longitudinal_state=target['longitudinal_state'])
            target_map[base_channel] = (channel, int(row['detector']))
            loo = pd.read_csv(replay/f'health/self-excluded-{channel}.csv')
            summary, history = fit_summary(loo, dt)
            row.update({'loo_'+k: v for k, v in summary.items()})
            row['prospective_gross_paired_seconds'] = row['original_paired_seconds']
            row['prospective_good_population_adaptive_seconds'] = row['adaptive_eligible_seconds'] if row['reference_population'] else 0.
            row['prospective_equal_detector_percent'] = 100*row['prospective_good_population_adaptive_seconds']/denominator
            row['additional_cost_beyond_treatment'] = 'unmeasured-no-full-network-treatment-mask;not-sensitivity'
            followed.append(row)
            for r in history:
                r.update(observation=obs, baseline_channel=base_channel, channel=channel, reference='target-self-excluded')
                histories.append(r)
        raw = {base: np.memmap(cfg['detectors'][d]['samples']['path'], dtype='<f8', mode='r', shape=(receipt['rows'], 4))[:, 0] for base, (channel, d) in target_map.items()}
        for name in ['health', 'partition-0', 'partition-1']:
            reference = np.fromfile(replay/name/'reference.f64', dtype='<f8').reshape(-1, 3)
            ff = fits if name == 'health' else pd.read_csv(replay/name/'fits.csv')
            fi = {d: group for d, group in ff.groupby('detector')}
            rr = load_yaml(replay/name/'intervals.yaml')
            for index, r in enumerate(rr):
                reference_rows.append({'observation': obs, 'reference': name, 'interval': index,
                                       'first': r['rows'][0], 'past_last': r['rows'][1],
                                       'contributors': r['contributors'], 'reference_scatter': r['reference_scatter'],
                                       'speed_eligible_seconds': r['speed_eligible_rows']*dt,
                                       'available_reference_seconds': np.isfinite(reference[r['rows'][0]:r['rows'][1], 1]).sum()*dt})
            for base_channel, (_, d) in target_map.items():
                for control in [301, 492]:
                    if base_channel == control or (base_channel == 492 and control == 301):
                        continue
                    if control not in target_map:
                        continue
                    control_d = target_map[control][1]
                    result, pieces = matched_comparison(fi.get(d, empty), fi.get(control_d, empty), raw[base_channel], raw[control], reference[:, 1], dt)
                    keys = {'observation': obs, 'reference': name, 'baseline_channel': base_channel, 'control_baseline_channel': control}
                    result.update(keys)
                    pair_rows.append(result)
                    for k, v in keys.items():
                        pieces[k] = v
                    pair_pieces.append(pieces)
        observations.append({'observation': obs, 'source_revision': receipt['source_revision'],
                             'detectors': len(table), 'initial_population': int(table.reference_population.sum()),
                             'targets_with_fits': int((table.fit_seconds > 0).sum()),
                             'intervals': len(intervals), 'contributor_min_median_max': np.quantile([s['contributors'] for s in intervals], [0, .5, 1]).tolist(),
                             'adaptive_good_population_detector_seconds': denominator,
                             'fit_seconds_total': float(table.fit_seconds.sum()),
                             'runtime': {k: v for k, v in health_receipt.items() if k.endswith('_seconds') or k.endswith('_bytes')}})
    full = pd.concat(census, ignore_index=True)
    compact = pd.DataFrame(followed)
    pairs = pd.DataFrame(pair_rows)
    refs = pd.DataFrame(reference_rows)
    history = pd.DataFrame(histories)
    full.to_csv(output/'full-network-census.csv', index=False)
    compact.to_csv(output/'followed-targets.csv', index=False)
    pairs.to_csv(output/'matched-comparisons.csv', index=False)
    pd.concat(pair_pieces, ignore_index=True).to_csv(output/'matched-support-pieces.csv', index=False)
    refs.to_csv(output/'reference-adequacy.csv', index=False)
    history.to_csv(output/'self-excluded-interval-history.csv', index=False)
    dump_json(output/'observations.json', observations)
    make_plots(full, compact, pairs, refs, history, output)
    print(compact[[k for k in ['observation', 'baseline_channel', 'channel', 'relative_gain', 'loo_gain_duration_weighted_median', 'loo_correlation_duration_weighted_median', 'loo_negative_fit_seconds', 'loo_fit_seconds'] if k in compact]].to_string(index=False))


def make_plots(full, compact, pairs, refs, history, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, obs in enumerate(sorted(full.observation.unique())):
        for name, group in refs[refs.observation == obs].groupby('reference'):
            axes[0, col].plot(group.interval, group.reference_scatter, label=name)
            axes[1, col].plot(group.interval, group.contributors, label=name)
        axes[0, col].set(title=f'{obs} / network 0', ylabel='Reference MAD (native x)')
        axes[1, col].set(xlabel='Existing processing interval', ylabel='Fixed contributor count')
        axes[0, col].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Frozen estimator; one initial reference and two disjoint peer subsets')
    fig.savefig(output/'reference-adequacy.png', dpi=145)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    for base, ax in zip(FOLLOW, axes.flat):
        for obs, group in history[history.baseline_channel == base].groupby('observation'):
            series = group.set_index('interval').gain.reindex(range(int(group.interval.max())+1))
            ax.plot(series.index, series, marker='.', markersize=2, label=str(obs))
        ax.set(title=f'Baseline target {base}', xlabel='Existing interval (within each observation)', ylabel='Signed self-excluded gain')
        ax.axhline(0, color='black', lw=.5)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Raw gain is reference-dependent; compare ratios and q separately\n426 retains its existing flags and unavailable calibration; gaps are not measurements')
    fig.savefig(output/'response-history.png', dpi=145)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for obs, ax in zip(sorted(full.observation.unique()), axes):
        group = full[full.observation == obs]
        eligible = group[group.reference_population == 1]
        ax.scatter(eligible.relative_gain, eligible.correlation, s=9, alpha=.55, label='good APT/Tune population')
        other = group[group.reference_population != 1]
        ax.scatter(other.relative_gain, other.correlation, s=12, marker='x', alpha=.5, label='other targets with available q')
        for row in compact[(compact.observation == obs) & compact.channel.notna()].itertuples():
            if np.isfinite(row.relative_gain) and np.isfinite(row.correlation):
                ax.annotate(str(row.baseline_channel), (row.relative_gain, row.correlation), fontsize=8)
        ax.set(title=str(obs), xlabel='Original runtime median q (initial reference)', ylabel='Original runtime median correlation')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Full inventories retained in CSV, including missing calibration and unavailable fits')
    fig.savefig(output/'population-comparison.png', dpi=145)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'report'])
    parser.add_argument('selection', type=Path)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare(args.selection, args.evidence/'prepared')
    else:
        report(args.selection, args.evidence, args.evidence/'analysis')


if __name__ == '__main__':
    main()
