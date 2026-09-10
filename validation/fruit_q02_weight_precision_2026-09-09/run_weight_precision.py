#!/usr/bin/env python3
"""Isolated SCI-FRUIT Q02 r0.5 diagnostic; never runs a reduction or map."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

import netCDF4
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits, threadpool_info

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
Q = REPO / 'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review'
ARRAYS = ['a1100', 'a1400', 'a2000']
RTOL, ATOL = 1e-12, 1e-14


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


class BoundExceeded(RuntimeError):
    pass


class Run:
    def __init__(self, out, settings):
        self.out, self.settings = out, settings
        self.start = time.monotonic()
        self.events = 0

    def check(self, stage=None):
        elapsed = time.monotonic() - self.start
        size = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if (elapsed > self.settings['wall_time_limit_seconds'] or
                peak_bytes() > self.settings['aggregate_memory_limit_bytes'] or
                size > self.settings['output_limit_bytes']):
            raise BoundExceeded(f'Resource limit: seconds={elapsed}, peak={peak_bytes()}, bytes={size}')
        if stage:
            event = dict(seconds=round(elapsed, 3), stage=stage, peak_bytes=peak_bytes(), output_bytes=size)
            with (self.out / 'progress.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            print(json.dumps(event), flush=True)
            self.events += 1


def recover_edges(scans):
    scans = np.asarray(scans)
    require(scans.shape == (12, 2) and np.isfinite(scans).all(), 'Invalid chunk labels')
    require(np.equal(scans, np.floor(scans)).all(), 'Noninteger chunk labels')
    scans = scans.astype(np.int64)
    lengths = np.r_[scans[0, 1] + 1, np.diff(scans[:, 1])]
    require(scans[0, 0] == 0 and np.all(lengths > 0), 'Invalid first chunk or lengths')
    require(np.array_equal(np.diff(scans[:, 0]), np.diff(scans[:, 1])), 'Writer recurrence mismatch')
    return np.r_[0, np.cumsum(lengths)]


def split_rows(edges):
    training, evaluation, chunks, early = [], [], [], []
    for c, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        split = lo + (hi - lo) // 2
        tr = np.arange(lo, split)
        training.extend(tr)
        evaluation.extend(np.arange(split, hi))
        chunks.extend([c] * len(tr))
        early.extend(np.arange(len(tr)) < len(tr) // 2)
    return tuple(np.asarray(x) for x in (training, evaluation, chunks, early))


def moments(values, valid):
    valid = np.asarray(valid, bool) & np.isfinite(values)
    n = valid.sum(axis=0)
    mu = np.full(values.shape[1], np.nan)
    np.divide(np.where(valid, values, 0.).sum(axis=0), n, out=mu, where=n > 0)
    with np.errstate(invalid='ignore', over='ignore'):
        residual = np.where(valid, values - mu, 0.)
        v = np.full(values.shape[1], np.nan)
        np.divide(np.sum(residual * residual, axis=0), n, out=v, where=n > 0)
    return n, mu, v


def inverse_and_score(values, valid, stats):
    n, mu, v = stats
    w = np.full_like(v, np.nan, dtype=float)
    arithmetic = (n >= 2) & np.isfinite(v) & (v > 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        np.divide(1., v, out=w, where=arithmetic)
    arithmetic &= np.isfinite(w) & (w > 0)
    w[~arithmetic] = np.nan
    c = np.zeros(values.shape, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        residual2 = np.where(valid, (values - mu) ** 2, 0.)
        numerator = np.where(valid, residual2 - v, 0.)
        np.divide(-numerator, n * v * v, out=c, where=arithmetic[None, :])
    score_ok = arithmetic & (n > 2) & np.isfinite(c).all(axis=0) & np.any(c != 0, axis=0)
    c[:, ~score_ok] = 0.
    return w, c, score_ok


def normalize(weights, evaluation_counts):
    required = evaluation_counts > 0
    gamma = np.full_like(weights, np.nan, dtype=float)
    if not required.any():
        return gamma, np.nan, 'unavailable_empty_population'
    if not np.all(np.isfinite(weights[required]) & (weights[required] > 0)):
        return gamma, np.nan, 'unavailable_required_weight'
    mean = np.sum(evaluation_counts[required] * weights[required]) / evaluation_counts[required].sum()
    if not np.isfinite(mean) or mean <= 0:
        return gamma, np.nan, 'unavailable_normalization'
    gamma[required] = weights[required] / mean
    require(np.all(np.isfinite(gamma[required]) & (gamma[required] > 0)), 'Nonfinite normalized coefficients')
    return gamma, mean, 'available'


def joint_state(weights, counts, score_ok):
    gamma, mean, state = normalize(weights, counts)
    if state != 'available':
        return gamma, mean, state, state
    uncertainty = 'available' if np.all(score_ok[counts > 0]) else 'unresolved_required_score'
    return gamma, mean, state, uncertainty


def kernel_plan(times, tau):
    """Integral of overlaps of [T,T+tau); algebraically the Bartlett kernel."""
    t = np.asarray(times, float)
    require(t.ndim == 1 and len(t) > 0 and np.isfinite(t).all(), 'Bad kernel time axis')
    require(np.all(np.diff(t) > 0) and tau > 0, 'Kernel requires ordered unique times')
    t = t - t[0]  # Translation only; every elapsed gap is retained.
    endpoints = np.unique(np.r_[t, t + tau])
    left = endpoints[:-1]
    return (np.searchsorted(t, left, side='right'),
            np.searchsorted(t + tau, left, side='right'), np.diff(endpoints) / tau)


def kernel_quad(values, plan, batch=64):
    """Diagonal c'Kc via nonnegative interval integrals, in bounded batches."""
    if values.ndim == 1:
        values = values[:, None]
    starts, ends, lengths = plan
    result = np.empty(values.shape[1], float)
    for d in range(0, values.shape[1], batch):
        x = values[:, d:d + batch]
        prefix = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
        active = prefix[starts] - prefix[ends]
        result[d:d + batch] = np.sum((active * active) * lengths[:, None], axis=0)
    require(np.isfinite(result).all(), 'Nonfinite quadratic form')
    require(np.all(result >= 0), 'Negative nonnegative-integral quadratic form')
    return result


def bin_counts(times, valid, tau):
    ids = np.floor(times / tau).astype(np.int64)
    unique, inverse = np.unique(ids, return_inverse=True)
    counts = np.zeros((len(unique), valid.shape[1]), dtype=np.uint16)
    np.add.at(counts, inverse, valid.astype(np.uint16))
    return unique, counts


def quantiles(values, weights=None):
    values = np.asarray(values).ravel()
    weights = np.ones(values.shape) if weights is None else np.asarray(weights).ravel()
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        return [None, None, None]
    v, w = values[good], weights[good]
    order = np.argsort(v, kind='stable')
    v, w = v[order], w[order]
    cumulative = np.cumsum(w) / np.sum(w)
    return [float(v[min(np.searchsorted(cumulative, p, side='left'), len(v) - 1)]) for p in (.1, .5, .9)]


def series_summary(values, counts):
    values, counts = np.asarray(values).ravel(), np.asarray(counts).ravel()
    required = counts > 0
    good = required & np.isfinite(values)
    below = good & (values <= .20)
    above = good & (values > .20)
    total = int(counts[required].sum())
    available_count = int(counts[good].sum())
    return dict(required_groups=int(required.sum()), available_groups=int(good.sum()),
                unavailable_groups=int((required & ~good).sum()), above_goal_groups=int(above.sum()),
                at_or_below_goal_groups=int(below.sum()),
                group_quantiles_10_50_90=quantiles(values[required]),
                occurrence_weighted_quantiles_10_50_90=quantiles(values[required], counts[required]),
                below_goal_fraction_of_available_groups=float(below.sum() / good.sum()) if good.any() else None,
                below_goal_fraction_of_all_required_groups=float(below.sum() / required.sum()) if required.any() else None,
                available_evaluation_occurrences=available_count,
                below_goal_fraction_of_available_occurrences=float(counts[below].sum() / available_count) if available_count else None,
                below_goal_fraction_of_all_occurrences=float(counts[below].sum() / total) if total else None,
                unknown_fraction_of_all_occurrences=float(counts[required & ~good].sum() / total) if total else None,
                above_goal_fraction_of_all_occurrences=float(counts[above].sum() / total) if total else None)


def read_plain(f, key):
    v = f[key][:]
    require(not np.ma.getmaskarray(v).any(), 'Unexpected metadata mask: ' + key)
    return np.asarray(v)


def signal_values(variable, rows):
    """Only this function reads signal; callers provide one named half-chunk."""
    value = variable[rows, :]
    masked = np.ma.getmaskarray(value)
    data = np.asarray(np.ma.getdata(value), dtype=np.float64)
    valid = ~masked & np.isfinite(data)
    return data, valid, masked


def signal_finiteness(variable, rows):
    # Evaluation values never leave this function; no moments or other statistics.
    value = variable[rows, :]
    masked = np.ma.getmaskarray(value)
    finite = np.isfinite(np.ma.getdata(value))
    return ~masked & finite, masked, ~finite & ~masked


def load_observation(item, settings, run):
    obs = item['obsnum']
    prior = json.loads((Q/'r0.4/DISCOVERY_PREFLIGHT.json').read_text())
    prior = next(o for o in prior['observations'] if o['obsnum'] == obs)
    with netCDF4.Dataset(item['path'], 'r') as f:
        edges = recover_edges(read_plain(f, 'scan_indices'))
        require(edges.tolist() == settings['edges'], 'Recovered edges changed')
        require(read_plain(f, 'scan_indices').tolist() == prior['reported_scan_indices'], 'Original labels changed')
        raw = read_plain(f, 'raw_scan_indices')
        if obs == 123424:
            require(np.array_equal((raw[:, 1] - raw[:, 0] + 1) / 2, np.diff(edges)), 'Raw-index corroboration failed')
        else:
            require(read_plain(f, 'output_scan_index').tolist() == list(range(1, 13)), 'Output scan identity changed')
            require(np.diff(edges).tolist() == prior['corroboration']['chunk_summary_lengths'], 'Prior log lengths differ')
        tr, ev, chunk, early = split_rows(edges)
        t = read_plain(f, 'TelTime').astype(float)
        require(len(t) == edges[-1] and np.isfinite(t).all() and np.all(np.diff(t) > 0), 'Invalid elapsed time')
        t = t - t[0]
        nd = len(f.dimensions['n_dets'])
        require(nd == prior['identity']['slots'], 'Detector shape changed')
        require(f['signal'].shape == f['flags'].shape == (3628, nd), 'Signal/flag schema changed')
        expected_dtype = np.dtype('float64' if obs == 123424 else 'float32')
        require(np.dtype(f['signal'].dtype) == expected_dtype, 'Signal precision changed')
        base = {k: read_plain(f, k) for k in ('TelElAct', 'alt_phys', 'az_phys',
                'pointing_offset_alt', 'pointing_offset_az', 'apt_x_t', 'apt_y_t',
                'apt_uid', 'apt_nw', 'apt_array', 'apt_flag')}
        arr, uid, nw = base['apt_array'], base['apt_uid'], base['apt_nw']
        b = np.full((len(tr), nd), np.nan)
        valid = np.zeros(b.shape, bool)
        near = np.zeros(b.shape, bool)
        eval_n = np.zeros((12, nd), np.int64)
        central_n = np.zeros_like(eval_n)
        # Sequential exclusions: flags, coordinates, signal, then source guard.
        exclusions = np.zeros((12, nd, 4), np.int64)
        eval_exclusions = np.zeros((12, nd, 3), np.int64)
        raw_signal_bad = np.zeros((12, nd, 4), np.int64)
        participating = np.zeros(nd, bool)
        coord_error = [0., 0.]
        radius = settings['guard_arcsec'][str(obs)] * np.pi / 648000
        cursor = 0
        for c, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            run.check(f'{obs}: read training half and evaluation finite census, chunk {c}')
            ntr = (hi - lo) // 2
            flags = read_plain_slice(f, 'flags', slice(lo, hi))
            require(np.isin(flags, [0, 1]).all(), 'Unexpected flags')
            good = flags == 0
            participating |= good.any(axis=0)
            require(not np.any(good & (base['apt_flag'] != 0)[None, :]), 'Unexpected final APT/sample flag relation')
            e = base['TelElAct'][lo:hi, None]
            x = base['az_phys'][lo:hi, None] + (np.cos(e)*base['apt_x_t'] - np.sin(e)*base['apt_y_t'] + base['pointing_offset_az'][lo:hi, None])*np.pi/648000
            y = base['alt_phys'][lo:hi, None] + (np.cos(e)*base['apt_y_t'] + np.sin(e)*base['apt_x_t'] + base['pointing_offset_alt'][lo:hi, None])*np.pi/648000
            if obs == 123424:
                for index, (name, pred) in enumerate((('det_lon', x), ('det_lat', y))):
                    supplied = read_plain_slice(f, name, slice(lo, hi))
                    finite = np.isfinite(pred)
                    require(np.array_equal(finite, np.isfinite(supplied)), 'Coordinate finite pattern differs')
                    difference = float(np.max(np.abs(supplied[finite] - pred[finite])))
                    require(difference <= 1e-15, 'Coordinate adapter mismatch')
                    coord_error[index] = max(coord_error[index], difference)
            finite_coords = np.isfinite(x) & np.isfinite(y)
            r2 = x*x + y*y
            values, finite_signal, masked = signal_values(f['signal'], slice(lo, lo+ntr))
            raw_signal_bad[c, :, 0] = masked.sum(axis=0)
            raw_signal_bad[c, :, 1] = ((~finite_signal) & ~masked).sum(axis=0)
            initial = good[:ntr]
            geo = initial & finite_coords[:ntr]
            usable = geo & finite_signal
            source = r2[:ntr] > radius**2
            admitted = usable & source
            exclusions[c, :, 0] = (~initial).sum(axis=0)
            exclusions[c, :, 1] = (initial & ~finite_coords[:ntr]).sum(axis=0)
            exclusions[c, :, 2] = (geo & ~finite_signal).sum(axis=0)
            exclusions[c, :, 3] = (usable & ~source).sum(axis=0)
            b[cursor:cursor+ntr] = np.where(admitted, values, np.nan)
            valid[cursor:cursor+ntr] = admitted
            near[cursor:cursor+ntr] = admitted & (r2[:ntr] <= 4*radius**2)
            del values
            finite_eval, masked_eval, nonfinite_eval = signal_finiteness(f['signal'], slice(lo+ntr, hi))
            raw_signal_bad[c, :, 2] = masked_eval.sum(axis=0)
            raw_signal_bad[c, :, 3] = nonfinite_eval.sum(axis=0)
            egeo = good[ntr:] & finite_coords[ntr:]
            admitted_eval = egeo & finite_eval
            eval_exclusions[c, :, 0] = (~good[ntr:]).sum(axis=0)
            eval_exclusions[c, :, 1] = (good[ntr:] & ~finite_coords[ntr:]).sum(axis=0)
            eval_exclusions[c, :, 2] = (egeo & ~finite_eval).sum(axis=0)
            eval_n[c] = admitted_eval.sum(axis=0)
            central_n[c] = (admitted_eval & (r2[ntr:] <= radius**2)).sum(axis=0)
            require(np.all(admitted.sum(axis=0) + exclusions[c].sum(axis=1) == ntr), 'Training census does not close')
            require(np.all(eval_n[c] + eval_exclusions[c].sum(axis=1) == hi-lo-ntr), 'Evaluation census does not close')
            cursor += ntr
        require(participating.sum() == prior['identity']['any_flag_zero_slots'], 'Participating flag population changed')
        require(np.isfinite(uid[participating]).all() and np.isfinite(nw[participating]).all(), 'Required detector identity nonfinite')
        require(len(np.unique(uid[participating])) == participating.sum(), 'Duplicate participating UID')
        require(np.isin(arr[participating], [0, 1, 2]).all(), 'Unknown participating array')
        audit = dict(obsnum=obs, input=item, signal_dtype=str(expected_dtype),
                     recovered_edges=edges, original_scan_indices=prior['reported_scan_indices'],
                     coordinate_max_difference_rad=coord_error, elapsed_span_seconds=float(t[-1]),
                     original_chunk_elapsed_spans_seconds=[float(t[hi-1]-t[lo]) for lo, hi in zip(edges[:-1], edges[1:])],
                     training_candidate_rows=len(tr), evaluation_candidate_rows=len(ev),
                     training_signal_access='candidate first halves only; statistics use eligible source-excluded rows',
                     evaluation_signal_access='masked/nonfinite predicate only; values not retained',
                     stored_weights_access=False, input_units='declared mJy/beam; no new conformity or physical-flux claim')
        write_json(run.out / f'{obs}_input_audit.json', audit)
    return dict(obsnum=obs, b=b, valid=valid, near=near, times=t[tr], chunk=chunk,
                early=early, eval_n=eval_n, central_n=central_n, arr=arr,
                uid=uid, nw=nw, participating=participating, edges=edges, all_times=t,
                exclusions=exclusions, eval_exclusions=eval_exclusions,
                raw_signal_bad=raw_signal_bad)


def read_plain_slice(f, key, rows):
    value = f[key][rows, :]
    require(not np.ma.getmaskarray(value).any(), 'Unexpected metadata mask: ' + key)
    return np.asarray(value)


def chunk_statistics(data, out):
    arrays = {key: [] for key in ('n', 'mu', 'v', 'early_n', 'early_mu', 'early_v', 'late_n', 'late_mu', 'late_v')}
    for c in range(12):
        rows = data['chunk'] == c
        for prefix, part in (('', rows), ('early_', rows & data['early']), ('late_', rows & ~data['early'])):
            for key, value in zip(('n', 'mu', 'v'), moments(data['b'][part], data['valid'][part])):
                arrays[prefix+key].append(value)
    arrays = {key: np.stack(value) for key, value in arrays.items()}
    good = (arrays['early_n'] >= 2) & (arrays['late_n'] >= 2) & (arrays['early_v'] > 0) & (arrays['late_v'] > 0)
    ratio = np.full(good.shape, np.nan)
    ratio[good] = np.log(arrays['late_v'][good] / arrays['early_v'][good])
    arrays['late_to_early_log_scatter_ratio'] = ratio
    np.savez_compressed(out / f"{data['obsnum']}_chunk_diagnostics.npz", **arrays,
                        evaluation_counts=data['eval_n'], central_evaluation_counts=data['central_n'],
                        exclusions=data['exclusions'], evaluation_exclusions=data['eval_exclusions'],
                        raw_signal_bad=data['raw_signal_bad'], column=np.arange(len(data['arr'])),
                        uid=data['uid'], network=data['nw'], array=data['arr'])
    return arrays


def group_statistics(data, chunk_stats, k):
    fields = {key: [] for key in ('n', 'mu', 'v', 'w', 'score_ok', 'e', 'central_e', 'near_n', 'near_mu', 'near_v',
                                  'far_n', 'far_mu', 'far_v', 'span', 'mean_change_fraction', 'child_scatter_ratio', 'empty_children')}
    scores, rows_list = [], []
    for j in range(12//k):
        rows = np.flatnonzero((data['chunk'] >= j*k) & (data['chunk'] < (j+1)*k))
        x, valid = data['b'][rows], data['valid'][rows]
        n, mu, v = moments(x, valid)
        w, c, score_ok = inverse_and_score(x, valid, (n, mu, v))
        for name, value in zip(('n', 'mu', 'v', 'w', 'score_ok'), (n, mu, v, w, score_ok)):
            fields[name].append(value)
        scores.append(c)
        rows_list.append(rows)
        fields['e'].append(data['eval_n'][j*k:(j+1)*k].sum(axis=0))
        fields['central_e'].append(data['central_n'][j*k:(j+1)*k].sum(axis=0))
        for prefix, m in (('near_', data['near'][rows]), ('far_', valid & ~data['near'][rows])):
            for name, value in zip(('n', 'mu', 'v'), moments(x, m)):
                fields[prefix+name].append(value)
        t = data['times'][rows, None]
        span = np.max(np.where(valid, t, -np.inf), axis=0)-np.min(np.where(valid, t, np.inf), axis=0)
        span[n == 0] = np.nan
        fields['span'].append(span)
        cn, cm, cv = (chunk_stats[key][j*k:(j+1)*k] for key in ('n', 'mu', 'v'))
        with np.errstate(invalid='ignore', divide='ignore'):
            between = np.where(cn > 0, cn*(cm-mu)**2, 0.).sum(axis=0)/n
            within = np.where(cn > 0, cn*cv, 0.).sum(axis=0)/n
            positive = np.isfinite(v) & (v > 0)
            require(np.allclose((between+within)[positive], v[positive], rtol=RTOL, atol=0), 'Pooled-scatter decomposition failed')
            fraction = np.where(positive, between/v, np.nan)
        ratio = np.full(v.shape, np.nan)
        ok = np.all((cn >= 2) & np.isfinite(cv) & (cv > 0), axis=0)
        if k > 1:
            ratio[ok] = np.max(cv[:, ok], axis=0)/np.min(cv[:, ok], axis=0)
        fields['mean_change_fraction'].append(fraction)
        fields['child_scatter_ratio'].append(ratio)
        fields['empty_children'].append((cn == 0).sum(axis=0))
    fields = {key: np.stack(value) for key, value in fields.items()}
    valid_ratio = (fields['near_n'] >= 2) & (fields['far_n'] >= 2) & (fields['near_v'] > 0) & (fields['far_v'] > 0)
    fields['near_to_far_log_scatter_ratio'] = np.full(fields['n'].shape, np.nan)
    fields['near_to_far_log_scatter_ratio'][valid_ratio] = np.log(fields['near_v'][valid_ratio]/fields['far_v'][valid_ratio])
    return fields, scores, rows_list


def share(mask, e, central, gamma):
    total, core = int(e.sum()), int(central.sum())
    return dict(groups=int(mask.sum()), evaluation_occurrences=int(e[mask].sum()),
                evaluation_share=float(e[mask].sum()/total) if total else None,
                central_evaluation_occurrences=int(central[mask].sum()),
                central_evaluation_share=float(central[mask].sum()/core) if core else None,
                coefficient_mass=float(np.sum(e[mask]*gamma[mask])/total)
                if total and np.isfinite(gamma[e > 0]).all() else None)


def analyze_windows(data, chunk_stats, settings, run):
    summaries = []
    taus = settings['correlation_spans_seconds']
    full_plans = [kernel_plan(data['times'], tau) for tau in taus]
    for k in settings['window_chunks']:
        run.check(f"{data['obsnum']}: K={k} pooled statistics")
        fields, scores, rows_list = group_statistics(data, chunk_stats, k)
        shape = fields['n'].shape
        fields['gamma'] = np.full(shape, np.nan)
        fields['p_raw'] = np.full(shape+(len(taus),), np.nan)
        fields['p_normalized'] = np.full(shape+(len(taus),), np.nan)
        fields['occupied_bins'] = np.zeros(shape+(len(taus),), np.int32)
        occupancy = {}
        for j, (c, rows) in enumerate(zip(scores, rows_list)):
            valid = data['valid'][rows]
            for h, tau in enumerate(taus):
                ids, counts = bin_counts(data['times'][rows], valid, tau)
                occupancy[f'window{j}_tau{h}_bin_ids'] = ids
                occupancy[f'window{j}_tau{h}_counts'] = counts
                fields['occupied_bins'][j, :, h] = (counts > 0).sum(axis=0)
                ok = fields['score_ok'][j] & (fields['e'][j] > 0)
                if ok.any():
                    variances = kernel_quad(c[:, ok], kernel_plan(data['times'][rows], tau))
                    fields['p_raw'][j, ok, h] = np.sqrt(variances)/fields['w'][j, ok]
            run.check()
        np.savez_compressed(run.out/f"{data['obsnum']}_K{k}_training_bins.npz", **occupancy)
        del occupancy
        for a, array in enumerate(ARRAYS):
            cols = np.flatnonzero(data['arr'] == a)
            e = fields['e'][:, cols]
            required = e > 0
            gamma, wbar, gamma_state, uncertainty_state = joint_state(fields['w'][:, cols], e, fields['score_ok'][:, cols])
            fields['gamma'][:, cols] = gamma
            if uncertainty_state == 'available':
                cbar = np.zeros(len(data['times']))
                for j, (c, rows) in enumerate(zip(scores, rows_list)):
                    cbar[rows] = np.sum(c[:, cols]*e[j][None, :], axis=1)/e.sum()
                for j, (c, rows) in enumerate(zip(scores, rows_list)):
                    active_cols = cols[required[j]]
                    for start in range(0, len(active_cols), settings['batch_columns']):
                        dc = active_cols[start:start+settings['batch_columns']]
                        g = fields['gamma'][j, dc]
                        influence = -cbar[:, None]*g[None, :]
                        influence[rows] += c[:, dc]
                        influence /= wbar
                        for h, plan in enumerate(full_plans):
                            fields['p_normalized'][j, dc, h] = np.sqrt(kernel_quad(influence, plan))/g
                        run.check()
            p = fields['p_normalized'][:, cols]
            raw_p = fields['p_raw'][:, cols]
            missing = required & ~np.isfinite(fields['w'][:, cols])
            unknown = required & ~np.isfinite(p).all(axis=2)
            over = required & np.any(np.isfinite(p) & (p > .2), axis=2)
            central = fields['central_e'][:, cols]
            earlylate = chunk_stats['late_to_early_log_scatter_ratio'][:, cols]
            summary = dict(obsnum=data['obsnum'], array=array, K=k,
                windows=12//k, required_groups=int(required.sum()),
                required_detector_columns=int(required.any(axis=0).sum()),
                evaluation_occurrences=int(e.sum()), central_evaluation_occurrences=int(central.sum()),
                no_training_groups=int(np.sum(required & (fields['n'][:, cols] == 0))),
                one_training_groups=int(np.sum(required & (fields['n'][:, cols] == 1))),
                two_training_groups=int(np.sum(required & (fields['n'][:, cols] == 2))),
                degenerate_or_unavailable_score_groups=int(np.sum(required & ~fields['score_ok'][:, cols])),
                weight_state=gamma_state, normalized_uncertainty_state=uncertainty_state,
                weight_unavailable=share(missing, e, central, gamma),
                normalized_uncertainty_unavailable=share(unknown, e, central, gamma),
                any_normalized_estimate_above_goal=share(over, e, central, gamma),
                training_count_quantiles_10_50_90=quantiles(fields['n'][:, cols][required]),
                training_span_quantiles_seconds_10_50_90=quantiles(fields['span'][:, cols][required]),
                exclusion_scope='All array columns, including inactive slots; sequential reasons are disjoint within each half.',
                training_exclusions=data['exclusions'][:, cols].sum(axis=(0, 1)),
                evaluation_exclusions=data['eval_exclusions'][:, cols].sum(axis=(0, 1)),
                raw_signal_masked_nonfinite_counts=data['raw_signal_bad'][:, cols].sum(axis=(0, 1)),
                child_scatter_ratio_quantiles_10_50_90=quantiles(fields['child_scatter_ratio'][:, cols][required], e[required]),
                mean_change_fraction_quantiles_10_50_90=quantiles(fields['mean_change_fraction'][:, cols][required], e[required]),
                radial_log_scatter_ratio_quantiles_10_50_90=quantiles(fields['near_to_far_log_scatter_ratio'][:, cols][required], e[required]),
                late_early_log_scatter_ratio_quantiles_10_50_90=quantiles(earlylate, data['eval_n'][:, cols]),
                precision=[dict(tau_seconds=tau, normalized=series_summary(p[:, :, h], e),
                                unnormalized=series_summary(raw_p[:, :, h], e),
                                occupied_bin_quantiles_10_50_90=quantiles(fields['occupied_bins'][:, cols, h][required]))
                           for h, tau in enumerate(taus)],
                finite_data_coverage='unavailable; all precision estimates descriptive')
            # Sensitivity is defined only when all six prescribed values exist.
            minp = np.full(e.shape, np.nan); maxp = minp.copy()
            complete = required & np.isfinite(p).all(axis=2)
            if complete.any():
                minp[complete] = p[complete].min(axis=1)
                maxp[complete] = p[complete].max(axis=1)
            summary['sensitivity_min_p_quantiles_10_50_90'] = quantiles(minp, e)
            summary['sensitivity_max_p_quantiles_10_50_90'] = quantiles(maxp, e)
            summary['all_tau_at_or_below_goal_groups'] = int(np.sum(complete & (maxp <= .2)))
            summaries.append(summary)
            run.check(f"{data['obsnum']}: K={k} {array}: {gamma_state}, uncertainty={uncertainty_state}")
        np.savez_compressed(run.out/f"{data['obsnum']}_K{k}_groups.npz", **fields,
                            column=np.arange(len(data['arr'])), uid=data['uid'], network=data['nw'], array=data['arr'])
        write_json(run.out/f"{data['obsnum']}_K{k}_summary.json", [s for s in summaries if s['K'] == k])
        del scores, fields
    return summaries


def lag_matrices(times, chunks, edges):
    delta = times[None, :] - times[:, None]
    same = chunks[:, None] == chunks[None, :]
    return [[sparse.csr_matrix(((delta > 0) & (delta >= lo) & (delta < hi) & (same if category == 0 else ~same)).astype(float))
             for category in range(2)] for lo, hi in zip(edges[:-1], edges[1:])]


def pair_statistics(series, valid, adjacency):
    x = np.where(valid, series, 0.)
    m = valid.astype(float)
    am = adjacency @ m
    pair_n = np.sum(m*am, axis=0).astype(np.int64)
    numerator = np.sum(x*(adjacency @ x), axis=0)
    left2 = np.sum(x*x*am, axis=0)
    right2 = np.sum(m*(adjacency @ (x*x)), axis=0)
    corr = np.full(x.shape[1], np.nan)
    good = (pair_n > 0) & (left2 > 0) & (right2 > 0)
    corr[good] = numerator[good]/np.sqrt(left2[good]*right2[good])
    require(np.all(np.abs(corr[good]) <= 1 + RTOL), 'Pair correlation out of range')
    return pair_n, corr


def analyze_lags(data, chunk_stats, settings, run):
    matrices = lag_matrices(data['times'], data['chunk'], settings['lag_edges_seconds'])
    nd = len(data['arr'])
    pair_n = np.zeros((nd, 7, 2), np.int64)
    corr = np.full((nd, 7, 2, 2), np.nan)
    columns = np.flatnonzero(data['eval_n'].sum(axis=0) > 0)
    for start in range(0, len(columns), settings['batch_columns']):
        dc = columns[start:start+settings['batch_columns']]
        mu = chunk_stats['mu'][:, dc][data['chunk']]
        v = chunk_stats['v'][:, dc][data['chunk']]
        n = chunk_stats['n'][:, dc][data['chunk']]
        valid = data['valid'][:, dc] & (n >= 2) & np.isfinite(v) & (v > 0)
        z = np.zeros(valid.shape)
        np.divide(np.where(valid, data['b'][:, dc]-mu, 0.), np.sqrt(np.where(valid, v, 1.)), out=z)
        q = np.where(valid, z*z-1, 0.)
        for h, per_category in enumerate(matrices):
            for category, mat in enumerate(per_category):
                count, corr_z = pair_statistics(z, valid, mat)
                _, corr_q = pair_statistics(q, valid, mat)
                pair_n[dc, h, category] = count
                corr[dc, h, category, 0] = corr_z
                corr[dc, h, category, 1] = corr_q
        run.check()
    np.savez_compressed(run.out/f"{data['obsnum']}_lag_diagnostics.npz", pair_counts=pair_n,
                        correlations=corr, column=np.arange(nd), uid=data['uid'], network=data['nw'], array=data['arr'])
    summaries = []
    for a, array in enumerate(ARRAYS):
        cols = (data['arr'] == a) & (data['eval_n'].sum(axis=0) > 0)
        for h, (lo, hi) in enumerate(zip(settings['lag_edges_seconds'][:-1], settings['lag_edges_seconds'][1:])):
            for category, name in enumerate(('within_chunk', 'cross_chunk')):
                summaries.append(dict(obsnum=data['obsnum'], array=array, lower_seconds=lo, upper_seconds=hi,
                    category=name, participating_detector_columns=int(cols.sum()),
                    detector_columns_with_pairs=int(np.sum(pair_n[cols, h, category] > 0)),
                    pair_count_quantiles_10_50_90=quantiles(pair_n[cols, h, category]),
                    centered_signal_correlation_quantiles_10_50_90=quantiles(corr[cols, h, category, 0]),
                    squared_signal_correlation_quantiles_10_50_90=quantiles(corr[cols, h, category, 1])))
    write_json(run.out/f"{data['obsnum']}_lag_summary.json", summaries)
    run.check(f"{data['obsnum']}: lag diagnostics complete")
    return summaries


def plot_results(summaries, run):
    os.environ['MPLCONFIGDIR'] = '/private/tmp/fruit-q02-precision-mpl-20260909'
    os.environ['XDG_CACHE_HOME'] = '/private/tmp/fruit-q02-precision-cache-20260909'
    Path(os.environ['MPLCONFIGDIR']).mkdir(exist_ok=True)
    Path(os.environ['XDG_CACHE_HOME']).mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    colors = ['#1976a3', '#c66922', '#487747']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for ax, obs in zip(axes, (123424, 152389)):
        for array, color in zip(ARRAYS, colors):
            rows = sorted((s for s in summaries if s['obsnum'] == obs and s['array'] == array), key=lambda s:s['K'])
            ax.plot([s['K'] for s in rows], [100*s['weight_unavailable']['evaluation_share'] for s in rows], 'o-', label=array, color=color)
        ax.set(title=str(obs), xlabel='Original chunks per training window', xticks=[1,2,4])
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Evaluation occurrences lacking a weight (%)')
    fig.suptitle('Training support: no detector removal or fallback')
    fig.tight_layout()
    fig.savefig(run.out/'training_availability.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row, obs in enumerate((123424, 152389)):
        for col, array in enumerate(ARRAYS):
            ax = axes[row, col]
            for k, color in zip((1,2,4), colors):
                s = next(s for s in summaries if s['obsnum'] == obs and s['array'] == array and s['K'] == k)
                x = [p['tau_seconds'] for p in s['precision']]
                y = [p['unnormalized']['occurrence_weighted_quantiles_10_50_90'][1] for p in s['precision']]
                ax.plot(x, 100*np.array(y, dtype=float), 'o-', label=f'{k} chunk(s)', color=color)
            ax.axhline(20, color='#777777', linestyle=':', linewidth=1)
            ax.set(title=f'{obs} · {array}', xscale='log', xticks=[.125,.25,.5,1,2,4])
            ax.set_xticklabels(['.125','.25','.5','1','2','4'])
            ax.grid(alpha=.15)
            if row == 1: ax.set_xlabel('Correlation span (s)')
            if col == 0: ax.set_ylabel('Estimated fractional SD (%)')
    axes[0,0].legend(frameon=False, fontsize=9)
    fig.suptitle('Unnormalized inverse scatter: occurrence-weighted median\nAvailable groups only; this is not the normalized 20% target or a confidence bound', fontsize=11)
    fig.tight_layout()
    fig.savefig(run.out/'unnormalized_precision_sensitivity.png', dpi=180)
    plt.close(fig)
    run.check('Figures written')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    require(args.attempt.startswith('attempt_') and '/' not in args.attempt and '\\' not in args.attempt, 'Invalid attempt name')
    out = HERE / args.attempt
    out.mkdir(exist_ok=False)
    settings = json.loads((HERE/'settings.json').read_text())
    run = Run(out, settings)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(BoundExceeded('Two-hour wall-clock limit')))
    signal.alarm(settings['wall_time_limit_seconds'])
    try:
        require(digest(Q/'r0.5/WEIGHT_PRECISION_PROTOCOL.md') == settings['protocol_sha256'], 'Approved protocol changed')
        review = json.loads((Q/'r0.5/REVIEW_MANIFEST.json').read_text())
        for key, value in review['proposed_settings'].items():
            require(settings[key] == value, 'Approved setting changed: '+key)
        require(settings['inputs'] == sorted(review['inputs_recovered_from_prior_preflight'], key=lambda x:x['obsnum']), 'Approved inputs changed')
        require(settings['guard_arcsec'] == {'123424':28.59742792952955,'152389':27.38359774550894}, 'Approved guards changed')
        require(settings['edges'] == [0,289,594,899,1204,1509,1814,2119,2424,2729,3034,3339,3628], 'Approved edges changed')
        require(settings['lag_edges_seconds'] == [0,.125,.25,.5,1,2,4,8], 'Approved lag bins changed')
        sources = [Path(__file__).resolve(), HERE/'settings.json', HERE/'test_weight_precision.py',
                   Q/'r0.5/WEIGHT_PRECISION_PROTOCOL.md', Q/'WEIGHT_PRECISION_APPROVAL_2026-09-09.md']
        identities = [dict(path=str(p.relative_to(REPO)), bytes=p.stat().st_size, sha256=digest(p)) for p in sources]
        from test_weight_precision import verify
        with threadpool_limits(limits=1):
            verified = verify(sys.modules[__name__])
            write_json(out/'DETERMINISTIC_VERIFICATION.json', verified)
            write_json(out/'RUN_START.json', dict(started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                       decision=settings['decision'], sources=identities, settings=settings,
                       verification='PASS', python=sys.version,
                       environment={name:importlib.metadata.version(name) for name in ('numpy','scipy','netCDF4','matplotlib','threadpoolctl')},
                       threadpools=threadpool_info(), numerical_threads=1,
                       scientific_signal_access_at_record_time=False))
            # Both identities must pass before either scientific signal is read.
            for item in settings['inputs']:
                path = Path(item['path'])
                require(path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], 'Input identity mismatch: '+str(item['obsnum']))
            run.check('Both input hashes and deterministic verification passed; begin approved access')
            summaries, lag_summaries = [], []
            for item in settings['inputs']:
                data = load_observation(item, settings, run)
                chunks = chunk_statistics(data, out)
                summaries.extend(analyze_windows(data, chunks, settings, run))
                lag_summaries.extend(analyze_lags(data, chunks, settings, run))
                del data, chunks
            write_json(out/'SUMMARY.json', dict(status='complete', windows=summaries, lags=lag_summaries,
                       finite_data_coverage='unavailable', reserved_129081_access=False, map_or_reduction_executed=False,
                       quadratic_form_roundoff_adjustments=0,
                       computation='Bartlett overlap integral; nonnegative sum exactly represents the approved kernel'))
            plot_results(summaries, run)
            run.check('Diagnostic complete')
        write_json(out/'COMPLETION.json', dict(status='complete', elapsed_seconds=time.monotonic()-run.start,
                   peak_aggregate_process_bytes=peak_bytes(), subprocesses=0, numerical_threads=1,
                   scientific_changes='none beyond the approved diagnostic', finite_data_coverage='unavailable'))
    except BaseException as exc:
        write_json(out/'COMPLETION.json', dict(status='incomplete', error_type=type(exc).__name__, error=str(exc),
                   elapsed_seconds=time.monotonic()-run.start, peak_aggregate_process_bytes=peak_bytes()))
        (out/'FAILURE_TRACEBACK.txt').write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        products = [dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=digest(p))
                    for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('PRODUCT_MANIFEST.json','PRODUCT_MANIFEST.json.sha256')]
        manifest = out/'PRODUCT_MANIFEST.json'
        write_json(manifest, dict(products=products, payload_bytes=sum(p['bytes'] for p in products),
                   external_inputs_modified=False, output_root=str(out)))
        (out/'PRODUCT_MANIFEST.json.sha256').write_text(digest(manifest)+'  PRODUCT_MANIFEST.json\n')


if __name__ == '__main__':
    main()
