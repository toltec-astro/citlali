#!/usr/bin/env python3
"""Exploratory spectral/transient audit; no production flags or treatment plans.

Full pooled PSD/background arrays remain separate from this conditional census.
All time costs are paired detector-time unions, never sums of x and r costs.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import numpy as np
from rtc_disturbance_burden import (records, digest, write_json, compressed_writer,
                                  emit, union, intersect, subtract, measure)


THRESHOLDS = (.1, .2, .3)


def independent_windows(windows):
    """Greedy native-row disjoint subsequence, including physical gaps."""
    selected, end = [], -1
    for i, row in enumerate(windows):
        if int(row[2]) >= end:
            selected.append(i)
            end = int(row[3])
    return np.asarray(selected, dtype=int)


def cover_scans(intervals, duration_us, phase_us=0):
    """Conditional fixed-duration scan model. These are NOT actual PCA scans."""
    if duration_us <= 0:
        raise ValueError('nonpositive hypothetical scan duration')
    spans = []
    for a, b in union(intervals):
        first = (a-phase_us)//duration_us
        last = (b-1-phase_us)//duration_us
        spans.append((phase_us+first*duration_us, phase_us+(last+1)*duration_us))
    return union(spans)


def incremental_cost(eligible, baseline, proposed):
    retained = subtract(eligible, baseline)
    return measure(intersect(retained, proposed))


def noise_factor(information, lost):
    if not math.isfinite(information) or information <= 0 or not 0 <= lost <= information:
        return None
    return math.sqrt(information/(information-lost)) if lost < information else None


def row_support(rows, cells):
    """Preserve each physical cell and gap; no elapsed-time substitution."""
    spans = []
    for a, b in rows:
        a, b = int(a), int(b)
        if not 0 <= a < b <= len(cells):
            raise ValueError('window outside exact native cells')
        starts = np.r_[a, np.flatnonzero(cells[a+1:b, 2] > cells[a:b-1, 3])+a+1]
        ends = np.r_[starts[1:], b]
        spans.extend((int(cells[x, 2]), int(cells[y-1, 3])) for x, y in zip(starts, ends))
    return union(spans)


def describe_coordinate(meta, arrays, windows, frequency):
    available = meta['cause'] in (0, 1)
    result = dict(available=available, cause=meta['cause'], windows=len(windows),
                  independent_windows=0, narrow_fraction=0., narrow_frequency_hz=None,
                  narrow_contrast=None, narrow_width_hz=None, recurrence=None,
                  low_frequency_fraction=None, broad_or_crowded_review=False,
                  profile_sensitive_review=False, burst_ratio=None)
    if not available:
        return result, {}
    psd = arrays[0]; total = float(np.sum(psd))
    if not np.isfinite(psd).all() or total <= 0:
        return result, {}
    result['low_frequency_fraction'] = float(psd[frequency < 2].sum()/total)
    df = float(frequency[1]-frequency[0])
    # Exploratory screens only. 2 Hz is an audit grouping boundary, not an
    # optical-safe band. Complete extents with high contrast and <=2 Hz width.
    def eligible_regions(profile):
        return [r for r in meta[f'regions_{profile}'] if frequency[r[2]] >= 2
                and (r[1]-r[0])*df <= 2 and not r[6]
                and r[5] is not None and r[5] >= 10 and r[4] is not None]
    good = eligible_regions(2)
    best = max(good, key=lambda r: r[4], default=None)
    all_high = [r for r in meta['regions_2'] if frequency[r[2]] >= 2 and r[4] is not None]
    producer_target = max(all_high, key=lambda r: r[3], default=None)
    ids = independent_windows(windows)
    result['independent_windows'] = len(ids)
    if len(ids):
        power = windows[ids, 4]
        median = float(np.median(power))
        result['burst_ratio'] = float(np.quantile(power, .99)/median) if median > 0 else None
    selections = {}
    if best:
        target = int(best[2])
        result.update(narrow_fraction=float(best[4]), narrow_frequency_hz=float(frequency[target]),
                      narrow_contrast=float(best[5]), narrow_width_hz=(best[1]-best[0])*df)
        # The exporter retains the strongest >=2 Hz region's 3-bin power, plus
        # each window's largest 3-bin band. Never substitute a different target.
        if producer_target and target == producer_target[2]:
            fractions = windows[:, 9]
            result['recurrence'] = float(np.mean(fractions[ids] >= .1)) if len(ids) else None
            for threshold in THRESHOLDS:
                selections[threshold] = windows[fractions >= threshold, 2:4].astype(int).tolist() if best[4] >= threshold else []
        else:
            result['recurrence_unavailable_reason'] = 'narrow region differs from exported pooled target'
    for threshold in THRESHOLDS:
        selections.setdefault(threshold, [])
    if len(good) >= 3 and sum(r[4] for r in good) >= .1:
        result['broad_or_crowded_review'] = True
    high4 = [r for r in meta['regions_4'] if frequency[r[2]] >= 2 and r[4] is not None]
    strongest2 = max((r[4] for r in all_high), default=0)
    strongest4 = max((r[4] for r in high4), default=0)
    result['profile_sensitive_review'] = bool(strongest4 >= .1 and strongest4 > 2*strongest2)
    result['high_region_fraction_2'] = strongest2
    result['high_region_fraction_4'] = strongest4
    return result, selections


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--burden', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    invocations = sorted(records(a.campaign/'invocations.jsonl'), key=lambda r: (r['observation'], r['network']))
    if len({(x['observation'],x['network']) for x in invocations}) != len(invocations):
        raise ValueError('duplicate input invocation')
    accounting = {(r['observation'],r['network'],r['detector']):r for r in records(a.burden/'detector-accounting.jsonl.gz')}
    selections = json.loads(a.selection.read_text())
    independent = {(s['observation'],s['network'],s['channel']) for s in selections if s['selection']=='independent-hash'}
    totals = Counter(); cohorts=defaultdict(Counter); costs=defaultdict(lambda:defaultdict(float)); families=defaultdict(Counter)
    per_observation=defaultdict(Counter); candidates=[]; time_concentration=[]; inspection=[]
    with compressed_writer(a.output/'detectors.jsonl.gz') as output:
        for entry in invocations:
            if entry['exit_status'] or not entry['exact_prior_input_binding'] or not entry['input_metadata_unchanged']:
                raise ValueError('unbound/failed invocation')
            obs,nw=entry['observation'],entry['network'];folder=a.campaign/f'{obs}-{nw:02d}'
            receipt=json.loads((folder/'receipt.json').read_text())
            if receipt['Apply'] or receipt['VAL_generation'] != 0 or receipt['original_pair_fingerprint_before'] != receipt['original_pair_fingerprint_after']:
                raise ValueError('initial original evidence contract failed')
            meta=list(records(folder/'spectra.jsonl'))
            spectral=np.memmap(folder/'spectra.f64',dtype='<f8',mode='r',shape=(receipt['channels'],2,4,receipt['bins']))
            win=np.memmap(folder/'windows.f64',dtype='<f8',mode='r',shape=(receipt['window_records'],14))
            with gzip.open(a.burden/'native-time'/f'{obs}-{nw:02d}.jsonl.gz','rt') as stream:
                timing=json.loads(next(stream));cells=np.array([json.loads(l) for l in stream],dtype=np.int64)
            if timing['raw_sha256'] != receipt['raw_sha256'] or len(cells)!=receipt['rows']:
                raise ValueError('native support belongs to different raw input')
            native=np.fromfile(folder/'native-time.f64',dtype='<f8')
            if np.max(np.abs(native*1e6-cells[:,1])) > 1:
                raise ValueError('native time does not match preserved integer cells')
            frequency=np.arange(receipt['bins'])/(receipt['fft_samples']*receipt['measured_interval'])
            for d in range(receipt['channels']):
                key=(obs,nw,d);old=accounting[key];m=meta[2*d]
                if any(mm['detector'] != d or mm['coordinate'] != c or mm['occurrence'] != old['occurrence'] or mm['tune_valid'] != old['tune_valid'] for c,mm in enumerate(meta[2*d:2*d+2])):
                    raise ValueError('exact detector/coordinate/occurrence join failed')
                eligible=old['eligible_intervals_us'];noise=old['intervals_us']['noise_screening_required'];direct=old['intervals_us']['direct'];candidate=old['intervals_us']['candidate_inclusive']
                exposure=measure(eligible);base=measure(subtract(eligible,noise));apt_good=m['apt_flag']==0 and m['apt_flag2']==0
                if apt_good != old['peer_eligible'] and old['tune_valid']:
                    raise ValueError('APT quality relation differs from preserved census')
                sens=m['apt_sens'];weighted=bool(apt_good and sens is not None and math.isfinite(sens) and sens>0);weight=1/sens**2 if weighted else 0.
                array=old['array'];cohort=cohorts[array];cohort['producer_eligible_us']+=exposure;cohort['baseline_us']+=base;cohort['apt_good_us']+=base if apt_good else 0
                cohort['weighted_coverage_us']+=base if weighted else 0;cohort['information_proxy']+=base/1e6*weight
                totals['detector_occurrences']+=1;totals['eligible_occurrences']+=exposure>0;totals['eligible_us']+=exposure;totals['baseline_us']+=base
                descriptions=[];active=[]
                for c in range(2):
                    mm=meta[2*d+c];start=mm['window_offset'];ww=win[start:start+mm['window_count']]
                    if len(ww) and (np.any(ww[:,0]!=d) or np.any(ww[:,1]!=c)):
                        raise ValueError('window offset or coordinate mismatch')
                    description,sel=describe_coordinate(mm,spectral[d,c],ww,frequency)
                    descriptions.append(description);active.append({t:row_support(sel[t],cells) for t in THRESHOLDS} if sel else {t:[] for t in THRESHOLDS})
                    totals[f'available_{c}']+=description['available']
                row=dict(observation=obs,network=nw,detector=d,array=array,occurrence=old['occurrence'],
                         eligible_us=exposure,baseline_us=base,apt_good=apt_good,sens=sens,weighted=weighted,
                         independent_selection=key in independent,prior_health_concern=old['health_review_concern'],
                         coordinates=descriptions,exploratory_only=True,Apply=False,losses={})
                for threshold in THRESHOLDS:
                    for scope in ('x','paired'):
                        axes=(0,) if scope=='x' else (0,1)
                        chosen=any(descriptions[c]['narrow_fraction']>=threshold for c in axes)
                        name=f'{scope}_narrow_{int(threshold*100)}'
                        intervals=union(i for c in axes for i in active[c][threshold])
                        if not chosen: continue
                        totals[name+'_occurrences']+=exposure>0
                        per_observation[obs][name+'_occurrences']+=exposure>0
                        direct_overlap=measure(intersect(intervals,direct));candidate_overlap=measure(intersect(intervals,candidate));noise_overlap=measure(intersect(intervals,noise))
                        proposed=dict(observation=eligible,active_windows=intervals)
                        for duration in (5,10,20):
                            for phase in (0,duration/2):proposed[f'scan_model_{duration}s_phase{phase:g}']=cover_scans(intervals,int(duration*1e6),int(phase*1e6))
                        loss={mode:incremental_cost(eligible,noise,bounds) for mode,bounds in proposed.items()}
                        row['losses'][name]=dict(incremental_us=loss,active_us=measure(intersect(eligible,intervals)),direct_overlap_us=direct_overlap,candidate_overlap_us=candidate_overlap,noise_overlap_us=noise_overlap,recurrence_bound=all(descriptions[c].get('recurrence') is not None for c in axes if descriptions[c]['narrow_fraction']>=threshold))
                        for mode,value in loss.items():
                            costs[(name,mode,array)]['lost_us']+=value
                            costs[(name,mode,array)]['apt_good_lost_us']+=value if apt_good else 0
                            costs[(name,mode,array)]['lost_information']+=value/1e6*weight
                            # Existing direct support is a conditional comparison,
                            # not accepted production exclusion.
                            costs[(name,mode,array)]['incremental_after_direct_us']+=incremental_cost(eligible,union(noise+direct),proposed[mode])
                        if name=='x_narrow_10':
                            totals['x10_active_us']+=measure(intersect(eligible,intervals));totals['x10_direct_overlap_us']+=direct_overlap;totals['x10_candidate_overlap_us']+=candidate_overlap;totals['x10_noise_overlap_us']+=noise_overlap
                            recurrence=descriptions[0]['recurrence'];totals['x10_recurrence_unavailable']+=recurrence is None
                            totals['x10_persistent_occurrences']+=recurrence is not None and recurrence>=.6 and descriptions[0]['independent_windows']>=3
                            f=descriptions[0]['narrow_frequency_hz'];binid=int(round(f/(frequency[1]-frequency[0])))
                            families[binid]['occurrences']+=exposure>0;families[binid]['eligible_us']+=exposure;families[binid][f'obs_{obs}']+=1;families[binid][f'nw_{nw}']+=1
                            time_concentration.append((exposure,obs,nw,d));per_observation[obs]['x10_eligible_us']+=exposure
                for c,desc in enumerate(descriptions):
                    for name in ('broad_or_crowded_review','profile_sensitive_review'):
                        totals[f'{name}_{c}']+=bool(desc[name] and exposure)
                if descriptions[0]['low_frequency_fraction'] is not None:totals['low_frequency_dominated_x']+=descriptions[0]['low_frequency_fraction']>=.8 and exposure>0
                if key in independent or any(folder.glob(f'samples-{d}.f64')):
                    inspection.append(dict(observation=obs,network=nw,detector=d,independent=key in independent,apt_good=apt_good,eligible=exposure>0,coordinates=descriptions))
                if exposure and (descriptions[0]['narrow_fraction']>=.1 or descriptions[0]['broad_or_crowded_review'] or descriptions[0]['profile_sensitive_review']):
                    candidates.append(dict(observation=obs,network=nw,detector=d,coordinates=descriptions,apt_good=apt_good))
                emit(output,row)
            per_observation[obs]['files']+=1
            print(f'Analyzed {obs}-{nw:02d}',flush=True)
    cost_rows=[]
    for (scenario,mode,array),v in sorted(costs.items()):
        c=cohorts[array];v=dict(v);v.update(scenario=scenario,mode=mode,array=array,
            lost_detector_time_fraction=v['lost_us']/c['baseline_us'] if c['baseline_us'] else None,
            lost_information_fraction=v['lost_information']/c['information_proxy'] if c['information_proxy'] else None,
            rms_noise_factor=noise_factor(c['information_proxy'],v['lost_information']))
        cost_rows.append(v)
    ranked=sorted(time_concentration,reverse=True);concentration={}
    for fraction in (.01,.05,.1):
        n=max(1,math.ceil(totals['eligible_occurrences']*fraction));concentration[str(fraction)]=dict(detectors=n,selected_exposure_us=sum(v[0] for v in ranked[:n]),all_selected_exposure_us=sum(v[0] for v in ranked))
    summary=dict(files=len(invocations),observations=len(per_observation),totals=dict(totals),
        cohorts={k:dict(v) for k,v in cohorts.items()},costs=cost_rows,concentration=concentration,
        per_observation={str(k):dict(v) for k,v in per_observation.items()},
        families=[dict(bin=k,frequency_hz=k/(488*.008192062377929688),**dict(v)) for k,v in sorted(families.items(),key=lambda kv:kv[1]['eligible_us'],reverse=True)],
        method=dict(thresholds=THRESHOLDS,contrast=10,maximum_width_hz=2,audit_low_frequency_boundary_hz=2,
            recurrence='disjoint accepted windows; fraction with pooled-target three-bin raw power >=10% of all stored PSD power; >=60% descriptive persistent group',
            windows='original accepted 4s Hann windows; time-local arithmetic independently reproduces pooled evidence; a single window is not separately qualified RTC evidence',
            scan_cost='conditional 5/10/20-second fixed cells, two phases; actual native-to-PCA association unavailable',
            sensitivity='per-array static sum(t/sens^2) for finite positive matched APT sens and flag=flag2=0; fixed independent-noise proxy, no PCA/maps; uncovered exposure retained in detector-time denominator',
            exclusion='baseline existing noise-screening-required; measured transient direct/candidate overlap separately conditional; no accepted source-bound corpus Apply',
            classifications='exploratory descriptor screens; not physical labels, notch admission or production flags',
            cadence='conditional producer-clock arithmetic within four epoch-double ULPs; no physical timing or operational jitter qualification'),
        bindings=dict(burden_accounting_sha256=digest(a.burden/'detector-accounting.jsonl.gz'),
                      campaign_invocations_sha256=digest(a.campaign/'invocations.jsonl'),selection_sha256=digest(a.selection)),Apply=False)
    write_json(a.output/'summary.json',summary);write_json(a.output/'inspection-index.json',inspection)
    write_json(a.output/'descriptor-selections.json',candidates)


if __name__=='__main__':main()
