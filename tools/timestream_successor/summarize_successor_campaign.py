#!/usr/bin/env python3
"""Compact population/cost accounting from the fixed campaign's preserved products."""
import argparse
import json
from pathlib import Path
import yaml
from ptc_spod import digest, require


def read_yaml(path):
    with path.open() as f:return yaml.load(f,Loader=yaml.CSafeLoader)


def residual_summary(profile):
    """Copy fixed-frequency evidence; do not rank peaks or pool network patterns."""
    r=profile.get('matched_residual') or {}
    return dict(fft_seconds=profile.get('requested_fft_seconds'),state=r.get('state','unavailable'),
        matched_windows=len(r.get('window_indices',[])),control_windows=len(r.get('control_window_indices',[])),
        control_state=r.get('control_state','unavailable'),
        folds=[dict(fold=f['fold'],state=f['state'],training_windows=len(f['pattern_training_window_indices']),
            evaluation_windows=len(f['evaluation_window_indices']),
            controlled_evaluation_windows=len(f.get('controlled_evaluation_window_indices',[])),
            frequency=f.get('frequency',[]),controlled_pattern=f.get('controlled_pattern',[]))
            for f in r.get('separate_pattern_folds',[])])


def array_coverage(coverage):
    result=[]
    for array in sorted({n.get('array','unbound') for n in coverage}):
        members=[n for n in coverage if n.get('array','unbound')==array]
        requested=[n for n in members if n['requested']]
        row=dict(array=array,networks=[n['network'] for n in members],
            missing_raw_networks=[n['network'] for n in members if not n['raw_available']],
            requested_networks=[n['network'] for n in requested],
            supplied_detectors=sum(n['supplied_detectors'] for n in members),
            requested_detectors=sum(n['requested_detectors'] for n in requested))
        # Unknown completion is not zero availability. Keep partial totals and
        # their assessed network identities explicit rather than hiding gaps.
        for field in ('CAL_supported_detectors','PTC_supported_detectors','CAL_detector_seconds',
                      'PTC_detector_seconds','diagnostic_population','supplemental_population'):
            assessed=[n for n in requested if n.get(field) is not None]
            row[field]=dict(total=sum(n[field] for n in assessed),
                assessed_networks=[n['network'] for n in assessed],complete=len(assessed)==len(requested))
        row['interpretation']='population/support totals only; patterns, cadences and residual power remain network-specific'
        result.append(row)
    return result


def summarize(root,inventory):
    inv=json.loads(inventory.read_text());campaign=json.loads((root/'CAMPAIGN.json').read_text())
    coverage=[];costs=[]
    requests={n['network']:n for n in read_yaml(root/'workers-1/products/observation-receipt.yaml')['networks']}
    for n in requests.values():require(digest(n['input']['path'])==n['input']['sha256'],'requested population binding changed')
    for network in inv['networks']:
        nw=network['network'];row={k:v for k,v in network.items() if k in ('network','array','raw_available','requested','requested_detectors','native_rows','output_sampling_hz','reason')}
        row['supplied_detectors']=row.get('requested_detectors',0)
        row['requested']=nw in requests
        row['requested_detectors']=len(json.loads(Path(requests[nw]['input']['path']).read_text())['detectors']) if nw in requests else 0
        row['execution_state']=requests[nw]['state'] if nw in requests else 'not-requested'
        product=root/'workers-1/products'/f'network{nw}';calpath=product/'donor-continuity/cal/receipt.yaml';ptcpath=product/'donor-continuity/ptc/receipt.yaml'
        if calpath.exists():
            cal=read_yaml(calpath);row.update(CAL_supported_detectors=sum(d['available']>0 for d in cal['detectors']),
                CAL_available=cal['available'],CAL_detector_seconds=cal['available']/network['output_sampling_hz'])
        if ptcpath.exists():
            ptc=read_yaml(ptcpath);row.update(PTC_available=ptc['available'],PTC_detector_seconds=ptc['available']/network['output_sampling_hz'],
                PTC_failed_fits=ptc['failed_fits'],PTC_total_fits=len(ptc['segments']),rank=ptc['rank'],method=ptc['method'])
        diag=root/f'diagnostic-network{nw}/result.json'
        if diag.exists():
            d=json.loads(diag.read_text());row['diagnostic_population']=len(d['cohort_columns']);row['diagnostic_state']=d.get('state','available-profile-specific')
            row['PTC_supported_detectors']=(d.get('PTC_support') or {}).get('represented_detectors')
            row['diagnostic_support']=[{k:p.get(k) for k in ('requested_fft_seconds','state','common_seconds','unique_Fourier_support_seconds')} for p in d['profiles']]
            row['supplemental_population']=len(d.get('temporal_supplement',{}).get('columns',[]))
            row['observation_seconds']=d.get('observation_seconds')
            row['matched_residual']=[residual_summary(p) for p in d['profiles']]
            row['supplemental_support']=[{k:p.get(k) for k in ('requested_fft_seconds','state','common_seconds','unique_Fourier_support_seconds')}
                for p in d.get('temporal_supplement',{}).get('profiles',[])]
        coverage.append(row)
    for run in sorted(list(root.glob('workers-*'))+list(root.glob('repeat-*-workers-*'))):
        observation=read_yaml(run/'products/observation-receipt.yaml');metrics=json.loads((run/'RUN.json').read_text())
        row=dict(run=run.name,repeat=run.name.startswith('repeat-'),workers=metrics['workers'],wall_seconds=metrics['wall_seconds'],peak_rss_bytes=metrics['polled_peak_rss_bytes'],
            output_bytes=metrics['output_bytes'],CPU_sampled_seconds=max((s['user_seconds']+s['system_seconds'] for s in metrics['samples']),default=0),
            OS_threads_max=max((s['os_threads'] for s in metrics['samples']),default=0),
            started_network_workers=observation.get('started_network_workers'),
            observed_Eigen_threads=observation['observed_Eigen_threads'],shared_preparation_seconds=observation['shared_preparation_seconds'],networks=[])
        for n in observation['networks']:
            trace=run/'products'/f"network{n['network']}"/'performance.jsonl';stages=[];last=0.
            if trace.exists():
                for line in trace.read_text().splitlines():
                    x=json.loads(line)
                    if 'stage' in x:
                        stages.append(dict(stage=x['stage'],interval_wall_seconds=x['seconds']-last));last=x['seconds']
            ptc_cost=None
            receipt=run/'products'/f"network{n['network']}"/'donor-continuity/ptc/receipt.yaml'
            if receipt.exists():
                ptc=read_yaml(receipt)
                ptc_cost={k:sum(float(s.get(k,0.)) for s in ptc['segments'])
                          for k in ('preparation_seconds','fit_seconds','apply_seconds','output_seconds')}
                ptc_cost.update(wall_seconds=float(ptc['wall_seconds']),
                    accounting='segment preparation/fit/apply/output sums; nested fit timers excluded; wall includes orchestration')
            row['networks'].append(dict(n,stages=stages,PTC_cost=ptc_cost))
        costs.append(row)
    return dict(coverage=coverage,arrays=array_coverage(coverage),costs=costs,
        scope='one observation; supplied, requested and available populations separate; diagnostic support is not production loss',
        accounting='per-network wall stages may overlap; CPU/RSS belong to whole process, not additive worker attribution',
        campaign_state=campaign.get('state','incomplete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True)
    a=p.parse_args();(a.campaign/'SUMMARY.json').write_text(json.dumps(summarize(a.campaign,a.inventory),indent=2)+'\n')
