#!/usr/bin/env python3
"""Compact population/cost accounting from the fixed campaign's preserved products."""
import argparse
import json
from pathlib import Path
import yaml


def read_yaml(path):
    with path.open() as f:return yaml.load(f,Loader=yaml.CSafeLoader)


def summarize(root,inventory):
    inv=json.loads(inventory.read_text());campaign=json.loads((root/'CAMPAIGN.json').read_text())
    coverage=[];costs=[]
    for network in inv['networks']:
        nw=network['network'];row={k:v for k,v in network.items() if k in ('network','array','raw_available','requested','requested_detectors','native_rows','output_sampling_hz','reason')}
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
        coverage.append(row)
    for run in sorted(root.glob('workers-*')):
        observation=read_yaml(run/'products/observation-receipt.yaml');metrics=json.loads((run/'RUN.json').read_text())
        row=dict(workers=metrics['workers'],wall_seconds=metrics['wall_seconds'],peak_rss_bytes=metrics['polled_peak_rss_bytes'],
            output_bytes=metrics['output_bytes'],CPU_sampled_seconds=max((s['user_seconds']+s['system_seconds'] for s in metrics['samples']),default=0),
            OS_threads_max=max((s['os_threads'] for s in metrics['samples']),default=0),
            observed_Eigen_threads=observation['observed_Eigen_threads'],shared_preparation_seconds=observation['shared_preparation_seconds'],networks=[])
        for n in observation['networks']:
            trace=run/'products'/f"network{n['network']}"/'performance.jsonl';stages=[];last=0.
            if trace.exists():
                for line in trace.read_text().splitlines():
                    x=json.loads(line)
                    if 'stage' in x:
                        stages.append(dict(stage=x['stage'],interval_wall_seconds=x['seconds']-last));last=x['seconds']
            row['networks'].append(dict(n,stages=stages))
        costs.append(row)
    return dict(coverage=coverage,costs=costs,
        scope='one observation; eleven supplied networks; missing 6 and 10 explicit; diagnostic support is not production loss',
        accounting='per-network wall stages may overlap; CPU/RSS belong to whole process, not additive worker attribution',
        campaign_state=campaign.get('state','incomplete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True)
    a=p.parse_args();(a.campaign/'SUMMARY.json').write_text(json.dumps(summarize(a.campaign,a.inventory),indent=2)+'\n')
