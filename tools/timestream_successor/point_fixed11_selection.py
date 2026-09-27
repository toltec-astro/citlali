"""Freeze one observation-specific experimental set from the completed x audit.

Not a production classifier: the owner delegated this one-time enumeration.
The prior descriptive >5 prominence is evidence, not a newly calibrated law.
"""
import argparse,gzip,json,hashlib
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from point_rtc_diagnostics import read,yread,bands
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound
from inspect_forward_map import require
from run_rtc_notch_recovery import finite_trial
REPO=Path(__file__).resolve().parents[2]
PRIOR=REPO/'validation/rtc_cost_response_2026-09-27/compact'
CENTER=11.006255844577186
NOTCH_SHA='879c4b7978b3ce08ca6844594e8730ba717e459d3e5db387f7a15957eb055fa7'
LPF_SHA='e25377075b9b20147bfc11b9c4b9ab792dd60576166f6c25dbc8b9aaf75e8967'
def coeff_sha(h):return hashlib.sha256(np.asarray(h,dtype='<f8').tobytes()).hexdigest()
def disposition(p,f,windows,nfft):
    if not np.isfinite(p).all() or not windows:return 'uncertain','no complete calibrated source-excluded spectrum',{}
    ix=np.flatnonzero((f>=9.75)&(f<=12.25));peak=int(ix[np.argmax(p[ix])]);b=bands(p,f)
    prominent=sorted(int(w['start']) for w in windows if w['value']>5)
    persistent=bool(prominent and prominent[-1]-prominent[0]>=nfft)
    resolved=p[peak]>p[peak-1] and p[peak]>p[peak+1]
    target=CENTER-.5<=f[peak]<=CENTER+.5
    ev=dict(peak_hz=float(f[peak]),aggregate_line_power=float(b['line_power']),aggregate_line_excess=float(b['line_excess']),aggregate_prominence=float(b['line_prominence']),resolved_local_maximum=bool(resolved),inside_fixed_design_cutoffs=bool(target),complete_windows=len(windows),prominent_gt5_windows=len(prominent),prominent_native_starts=prominent,nonoverlapping_prominent_windows=bool(persistent))
    if resolved and target and b['line_prominence']>5 and persistent:return 'selected','resolved target feature and prior-audit prominent evidence in multiple nonoverlapping source-excluded windows',ev
    if b['line_prominence']>5 or prominent:return 'uncertain','equivocal target association, weak aggregate, or insufficient persistent support; retain control',ev
    return 'unselected','no clearly identified target feature in available diagnostics; not a claim of line-free data',ev

def prepare(dest):
    require(not (dest/'APPLICABILITY.json').exists(),'selection already frozen')
    dest.mkdir(parents=True,exist_ok=True);z=np.load(PRIOR/'report/A/population-spectra.npz');f=z['native_frequency'];nfft=2*(len(f)-1)
    windowpath=PRIOR/'report/A/window-bands.json.gz';allrows=json.loads(gzip.decompress(windowpath.read_bytes()));windows=defaultdict(list)
    for row in allrows:
        if row['domain']=='native-matched' and row['stage']=='raw' and row['metric']=='line_prominence':windows[row['network'],row['detector']].append(row)
    lp=read(REPO/'data/development/bright_point_base_v1/a2000-rtc.json')['coefficients'];require(coeff_sha(lp)==LPF_SHA,'historical LPF differs')
    nets=[];inputs=[]
    for nw in (11,12):
        path=PRIOR/f'A-REF-rep1/upstream/products/network{nw}/receipt.yaml';path=path if path.exists() else Path(str(path)+'.gz')
        import yaml
        receipt=yaml.load(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes(),Loader=yaml.CSafeLoader)
        require(len(receipt['native_runs'])==1,'one-time population cannot silently subdivide new physical runs')
        cfg=read(PRIOR/f'A-REF-rep1/inputs/network{nw}-input.json');dt=receipt['cadence_interval_seconds']
        trial=finite_trial(dt,CENTER,3,lp,full_width_hz=1)
        require(coeff_sha(trial['finite_notch'])==NOTCH_SHA,'accepted historical coefficient vector not exactly reproduced at measured cadence')
        net=dict(network=nw,measured_interval_seconds=dt,native_runs=receipt['native_runs'],input_bindings={k:cfg[k] for k in ('raw','tune','manifest','telescope','filter_plan')},detectors=[])
        for d in receipt['detectors']:
            i=d['detector'];status,reason,evidence=disposition(z[f'n{nw}_raw'][:,i],f,windows[nw,i],nfft)
            net['detectors'].append(dict(detector=i,channel=d['channel'],occurrence=d['occurrence'],array_association=d['array_association'],physical_run=0,disposition=status,reason=reason,evidence=evidence))
        net['counts']=dict(Counter(d['disposition'] for d in net['detectors']));net['counts']['total']=len(net['detectors']);nets.append(net);inputs.append(bound(path))
    design=dict(schema='point-fixed11-design-v1',historical_source='898ed9a4e1c142f0334cc6f5dfe9974536ee227a',historical_center_hz=CENTER,printed_owner_center_hz=11.0062558446,center_scope='same historical full precision; printed directive rounds it, no recentering',measured_interval_seconds=dt,nominal_interval_seconds=.008192,coefficients_sha256=NOTCH_SHA,lowpass_coefficients_sha256=LPF_SHA,recipe_binding=bound(REPO/'tools/timestream_successor/run_rtc_notch_recovery.py'),factor=2,phase=0,paired_coordinates=True,**trial)
    save_json(dest/'DESIGN.json',design)
    save_json(dest/'APPLICABILITY.json',dict(schema='point-fixed11-applicability-v1',authority='owner-fixed11-comparison-2026-09-27',observation=[152430,0,2],tune=[152430,0,1],array='a2000',frozen_before_candidate=True,policy_scope='one-time owner-delegated experimental enumeration; no universal detection threshold/classifier',evidence_recipe='unchanged4s Hann source-excluded raw-x audit; aggregate resolved target-band maximum, existing descriptive prominence>5, corroborated by >=2 nonoverlapping prominent windows; ambiguous/unsupported pairs remain control',time_evidence_limit='saved window prominence is target-band maximum, not a per-window frequency fit; aggregate resolved feature supplies target association; no candidate evidence used',physical_domain='one original physical native run per network; no new temporal switching',design=bound(dest/'DESIGN.json'),evidence=[bound(PRIOR/'report/A/population-spectra.npz'),bound(windowpath),bound(PRIOR/'report/A/diagnostic-recipe.json'),*inputs],networks=nets))
    print(json.dumps({n['network']:n['counts'] for n in nets}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
