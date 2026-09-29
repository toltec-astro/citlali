"""Saved historical checks only; no raw inputs, reductions, retuning or membership claim."""
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_additive_diagnostics import save_json
from inspect_forward_map import require,digest
D=REPO/'validation/rtc_two_stage_2026-09-28'
def check():
    evidence=json.loads((D/'selection/candidate-evidence.json').read_text())['annotations'];cohort={r['key']:r for r in json.loads((D/'selection/candidate-cohort.json').read_text())};z=np.load(D/'compact/report/new-windows.npz');records=[]
    for q in evidence:
        key=q['key'];t=z[key+'_time'];contrast={};adj={}
        for c in ('x','r'):
            a=z[key+'_raw_'+c];before=(t>=q['before'][0])&(t<q['before'][1]);during=(t>=q['during'][0])&(t<q['during'][1]);delta=float(np.median(a[during])-np.median(a[before]));require(np.isclose(delta,q['contrasts'][c]['delta'],rtol=1e-12,atol=1e-20),'saved original contrast changed');contrast[c]=delta
            peer=z[key+'_peer_'+c];adj[c]=[float(np.corrcoef(a,peer[:,i])[0,1]) for i in range(2)]
        ratio=cohort[key]['paired_coherent_count']/cohort[key]['cohort_n'];nominal=np.hypot(z[key+'_longitude_arcsec'],z[key+'_latitude_arcsec'])
        # Only grid windows completely contained in the saved excerpt can be tested.
        onsets=np.arange(np.ceil((t[0]+1)/.125),np.floor((t[-1]+.008192-2)/.125)+1)*.125
        records.append(dict(key=key,contrast=contrast,historical_coherent_fraction=ratio,adjacent_correlations=adj,nominal_source_min_arcsec=float(nominal.min()),old_full_window_passes_new_source_guard=bool((nominal>60).all()),complete_new_grid_windows_in_excerpt=len(onsets),full_new_selector_replay='unavailable: complete historical peer waveforms and grid-aligned window margins not retained'))
    by={r['key']:r for r in records};d88=next(r for r in evidence if r['key']=='n2_new1')
    require(d88['contrasts']['x']['target_to_cohort_MAD']>10 and d88['contrasts']['r']['target_to_cohort_MAD']>10 and by['n2_new1']['historical_coherent_fraction']<.01,'historical D88 descriptive acceptance changed')
    require(all(abs(c)<.3 for cs in by['n2_new1']['adjacent_correlations'].values() for c in cs),'historical D88 adjacency changed')
    require(by['n1_new0']['historical_coherent_fraction']>=.01,'shared-waveform control lost')
    require(by['n2_t5']['nominal_source_min_arcsec']<60,'source-contact control lost')
    return dict(pass_=True,checks=records,interpretation='Checks recover historical arithmetic and rejected controls. They do not certify that the portable rule recovers D88. The fixed nominal-center guard and fixed grid intentionally differ from manual fitted-center selection; no threshold or window was tuned after these checks.',bindings=[dict(path=str(p),sha256=digest(p)) for p in [D/'selection/candidate-evidence.json',D/'selection/candidate-cohort.json',D/'compact/report/new-windows.npz']])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();save_json(a.output,check())
