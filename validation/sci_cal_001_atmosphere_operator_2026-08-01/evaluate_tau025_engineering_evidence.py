#!/usr/bin/env python3
"""Read-only D007 continuous LOS-tau evaluation from the frozen _002 cache."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

import run_am12_successor_adoption_study as successor

PACKAGE = Path(__file__).resolve().parent
ROOT = Path('/Users/gwilson/work_toltec/local_data/sci_cal_001_tau025_engineering_extension_002_root')
TOLTECA = Path('/Users/gwilson/GitHub/tolteca')
TAU = {'tau015': .15, 'tau020': .20, 'tau025': .25,
       'tau01625': .1625, 'tau0175': .175, 'tau01875': .1875,
       'tau02125': .2125, 'tau0225': .225, 'tau02375': .2375}
CONSTRUCTION = ('tau015', 'tau020', 'tau025')
ALPHAS = (-1, 0, 2, 4)

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def bands():
    result=[]
    for array in ('a1100','a1400','a2000'):
        rel=f'tolteca/data/cal/toltec_passband/data/{array}_passband.ecsv'
        data=subprocess.check_output(['git','-C',str(TOLTECA),'show',f'{successor.TOLTECA_COMMIT}:{rel}'])
        f,r=successor.parse_primary_ecsv(data,array)
        result.append(successor.Bandpass(f'toltec_v1_{array}',array,'primary',f,r,rel,digest_bytes(data),successor.TOLTECA_COMMIT,'ECSV v1'))
    return result

def digest_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()

def main() -> int:
    context=json.loads((ROOT/'execution_context.json').read_text())
    manifest=json.loads((ROOT/'manifests/execution_manifest.json').read_text())
    assert digest(ROOT/'execution_context.json') == manifest['execution_context_sha256']
    assert context['full_run_count']==1275 and context['scale_trace_count']==225
    assert manifest['full_grid_count']==1275 and manifest['scale_trace_count']==225
    assert manifest['raw_sidecar_pair_count']==23024
    bps=bands(); records={}
    for side in (ROOT/'sidecars').glob('*.json'):
        s=json.loads(side.read_text())
        if not s['run_id'].startswith('tau025e001/') : continue
        q=s['request']; node=q['target']; profile=q['profile']; elev=90-int(q['zenith_angle_deg'])
        raw=ROOT/'raw_outputs'/side.with_suffix('.txt').name
        assert digest(raw)==s['raw_sha256'] and s['return_code'] in (0,1)
        parsed=successor.parse_am_output(raw.read_bytes(),s['run_id'])
        for band in bps:
            tx=np.interp(band.frequency_ghz,parsed.frequency_ghz,parsed.transmission)
            for alpha in ALPHAS:
                records[profile,node,elev,band.array,alpha]=-math.log(float(np.dot(band.weights(alpha),tx)))
    assert len(records)==1275*3*4
    profiles=sorted({p for p,*_ in records})
    rows=[]; structural=[]
    for candidate in ('piecewise_linear_los_tau_v1','pchip_los_tau_v1'):
        errors=[]; exact=[]; opacity_bad=0; elevation_bad=0; continuity=0.
        for profile in profiles:
          for band in bps:
           for alpha in ALPHAS:
            es=np.array([25,35,45,55,65,75,80.],float); ts=np.array([.15,.2,.25])
            nodes=np.array([[records[profile,n,int(e),band.array,alpha] for e in es] for n in CONSTRUCTION])
            def ev(t,e):
                tq=np.asarray(t,float); eq=np.asarray(e,float)
                lev=np.asarray(PchipInterpolator(es,nodes,axis=1)(eq),float)
                if candidate.startswith('piecewise'):
                    return np.column_stack([np.interp(tq,ts,lev[:,j]) for j in range(eq.size)])
                return np.asarray(PchipInterpolator(ts,lev,axis=0)(tq),float)
            for n in CONSTRUCTION:
             for e in es:
                exact.append(abs(float(ev([TAU[n]],[e])[0,0])-records[profile,n,int(e),band.array,alpha]))
            for n,v in TAU.items():
             if n in CONSTRUCTION: continue
             for e in (29,41,53,67,79):
                predicted=float(ev([v],[e])[0,0]); truth=records[profile,n,e,band.array,alpha]
                err=math.expm1(predicted-truth); errors.append(err)
                rows.append({'candidate':candidate,'profile':profile,'node':n,'elevation_deg':e,'array':band.array,'alpha':alpha,'truth_los_tau':f'{truth:.17e}','candidate_los_tau':f'{predicted:.17e}','fractional_correction_error':f'{err:.17e}'})
            tg=np.linspace(.15,.25,101); eg=np.linspace(25,80,111); grid=ev(tg,eg)
            opacity_bad += int(np.count_nonzero(np.diff(grid,axis=0)<-1e-12)); elevation_bad += int(np.count_nonzero(np.diff(grid,axis=1)>1e-12))
            if not (np.all(np.isfinite(grid)) and np.all(grid>=0)): raise RuntimeError('physical-domain failure')
            left=np.nextafter(.15,-np.inf); right=np.nextafter(.15,np.inf)
            continuity=max(continuity,float(np.max(np.abs(np.expm1(ev([right],eg)[0]-ev([left],eg)[0])))))
        a=np.asarray(errors); structural.append({'candidate':candidate,'max_fractional_correction_error':float(np.max(abs(a))),'p95':float(np.quantile(abs(a),.95)),'rms':float(np.sqrt(np.mean(a*a))),'exact_node_max_los_tau_error':float(max(exact)),'opacity_violations':opacity_bad,'elevation_violations':elevation_bad,'nextafter_max_relative_correction_step':continuity})
    best=min(structural,key=lambda x:x['max_fractional_correction_error'])
    with (PACKAGE/'tau025_engineering_heldout_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result={'schema_version':'sci-cal-001-tau025-evaluation-v1','cache_root':str(ROOT),'execution_context_sha256':digest(ROOT/'execution_context.json'),'execution_manifest_sha256':digest(ROOT/'manifests/execution_manifest.json'),'profiles':profiles,'heldout_row_count':len(rows),'candidate_metrics':structural,'recommended_candidate':best['candidate'],'support':{'tau225':'0.15 <= tau225 <= 0.25','elevation_deg':'25 <= EL <= 80','profile_identity':'one of 25 frozen AMC profile identities; no generic profile inference'},'interpretation':'numerical representation evidence only; not observational calibration accuracy or operator adoption'}
    (PACKAGE/'tau025_engineering_evaluation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (PACKAGE/'SCI_CAL_001_TAU025_CONTINUOUS_OPERATOR_DECISION_BRIEF.md').write_text(f'''# SCI-CAL-001 TAU025 continuous-operator decision brief\n\nFrozen D007 evidence was evaluated read-only from cache manifest `{result['execution_manifest_sha256']}`. The held-out table has {len(rows)} rows across 25 profile identities, three TolTECA v1 passbands, and alpha={{-1,0,2,4}}.\n\nRecommendation for owner decision: select `{best['candidate']}` only as a **profile-identified evaluation operator** over `0.15 <= tau225 <= 0.25`, `25 <= EL <= 80`. It is fail-closed outside that domain or without one of the frozen AM profile identities. Its maximum held-out fractional correction error is `{best['max_fractional_correction_error']:.6%}` (p95 `{best['p95']:.6%}`, RMS `{best['rms']:.6%}`); exact-node LOS-tau error is `{best['exact_node_max_los_tau_error']:.3e}`.\n\nThe requested owner choice is whether this profile-conditioned numerical representation, including its {best['opacity_violations']} opacity and {best['elevation_violations']} sampled monotonicity violations, is acceptable for an engineering-only versioned operator. No generic profile selector was tested or inferred. This result is representation fidelity only; it does not establish observational 5--10% absolute flux accuracy or approximately 5% repeatability, and does not authorize implementation or production use.\n''')
    return 0
if __name__=='__main__': raise SystemExit(main())
