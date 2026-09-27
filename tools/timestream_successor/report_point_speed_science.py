"""Saved-data science comparison; uses existing ideal-addition measurement owner only."""
import argparse,json,time
from pathlib import Path
import numpy as np
from compare_point_pipeline import available
from point_fruit_diagnostics import load,patch_map
from point_empirical import source_patch
from point_reference import context
from point_guarded_feedback import infer_base,ideal_comparison
from point_response_cd import vector_measurement,MEASUREMENT_NAMES
from report_point_rank import equivalent,SIGMA
from point_additive_diagnostics import save_json

CHECK=(0,1,3,5,10)
def read(p):return json.loads(Path(p).read_text())
def map_path(out,k):return available(out/'M0/products/a2000/map.fits' if k==0 else out/f'run/BASE-POINT/REF/a2000/cycle-{k:02d}/native/products/total.fits')
def sigmas(rows):return {x['name']:x['conditional_sigma'] if x.get('identifiable',True) and x.get('source_transfer_supported',True) else None for x in rows}
def science(root,retained,report_dir=None):
    start=time.perf_counter();dest=report_dir or root/'report';dest.mkdir(parents=True,exist_ok=True)
    families=[('current10',root/'A-rank10-N10-rep1',10,'observed-als'),('current15',root/'A-rank15-N10-rep1',15,'observed-als')]
    for phase in ('B','C'):
        for tag,method in [('warm','warm-observed-als'),('covariance','masked-covariance-count-v1')]:
            rank=10 if phase=='B' else 15;out=root/f'{phase}-{tag}-rank{rank}-N10-rep1'
            if (out/'run/BASE-POINT/REF/a2000/SUMMARY.json').exists():families.append((tag+str(rank),out,rank,method))
    records=[];effects=[];plateaus=[];timing=[]
    for name,out,rank,method in families:
        summary=read(out/'run/BASE-POINT/REF/a2000/SUMMARY.json');hs=summary['histories'];end=sigmas(summary.get('endpoint_uncertainty',[]));sigma_by={10:end};current=method=='observed-als'
        if current:
            for k in CHECK[:-1]:sigma_by[k]=sigmas(read(root/f'A-rank{rank}-N{k}-rep1/run/BASE-POINT/REF/a2000/SUMMARY.json')['endpoint_uncertainty'])
        elif (out/'CHECKPOINTS.json').exists():
            for r in read(out/'CHECKPOINTS.json'):sigma_by[r['cycle']]=sigmas(r['uncertainties'])
        longs=[read(p) for p in sorted(root.glob(out.name[:-1]+'*/LAUNCH.json')) if read(p).get('status')=='completed']
        bycycle={}
        for k in range(11):
            v=[next((x['published_seconds'] for x in r['readiness'] if x['cycle']==k and x['status']=='accepted'),None) for r in longs];v=[x for x in v if x is not None]
            if v:bycycle[k]=dict(median=float(np.median(v)),minimum=min(v),maximum=max(v),repeats=len(v))
        timing.append(dict(family=name,method=method,rank=rank,bycycle=bycycle,exit_seconds=[r['exit_seconds'] for r in longs],analysis_seconds=[r['analysis_complete_seconds'] for r in longs],cpu_seconds=[sum(r['cpu_seconds'].values()) for r in longs],peak_tree_RSS_bytes=[r['aggregate_tree_sampled_RSS'] for r in longs]))
        if not summary.get('ready') or summary['completed_cycle']!=10:continue
        final=hs[-1]['measurement'];fparams=np.asarray(final['guard']['accepted_parameters']);fv=vector_measurement(final);series=[]
        for h in hs:
            if 'measurement' not in h:continue
            k=h['cycle_index'];m=h['measurement'];params=np.asarray(m['guard']['accepted_parameters']);v=vector_measurement(m);delta=v-fv
            row=dict(family=name,method=method,rank=rank,cycle=k,ready=h['ready'],fit_status=m['fit']['status'],values=dict(zip(MEASUREMENT_NAMES,v.tolist())),fit_parameters=params.tolist(),measurement=m,conditional_sigmas=sigma_by.get(k,{}),sigma_scope='own matching REF checkpoint; conditional statistical scale, not paired significance',T10_delta=dict(zip(MEASUREMENT_NAMES,delta.tolist())),T10_fraction={n:float(d/f) if abs(f)>1e-8 else None for n,d,f in zip(MEASUREMENT_NAMES,delta,fv)},T10_sigma_delta={n:float(d/end[SIGMA[n]]) if SIGMA.get(n) in end and end[SIGMA[n]] else None for n,d in zip(MEASUREMENT_NAMES,delta)})
            # Width sigma belongs to the fitted axis parameter, not sorted FWHM.
            row['parameter_T10_sigma_delta']={n:float(d/end[n]) if end.get(n) else None for n,d in zip(('peak','cx','cy','sx','sy','angle','constant'),params-fparams)}
            order=np.argsort(fparams[3:5])[::-1];own_order=np.argsort(params[3:5])[::-1]
            row['width_T10_sigma_delta']={n:float((params[3+own_order[j]]-fparams[3+order[j]])/end[('sx','sy')[order[j]]]) if end.get(('sx','sy')[order[j]]) else None for j,n in enumerate(('major','minor'))}
            row['fit_constrained']=m['fit'].get('constrained',False)
            records.append(row);series.append(row)
        for group,keys,source in [('pointing',('cx','cy'),'parameter_T10_sigma_delta'),('compact_peak',('peak',),'parameter_T10_sigma_delta'),('compact_template',('fixed_template',),'T10_sigma_delta'),('widths',('major','minor'),'width_T10_sigma_delta'),('broader_aperture',('aperture_2',),'T10_sigma_delta')]:
            earliest=next((r['cycle'] for i,r in enumerate(series) if all(s[source].get(key) is not None and abs(s[source][key])<=.25 for s in series[i:] for key in keys)),None)
            plateaus.append(dict(family=name,quantity=group,earliest_observed_N=earliest,definition='all later recorded values within0.25 own T10 conditional sigma; descriptive endpoint plateau only, not a stopping rule or truth test'))
        controls=retained/f'rank-{rank}' if current else out;c=context(out/'freeze/a2000')
        for branch in ('C-core-plus','C-core-minus','C-shoulder'):
            path=controls/f'run/BASE-POINT/{branch}/a2000/SUMMARY.json'
            if not path.exists():continue
            ih={r['cycle_index']:r for r in read(path)['histories']}
            for k in CHECK:
                if k not in ih or 'measurement' not in ih[k]:continue
                h=ih[k];ideal=h.get('ideal')
                if not ideal:
                    rp=controls/f'initial/{branch}/response.fits' if k==0 else controls/f'run/BASE-POINT/{branch}/a2000/cycle-{k:02d}/response.fits'
                    response,rh=load(available(rp));full,header=load(map_path(out,k));ks,vs,_,_,_=source_patch(response['CONTROL'],response['VALID'].astype(bool),full['NORMALIZATION'],rh)
                    rs,rsv,_,_=patch_map(map_path(out,k));im,_=infer_base(c,rs+ks,rsv&vs);ideal=ideal_comparison(c,hs[k]['measurement'],h['measurement'],im)
                v=vector_measurement(hs[k]['measurement']);sigma=sigma_by.get(k,{})
                for q,full in zip(ideal['comparisons'] or [],v):
                    n=q['name'];s=sigma.get(SIGMA.get(n));eta=h['response']['source_disk']['eta'];eq,why=equivalent(s,q['ratio'],eta,branch)
                    effects.append(dict(family=name,rank=rank,method=method,branch=branch,cycle=k,measurement=n,own_REF=float(full),**q,conditional_sigma=s,sigma_equiv=eq,sigma_equiv_scope=why,discrepancy_over_conditional_sigma=q['absolute_discrepancy']/s if s else None,fraction_of_full_source=q['absolute_discrepancy']/full if abs(full)>1e-8 else None,spatial_response=h['response']['source_disk'],raw_aperture_background=ideal['raw_aperture_and_background_increments'],ideal_measurement=ideal['ideal_map_addition'],ideal_scope='existing measurement owner applied to saved own REF plus retained ideal control; no native processing'))
    for file,value in [('science-measurements',records),('science-responses',effects),('science-plateaus',plateaus),('method-timing',timing)]:save_json(dest/(file+'.json'),value)
    save_json(dest/'science-report-cost.json',dict(seconds=time.perf_counter()-start,scope='saved map-only ideal control measurements and reporting, outside raw timing; no extra uncertainty endpoints'))
    print(json.dumps(dict(families=[x['family'] for x in timing],measurements=len(records),effects=len(effects),plateaus=plateaus,timing=timing),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--retained',type=Path,required=True);p.add_argument('--report-dir',type=Path);a=p.parse_args();science(a.root,a.retained,a.report_dir)
