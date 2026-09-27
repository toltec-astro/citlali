"""Saved native receipts and basis diagnostics, outside primary timing."""
import json,time
from pathlib import Path
import numpy as np
from inspect_forward_map import read_yaml,digest
from point_additive_diagnostics import save_json
from report_point_speed import stages

def read(p):return json.loads(p.read_text())
def native(root):
    started=time.perf_counter();dest=root/'report';rows=[];stage={};acquisition=[]
    for nw in (11,12):
        b=root/f'A-rank10-N10-rep1/upstream/products/network{nw}';r=read_yaml(b/'receipt.yaml');t=np.fromfile(b/'native-time.f64',dtype='<f8');assert digest(b/'native-time.f64')==r['native_time_sha256']
        acquisition.append(dict(network=nw,rows=len(t),first=float(t[0]),last=float(t[-1]),elapsed_seconds=float(t[-1]-t[0]),sample_integration_seconds=r['native_integration_seconds'],duration_with_final_sample_seconds=float(t[-1]-t[0]+r['native_integration_seconds']),physical_runs=r['native_runs'],native_time_sha256=r['native_time_sha256'],scope='network acquisition first-to-last native sample; not single-sample integration time'))
    for out in sorted(root.glob('*-N10-rep1')):
        execution=read(out/'EXECUTION.json') if (out/'EXECUTION.json').exists() else {};rank=execution.get('rank');method=execution.get('method');label=('current' if method=='observed-als' else 'warm' if method=='warm-observed-als' else 'covariance')+str(rank)
        if execution.get('status')=='completed':
            stage[label]=stages(out)
            events=[json.loads(line) for line in (out/'timing.jsonl').read_text().splitlines()]
            spans={n:sum(e['seconds'] for e in events if e['event']=='span' and e['name']==n) for n in {e['name'] for e in events if e['event']=='span'}}
            paths=[out/'upstream/RUN.json',out/'M0/RUN.json']+list((out/'run/BASE-POINT/REF/a2000').glob('cycle-*/native/RUN.json'))
            native=sum(read(p)['wall_seconds'] for p in paths)
            stage[label]['python_inclusive_spans']=spans
            stage[label]['launched_native_wall_seconds']=native
            stage[label]['outside_native_invocation_seconds']=spans['native_invocation_including_reservation_hashing_publication']-native
            stage[label]['overhead_scope']='outside sampled native subprocess invocations; includes campaign storage scans, reservation, pre/post bookkeeping; no directly isolated storage-scan timer, so causal attribution is code-supported inference'
        for branch in ('REF','C-core-plus','C-core-minus','C-shoulder'):
            paths=list((out/f'run/BASE-POINT/{branch}/a2000').glob('cycle-*/native/products/receipt.yaml'))
            if branch=='REF':paths+=list(out.glob('upstream/products/network*/donor-continuity/ptc/receipt.yaml'))
            else:paths+=list((out/f'initial/{branch}/native/products').glob('receipt.yaml'))
            for p in sorted(paths):
                r=read_yaml(p);k=r.get('cycle_index',0)
                for s in r.get('segments',[]):
                    orig=s.get('source_segment',s);stem=next(x[:-10] for x in orig['files'] if x.endswith('-input.f64'));nw=orig.get('network',r.get('network'));name=f'network-{nw}-{stem}';shape=(orig['detectors'],rank)
                    bpath=p.parent/(stem+'-basis.f64') if 'source_segment' not in s else p.parent/name/'basis.f64'
                    control=root/f'A-rank{rank}-N10-rep1';cp=control/f'upstream/products/network{nw}/donor-continuity/ptc/{stem}-basis.f64' if k==0 else control/f'run/BASE-POINT/REF/a2000/cycle-{k:02d}/native/products/{name}/basis.f64'
                    projector=None
                    if bpath.is_file() and cp.is_file():
                        b=np.fromfile(bpath,dtype='<f8').reshape(shape);a=np.fromfile(cp,dtype='<f8').reshape(shape);projector=float(np.linalg.norm(b@b.T-a@a.T))
                    row=dict(family=label,method=method,rank=rank,branch=branch,cycle=k,network=nw,segment=orig.get('scan'),native_interval=orig.get('native_interval'),detector_indices=orig.get('detector_indices'),output_detector_indices=orig.get('output_detector_indices'),fit_columns_in_output=orig.get('fit_columns_in_output'),scheduled_times=orig['scheduled_times'],detectors=orig['detectors'],output_detectors=orig.get('output_detectors'),eligible=orig['eligible'],retained=s['retained'],input_eligibility_changed=s.get('input_eligibility_changed',0),application_failed_times=s.get('application_failed_times'),realized_rank=s.get('realized_rank',s.get('basis_columns')),converged=s.get('converged'),stopping_reason=s.get('stopping_reason'),J=s.get('iterations'),objective=s.get('objective'),source_eligible_hash=orig['files'].get(stem+'-eligible.u8'),projector_Frobenius_vs_current_REF=projector,projector_scope='same rank/network/segment; later current residuals may differ, not a fixed-input equivalence test',native_receipt_sha256=digest(p),timings={n:v for n,v in s.items() if n.endswith('_seconds')})
                    for n in ('signed_spectrum','overlap_counts','overlap_frequencies','eigen_relative_residual','eigen_residual_tolerance','negative_to_positive_power','most_negative_to_largest_positive','warm_initializer_used','warm_scope','warm_predecessor'):row[n]=s.get(n)
                    rows.append(row)
    save_json(dest/'acquisition.json',acquisition);save_json(dest/'native-methods.json',rows);save_json(dest/'method-stages.json',stage);save_json(dest/'native-report-cost.json',dict(seconds=time.perf_counter()-started,scope='saved receipt/basis diagnostics, no native processing'))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);native(p.parse_args().root)
