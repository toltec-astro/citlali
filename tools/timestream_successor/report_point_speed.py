"""Saved-only timing summaries. No native processing or solver invocation."""
import json,time
from pathlib import Path
import numpy as np
from inspect_forward_map import read_yaml
from point_additive_diagnostics import save_json

def read(p):return json.loads(Path(p).read_text())

def stages(out):
    result=dict(networks=[],feedback=[])
    for nw in (11,12):
        base=out/f'upstream/products/network{nw}'
        rows=[json.loads(s) for s in (base/'performance.jsonl').read_text().splitlines()];marks=[r for r in rows if 'stage' in r]
        exclusive=[dict(stage=r['stage'],seconds=r['seconds']-before['seconds']) for before,r in zip(marks,marks[1:])]
        ptc=read_yaml(base/'donor-continuity/ptc/receipt.yaml')
        result['networks'].append(dict(network=nw,stage_intervals=exclusive,scope='per-network wall intervals; two networks overlap, do not sum as critical-path wall',ptc=[{k:v for k,v in s.items() if 'seconds' in k or k in ('scan','native_interval','scheduled_times','detectors','eligible','retained','iterations','converged','application_failed_times')} for s in ptc['segments']]))
    for p in sorted((out/'run/BASE-POINT/REF/a2000').glob('cycle-*/native/products/receipt.yaml')):
        r=read_yaml(p);timings={k:v for k,v in r.items() if k.endswith('_seconds')}
        segment={k:sum(s.get(k,0) for s in r['segments']) for k in ('wall_seconds','loading_seconds','projection_seconds','input_serialization_and_injection_seconds','subtraction_seconds','preparation_seconds','learning_seconds','application_seconds','apply_restore_seconds','initialization_seconds','covariance_seconds','decomposition_seconds','coefficient_seconds','basis_seconds','check_seconds')}
        # Sequential feedback segments only. Native residual is intentionally unattributed.
        residual=r['wall_seconds']-segment['wall_seconds']-r['MAP_seconds']-r['serialization_verification_seconds']
        result['feedback'].append(dict(cycle=r['cycle_index'],native_inclusive=timings,sequential_segment_totals=segment,native_adapter_unattributed_seconds=residual,solver_J=[dict(network=s['source_segment']['network'],segment=s['source_segment'].get('scan'),J=s['iterations'],converged=s['converged'],eligible=s['source_segment']['eligible'],retained=s['retained']) for s in r['segments']]))
    return result

def phase_a(root):
    started=time.perf_counter();dest=root/'report';dest.mkdir(exist_ok=True)
    campaign=read(root/'PHASE-A.json');assert campaign['status']=='completed'
    table=[];timings=[];details={}
    for rank in (10,15):
        longs=[read(root/f'A-rank{rank}-N10-rep{i}/LAUNCH.json') for i in (1,2,3)]
        for N in range(11):
            values=[next(v['published_seconds'] for v in r['readiness'] if v['cycle']==N) for r in longs]
            row=dict(rank=rank,N=N,prefix_ready_median=float(np.median(values)),prefix_ready_min=min(values),prefix_ready_max=max(values),prefix_ready_repeats=3)
            if N in (0,1,3,5):
                short=read(root/f'A-rank{rank}-N{N}-rep1/LAUNCH.json')
                row.update(terminating_ready=short['readiness'][-1]['published_seconds'],terminating_analysis=short['analysis_complete_seconds'],terminating_exit=short['exit_seconds'],terminating_repeats=1)
            if N==10:
                for key in ('analysis_complete_seconds','exit_seconds'):
                    row[key+'_median']=float(np.median([r[key] for r in longs]));row[key+'_range']=[min(r[key] for r in longs),max(r[key] for r in longs)]
            table.append(row)
        details[str(rank)]=stages(root/f'A-rank{rank}-N10-rep1')
    for row in campaign['runs']:
        out=root/row['name'];launch=read(out/'LAUNCH.json');events=[json.loads(s) for s in (out/'timing.jsonl').read_text().splitlines()]
        spans={n:sum(e['seconds'] for e in events if e['event']=='span' and e['name']==n) for n in {e['name'] for e in events if e['event']=='span'}}
        timings.append(dict(**row,launch={k:v for k,v in launch.items() if k!='samples'},python_inclusive_spans=spans,regression=read(out/'REGRESSION.json')['pass_']))
    save_json(dest/'PHASE-A-timing.json',table);save_json(dest/'PHASE-A-stages.json',details);save_json(dest/'PHASE-A-runs.json',timings)
    text=['# Phase A: measured unchanged POINT latency','', 'Scope: POINT 152430/0/2, a2000 networks 11/12, Apple M4 Pro, 14 cores, 48 GiB RAM. Two upstream network workers, serial feedback segments, one thread per numerical library. Fresh process and scientific state each time; filesystem cache not flushed, preflight hashing may warm inputs. All 14 trajectories / 78 feedback calls passed strict unchanged-science or exact-prefix gates. No science replacement.','', '| Rank | N | Prefix readiness median [range], s (3 runs) | Actual terminating analysis / exit, s |','|---|---:|---:|---:|']
    for row in table:
        if row['N'] not in (0,1,3,5,10):continue
        terminal=f"{row['terminating_analysis']:.2f} / {row['terminating_exit']:.2f} (single)" if row['N']<10 else f"{row['analysis_complete_seconds_median']:.2f} / {row['exit_seconds_median']:.2f} (medians)"
        text.append(f"| {row['rank']} | {row['N']} | {row['prefix_ready_median']:.2f} [{row['prefix_ready_min']:.2f}, {row['prefix_ready_max']:.2f}] | {terminal} |")
    text+=['','READY files contain actual map, measurements and current accepted/failed status and are atomically published. Parent polling independently recorded caller detection. Internal timestamps and process exit are separately retained. All shorter commands execute exactly N applications and one terminal endpoint; the saved m_N is unapplied.','', '## Leading bottleneck findings before alternative implementation','']
    for rank,d in details.items():
        f=d['feedback'];total=sum(x['native_inclusive']['wall_seconds'] for x in f);learn=sum(x['sequential_segment_totals']['learning_seconds'] for x in f);init=sum(x['sequential_segment_totals']['initialization_seconds'] for x in f);eigen=sum(x['sequential_segment_totals']['decomposition_seconds'] for x in f);other=sum(x['native_adapter_unattributed_seconds'] for x in f)
        j=[s['J'] for x in f for s in x['solver_J']]
        text.append(f"Rank {rank}, predetermined first T10 run: feedback native wall totals {total:.2f} s; PTC fits {learn:.2f} s inclusive, of which initialization {init:.2f} s and eigendecomposition {eigen:.2f} s. Native adapter time not attributed to segment work, MAP or final serialization/verification totals {other:.2f} s. J ranges {min(j)}–{max(j)} over network/segments/cycles. Nested fit/initialization/decomposition times must not be added. The large adapter residual prevents identifying PCA as the leading overall opportunity from solver cost alone.")
    text+=['','The upstream per-network trace separately reports ingress, RTC stages, CAL, PTC and publication. Network intervals overlap and are not summed into end-to-end wall time. Process CPU and sampled aggregate process-tree RSS are in the run table. Python/native startup, synchronous validation and cleanup remain on the measured path; no endpoint stage was omitted. Full-instrument operational latency and Unity performance remain unmeasured. No operational deadline or automatic stopping criterion has been invented.','', 'Phase A timing gate is complete. The authorized next step is B1/B2 implementation and matched-rank comparison. The final report will add scientific prefix/response comparisons from the retained controls and directly terminated endpoint scales.']
    (dest/'PHASE-A.md').write_text('\n'.join(text)+'\n')
    save_json(dest/'PHASE-A-report-cost.json',dict(seconds=time.perf_counter()-started,scope='saved-data report after timing; excluded from raw-to-result'))
    return table

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();phase_a(a.output)
