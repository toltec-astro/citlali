from pathlib import Path
import sys,json,numpy as np,hashlib,shutil,re
repo=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(repo/'tools/timestream_successor'))
from point_rtc_diagnostics import yread,read
from point_additive_diagnostics import save_json
from point_storage import tree_bytes
root=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928');dest=root/'report';rows=[]
for run in sorted(root.glob('*-*')):
 if (run/'LAUNCH.json').exists():
  family=run.name;rep=1;la=read(run/'LAUNCH.json');b=run/'upstream/products';o=yread(b/'observation-receipt.yaml');obs={r['stage']:r for s in (b/'performance.jsonl').read_text().splitlines() if 'stage'in (r:=json.loads(s))};span=obs['all_networks_finished']['seconds']-obs['shared_observation_preparation_complete']['seconds'];nets=[];spans=[];starts=[];ends=[]
  for n in o['networks']:
   nw=n['network'];marks=[r for s in (b/f'network{nw}/performance.jsonl').read_text().splitlines() if 'stage'in (r:=json.loads(s))];by={r['stage']:r for r in marks};start=by['spike_screen_start'];end=by['RTC_finalized'];origin=n['wall_seconds']-by['root_receipt_written']['seconds'];offset=span-n['wall_seconds']+origin;require_offset=offset>=0;assert require_offset;spans.append((start['seconds'],end['seconds'],offset));starts.append(start['process_user_seconds']+start['process_system_seconds']);ends.append(end['process_user_seconds']+end['process_system_seconds']);rr=yread(b/f'network{nw}/receipt.yaml')
   nets.append(dict(network=nw,RTC_seconds=end['seconds']-start['seconds'],process_CPU_envelope=end['process_user_seconds']+end['process_system_seconds']-start['process_user_seconds']-start['process_system_seconds'],intervals=[dict(stage=z['stage'],seconds=z['seconds']-a['seconds']) for a,z in zip(marks,marks[1:])],apply_operations=rr['donor_continuity']['Apply_operation_seconds'],plan_identity_not_attempt_count=rr['donor_continuity']['attempt'],actual_apply_starts=sum(m['stage']=='RTC_apply_start' for m in marks),actual_apply_completes=sum(m['stage']=='RTC_apply_complete' for m in marks)))
  lower=max(e-b for b,e,s in spans);upper=min(sum(e-b for b,e,s in spans),max(e+s for b,e,s in spans)-min(b for b,e,s in spans));cpu_bounds=[max(0,min(ends)-max(starts)),max(ends)-min(starts)]
  events=[json.loads(s) for s in (run/'timing.jsonl').read_text().splitlines()];sp={k:sum(r['seconds'] for r in events if r['event']=='span' and r['name']==k) for k in {r['name'] for r in events if r['event']=='span'}}
  rows.append(dict(family=family,rep=rep,ready=la['readiness'][-1]['published_seconds'],caller_ready=la['readiness'][-1]['caller_observed_seconds'],analysis=la['analysis_complete_seconds'],exit=la['exit_seconds'],CPU=sum(la['cpu_seconds'].values()),sampled_tree_RSS=la['aggregate_tree_sampled_RSS'],RTC_wall_occupancy_bounds=[lower,upper],RTC_fraction_of_exit_bounds=[lower/la['exit_seconds'],upper/la['exit_seconds']],RTC_CPU_bounds=cpu_bounds,networks=nets,python_inclusive_spans=sp,launch_source=read(run/'freeze/prospective.json')['source_revision']))
save_json(dest/'timing.json',rows)
save_json(dest/'timing-recipe.json',dict(wall='network clocks relative, never sum worker durations as latency; union lower=max duration; upper uses enclosing wrapper span and each worker duration minus last trace to bound its unknown start offset; no overlap inferred from network list order',CPU='process CPU counters: common RTC intersection gives lower bound, zero if no four-way intersection, union envelope gives upper (may include CAL on early finisher); no exclusive per-thread CPU attribution claimed',readiness='published READY contains map/guard/measurements. Parent50ms poll gives caller-detected readiness; poll timestamp sampled before path check may precede publication by a few ms. Analysis and command exit separate.',memory='sampled aggregate process-tree RSS50ms, not OS pressure or continuous high-water',phase='all four runs diagnostic-heavy with source probes and step chain serialization; normal required spectra/output/endpoint included, no overhead subtraction, no repeats or speedup inference'))
logs=[]
for out in sorted(root.glob('*-*/')):
 if not (out/'LAUNCH.json').exists():continue
 errors=[];warnings=0
 for f in out.rglob('*.log'):
  text=f.read_text(errors='replace');errors.extend(dict(file=str(f.relative_to(out)),line=line) for line in text.splitlines() if re.search(r'\[(?:error|critical)\]',line,re.I));warnings+=text.count('RuntimeWarning:')
 assert not errors,errors
 logs.append(dict(run=out.name,native_error_level_messages=0,python_RuntimeWarnings=warnings,uncertainty='all four saved endpoint operators and scales verified finite and exactly equal without BLAS contraction'))
save_json(dest/'log-audit.json',logs)
att=[json.loads(s) for s in (root/'attempts.jsonl').read_text().splitlines()];counts={k:sum(x['kind']==k for x in att) for k in {x['kind'] for x in att}};assert counts==dict(parent=4,trajectory=4,feedback=20,endpoint=4),counts
ledgers=[read(p) for p in root.glob('*/storage-ledger.json')];actual=tree_bytes(root)
save_json(dest/'resources.json',dict(attempts=counts,maximum= dict(trajectory=5,feedback=25,rtc_only=1),replacement_trajectories=0,actual_campaign_bytes_before_compaction=actual,peak_ceiling_bytes=30_000_000_000,conservative_max_reserved_bound=max(x['historical_bytes']+x['charged_bytes']+100_000_000 for x in ledgers),cumulative_native_write_bytes=sum(x['native_written_bytes'] for x in ledgers),free_space_reserve_bytes=2_000_000_000,storage_scope='actual reconciled footprint vs conservative charged-stage upper bound; write volume not retained footprint; all planned reservation/free-space gates passed'))
print('timings',[(r['family'],r['rep'],r['exit'],r['RTC_wall_occupancy_bounds'],r['CPU']) for r in rows]);print(counts,actual)
