import json,gzip,collections,pathlib,re
W=pathlib.Path('/private/tmp/citlali-point-additive-20260923');x=json.load(gzip.open(W/'validation/local_data_rtc_ptc_purge_2026-09-26/NETCDF_HEADERS.json.gz','rt'))
raw=[r for r in x if 'Data.Toltec.Is' in r['variables'] and 'Data.Toltec.Qs' in r['variables'] and 'Header.Toltec.ObsNum' in r['metadata']]
byid=collections.defaultdict(list)
for r in raw:
 m=r['metadata'];byid[tuple(m.get('Header.Toltec.'+k) for k in ['ObsNum','SubObsNum','ScanNum','RoachIndex','ObsType'])].append(r)
obs=collections.defaultdict(list)
for r in raw:
 m=r['metadata']
 if m.get('Header.Toltec.ObsType')==0:obs[tuple(m.get('Header.Toltec.'+k) for k in ['ObsNum','SubObsNum','ScanNum'])].append(r)
rows=[]
for key,rr in sorted(obs.items()):
 ids={r['metadata']['Header.Toltec.RoachIndex'] for r in rr};chosen=[next(r for r in rr if r['metadata']['Header.Toltec.RoachIndex']==n) for n in sorted(ids)]
 tunes={};missing=[]
 for r in chosen:
  m=r['metadata'];n=m['Header.Toltec.RoachIndex'];ti=tuple(m.get('Header.Toltec.'+k) for k in ['TargSweepObsNum','TargSweepSubObsNum','TargSweepScanNum']);found=[q for k,ls in byid.items() if k[:3]==ti and k[3]==n and k[4]!=0 for q in ls];tunes[n]={'target_id':ti,'raw_tune_paths':[q['path'] for q in found]}
  if not found:missing.append(n)
 row={'identity':key,'networks':sorted(ids),'tune_associations':tunes,'raw_tune_missing_networks':missing,'masks':sorted({r['metadata'].get('Header.Toltec.SelectedMask',-1) for r in rr}),'copies':len(rr),'unique_raw_bytes':sum(r['bytes'] for r in chosen),'sample_count_range':[min(r['variables']['Data.Toltec.Is']['shape'][0] for r in chosen),max(r['variables']['Data.Toltec.Is']['shape'][0] for r in chosen)],'time_endpoints':{r['metadata']['Header.Toltec.RoachIndex']:r.get('time_endpoints',{}) for r in chosen},'selected_paths':[r['path'] for r in chosen]};rows.append(row)
 print(key,sorted(ids),row['masks'],row['sample_count_range'],'tune',sorted(set(tuple(a['target_id']) for a in tunes.values())),'missing',missing)
json.dump(rows,open('/private/tmp/local-readiness-header-review.json','w'),indent=2)
