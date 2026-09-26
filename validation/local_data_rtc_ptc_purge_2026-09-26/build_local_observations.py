from pathlib import Path
import json,gzip,collections,re,shutil
R=Path('/Users/gwilson/work_toltec/local_data');V=Path('/private/tmp/citlali-point-additive-20260923/validation/local_data_rtc_ptc_purge_2026-09-26')
for f in ['local-readiness-configs.json','local-readiness-header-review.json','local-readiness-time-review.json','local-raw-readiness-review.md','check_readiness_headers.py','review_local_readiness.py','local-purge-safety-review-notes.md','local-purge-archive-header-review.json','local-purge-npz-header-review.json']:shutil.copy2(Path('/private/tmp')/f,V/f)
heads=json.load(gzip.open(V/'NETCDF_HEADERS.json.gz','rt'));inventory=json.load(gzip.open(V/'INVENTORY_BEFORE.json.gz','rt'));lookup={r['path']:r for r in inventory};hd={h['path']:h for h in heads}
configs=json.load(open(V/'local-readiness-configs.json'));skyreview={tuple(r['identity']):r for r in json.load(open(V/'local-readiness-header-review.json'))};times={tuple(r['identity']):r for r in json.load(open(V/'local-readiness-time-review.json'))}
def scalar(x):
 while isinstance(x,list)and len(x)==1:x=x[0]
 return x
def val(m,k,default=None):return scalar(m.get(k,default))
def ident(m,prefix):
 try:return tuple(int(val(m,prefix+k))for k in ['ObsNum','SubObsNum','ScanNum'])
 except:return None
raw=collections.defaultdict(list);processed=collections.defaultdict(list);support=collections.defaultdict(list)
for h in heads:
 m=h.get('metadata',{});id=ident(m,'Header.Toltec.');name=Path(h['path']).name
 if id and 'Data.Toltec.Is'in h['variables'] and 'Data.Toltec.Qs'in h['variables']:raw[(id,int(val(m,'Header.Toltec.ObsType',-1)))].append(h)
 elif id and any(k.startswith('Data.Kids.')for k in h['variables']):processed[id].append(h)
 if name.startswith(('tel_','hwpr_','toltec_hk_')):
  sid=ident(m,'Header.Dcs.') or ident(m,'Header.Hwpr.') or ident(m,'Header.Toltec.')
  if sid:support[sid].append(h)
# This is an input-presence/structure inventory; no byte identity claim for duplicates.
fitindex=collections.defaultdict(list)
for x in inventory:
 name=Path(x['path']).name
 m=re.match(r'(toltec\d+_\d{6}_\d{3}_\d{4}_)',name)
 if m and name.endswith('.txt'):fitindex[m[1]].append(x['path'])
rows=[]
for (id,typ),parts in sorted(raw.items()):
 pernet=collections.defaultdict(list)
 for h in parts:pernet[int(val(h['metadata'],'Header.Toltec.RoachIndex',-1))].append(h)
 conflicts=[];chosen=[]
 for nw,hs in sorted(pernet.items()):
  hs=sorted(hs,key=lambda x:('citlali-validation/v1/'not in x['path'],x['path']));chosen.append(hs[0])
  fingerprints={json.dumps([h['bytes'],h['variables'].get('Data.Toltec.Is'),h.get('time_endpoints'),{k:v for k,v in h['metadata'].items()if 'TargSweep'in k}],sort_keys=True)for h in hs}
  if len(fingerprints)>1:conflicts.append(dict(network=nw,kind='copy structure/size/time/tune disagreement; no payload hash',paths=[h['path']for h in hs]))
 sr=skyreview.get(id)if typ==0 else None;conf=[]
 for c in configs:
  name=c.get('identity',{}).get('name','');cid=tuple(int(v)for v in name.split('_')) if re.fullmatch(r'\d+_\d+_\d+',name)else None
  if cid==id:conf.append(c)
 configured_interfaces=sorted({d['interface']for c in conf for d in c['dependencies']if d['role']=='data'})
 telescope=[h for h in support[id]if Path(h['path']).name.startswith('tel_')and 'recomputed'not in Path(h['path']).name]
 derived_tel=[h for h in support[id]if Path(h['path']).name.startswith('tel_')and 'recomputed'in Path(h['path']).name]
 tunes=[];dependencies={h['path']for h in chosen}
 for nw,hs in sorted(pernet.items()):
  m=hs[0]['metadata'];tid=ident(m,'Header.Toltec.TargSweep')
  # TargSweep uses ObsNum/SubObsNum/ScanNum directly without dot.
  th=[h for (key,t),ps in raw.items()if key==tid and t!=0 for h in ps if int(val(h['metadata'],'Header.Toltec.RoachIndex',-1))==nw]
  ph=[h for h in processed.get(tid,[])if int(val(h['metadata'],'Header.Toltec.RoachIndex',-1))==nw]
  fit=[]
  if tid:
   pattern=f'toltec{nw}_{tid[0]:06d}_{tid[1]:03d}_{tid[2]:04d}_'
   fit=fitindex.get(pattern,[])
  tunes.append(dict(network=nw,identity=tid,raw_paths=[h['path']for h in th],processed_paths=[h['path']for h in ph],fit_report_paths=fit,evidence='Header.Toltec.TargSweepObsNum/SubObsNum/ScanNum per network; not obsnum proximity'))
  if th:dependencies.add(sorted(th,key=lambda h:h['path'])[0]['path'])
  if ph:dependencies.add(sorted(ph,key=lambda h:h['path'])[0]['path'])
  if fit:dependencies.add(sorted(fit)[0])
 if telescope:dependencies.add(sorted(telescope,key=lambda h:h['path'])[0]['path'])
 aptdeps=[dict(d) for c in conf for d in c['dependencies']if d['role']!='data']
 for c in conf:
  for d in c['dependencies']:
   if d['role']=='data' and not re.fullmatch('toltec[0-9]+',d['interface']) and d['local_candidates']:
    exact=[z for z in d['local_candidates']if d['recorded_path'].endswith('/'+z['path'])]
    dependencies.add((exact or d['local_candidates'])[0]['path'])
 for d in aptdeps:
  cand=d['local_candidates']
  matches=[z for z in cand if d['recorded_path'].endswith('/'+z['path'])]
  exact=sorted(matches,key=lambda z:-len(z['path']))[:1]
  d['exact_local_path']=exact[0]['path'] if len(exact)==1 else None
  d['identity_qualification']='Exact preserved project-relative path; not arbitrary same-basename APT replacement' if len(exact)==1 else 'No exact project-relative path match; candidate APT variants are not interchangeable'
  if len(exact)==1:
   dependencies.add(exact[0]['path'])
   import hashlib
   d['sha256']=hashlib.sha256((R/exact[0]['path']).read_bytes()).hexdigest()
 selected_expected=sorted(int(i[6:])for i in configured_interfaces if re.fullmatch('toltec[0-9]+',i))
 missing=[n for n in selected_expected if n not in pernet]
 status='completeness unverified (tune/VNA or other acquisition; no full expected input inventory)'
 if typ==0:
  if not telescope or any(not t['raw_paths']for t in tunes):status='partial raw package: original telescope and/or associated raw tunes missing'
  elif conflicts:status='completeness unverified: conflicting duplicate metadata'
  elif selected_expected and not missing and all(d['exact_local_path']for d in aptdeps):status='complete against saved reduction input inventory, with identified local raw/processed tune dependencies'
  else:status='raw/tune/telescope structurally present; expected inventory and calibration readiness unverified'
 selected_tel=(telescope or derived_tel);tm=selected_tel[0]['metadata']if selected_tel else {}
 rows.append(dict(identity=list(id),obs_type=typ,source=val(tm,'Header.Source.SourceName'),mode=val(tm,'Header.Dcs.ObsPgm'),goal=val(tm,'Header.Dcs.ObsGoal'),date_UT_header=val(tm,'Header.TimePlace.UTDate'),networks=sorted(pernet),array_assignment='usual0-6 a1100,7-10 a1400,11-12 a2000; historical network presence from headers',parts=[dict(path=h['path'],bytes=h['bytes'],network=val(h['metadata'],'Header.Toltec.RoachIndex'),dimensions=h['dimensions'],data_shape=h['variables'].get('Data.Toltec.Is'),time_endpoints=h.get('time_endpoints'),header_start=val(h['metadata'],'Header.Toltec.ObsStartTime'),header_end=val(h['metadata'],'Header.Toltec.ObsEndTime'))for h in parts],selected_raw_paths=[h['path']for h in chosen],raw_all_copies_bytes=sum(h['bytes']for h in parts),one_per_network_raw_bytes=sum(h['bytes']for h in chosen),duplicate_copy_count=len(parts)-len(pernet),conflicts=conflicts,expected_networks_from_saved_config=selected_expected,expected_inventory_evidence=[c['config']for c in conf],missing_expected_networks=missing,selected_mask_values=sorted({val(h['metadata'],'Header.Toltec.SelectedMask',-1)for h in parts}),expected_scope='Saved reduction input lists only; remote acquisition archive not checked. SelectedMask corroborates observed networks but historical bit semantics not independently established.',tune_associations=tunes,original_telescope_paths=[h['path']for h in telescope],recomputed_telescope_paths=[h['path']for h in derived_tel],other_support_paths=[h['path']for h in support[id]if not Path(h['path']).name.startswith('tel_')],timing_check=times.get(id),calibration_dependencies=aptdeps,archive_provenance=[dict(config=c['config'],recorded_paths=[d['recorded_path']for d in c['dependencies']],status='historically recorded local configuration; current Unity availability unverified')for c in conf],readiness=status,package_selected_input_paths=sorted(dependencies),package_selected_input_bytes=sum(lookup[p]['bytes']for p in dependencies),package_size_qualification='One metadata-matched file per network/dependency; exact chosen paths recorded; other physical copies remain. No payload deduplication claim or raw deletion. Shared inputs counted by path in corpus totals.'))
with gzip.open(V/'OBSERVATIONS.json.gz','wt')as f:json.dump(rows,f,separators=(',',':'))
with gzip.open(V/'SUPPORT_AND_PROCESSED_GROUPS.json.gz','wt')as f:json.dump(dict(processed=[dict(identity=k,paths=[h['path']for h in v],bytes=sum(h['bytes']for h in v),readiness='processed tune products only; corresponding raw/telescope completeness must be checked separately')for k,v in processed.items()],support=[dict(identity=k,paths=[h['path']for h in v],bytes=sum(h['bytes']for h in v),readiness='support products only; not a complete detector acquisition package')for k,v in support.items()]),f,separators=(',',':'))
sky=[r for r in rows if r['obs_type']==0]
lines=['# Local raw observation and tune inventory','', 'Completeness below is against local saved reduction input lists, not a live Unity acquisition inventory. All copies and all tune/VNA groups are in OBSERVATIONS.json.gz; processed/support-only identities are separately retained. Header readability and endpoints do not establish sample quality.','', '| Observation / sub / scan | Source / mode | Networks | Raw GB, one per network | Selected input GB | Raw tune identity | Readiness |','|---|---|---:|---:|---:|---|---|']
for r in sky:lines.append('| '+ ' | '.join(['/'.join(map(str,r['identity'])),str(r['source'])+' / '+str(r['mode']),str(len(r['networks'])),f"{r['one_per_network_raw_bytes']/1e9:.3f}",f"{r['package_selected_input_bytes']/1e9:.3f}",','.join(str(t)for t in sorted({tuple(t['identity'])for t in r['tune_associations']if t['identity']})),r['readiness']])+' |')
(V/'OBSERVATIONS.md').write_text('\n'.join(lines)+'\n')
print('All original detector acquisition groupings',len(rows),'sky',len(sky),'processed IDs',len(processed),'supportIDs',len(support));print(collections.Counter(r['readiness']for r in sky));print([(r['identity'][0],round(r['package_selected_input_bytes']/1e9,3),len(r['conflicts']))for r in sky])
