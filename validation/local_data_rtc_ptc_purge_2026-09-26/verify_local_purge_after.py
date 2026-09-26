from pathlib import Path
import json,gzip,os,stat,time,hashlib,subprocess,collections,sys
W=Path('/private/tmp/citlali-point-additive-20260923');V=W/'validation/local_data_rtc_ptc_purge_2026-09-26';R=Path('/Users/gwilson/work_toltec/local_data');sys.path.insert(0,str(W/'tools/timestream_successor'))
from local_data_purge import read,save,identity,digest,explain_purged
start=time.monotonic();receipt=read(V/'EXECUTION_RECEIPT.json');assert receipt['status']=='completed';targets=read(V/'TARGETS.json.gz');bytarget={r['path']:r for r in targets};before=read(V/'INVENTORY_BEFORE.json.gz');head=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip()
for r in targets:assert not (R/r['path']).exists() and not (R/r['path']).is_symlink(),r['path']
original_retained=0;protected_raw_files=0
headers=read(V/'NETCDF_HEADERS.json.gz');protected_netCDF={h['path']for h in headers if h['path']not in bytarget}
for r in before:
 if r['path']in bytarget:continue
 now=identity(R/r['path']);assert now=={k:r[k]for k in now},r['path'];original_retained+=1
 if r['path']in protected_netCDF:protected_raw_files+=1
for r in read(V/'SYMLINKS.json'):assert (R/r['path']).is_symlink()and os.readlink(R/r['path'])==r['target']
pres=read(V/'MIXED_CONTAINER_PRESERVATION.json.gz')
for r in pres:
 if r['status']=='pass':assert identity(R/r['companion'])==r['identity'] and digest(R/r['companion'])==r['sha256']
# Publish current availability separately from historical seals and indexes.
pathsfile=R/'LOCAL_DATA_PURGE_2026-09-26_PATHS.json.gz';assert not pathsfile.exists();save(pathsfile,sorted(bytarget))
index=dict(status='completed',directive='2026-09-26 local_data inventory and RTC/PTC purge',files=len(targets),logical_bytes=sum(r['bytes']for r in targets),paths_file=pathsfile.name,paths_sha256=digest(pathsfile),durable_repository='/Users/gwilson/GitHub/citlali',durable_git_record='refs/heads/codex/map-successor-first-forward-001:validation/local_data_rtc_ptc_purge_2026-09-26/RECOVERY_AND_DISPOSITION.md',pre_deletion_commit=head,target_manifest_sha256=digest(V/'TARGETS.json.gz'),disposition='Saved RTC/PTC restart state deliberately discarded; useful results remain; explicit reprocessing from documented upstream acquisition/tune/calibration required. No automatic regeneration.',historical_seals='Unchanged original seals are historical; referenced deleted RTC/PTC payloads intentionally absent.',non_targets='Original raw/tune/calibration/team inputs, source, maps, fit/basis/PSD and independent diagnostics unchanged.')
current=R/'LOCAL_DATA_PURGE_2026-09-26.json';assert not current.exists();save(current,index)
old=R/'citlali-validation/development-runs/successor-point-152389-importance-and-archive-compaction-20260926';newindex=old/'CURRENT_ARCHIVE_INDEX.json';assert not newindex.exists();save(newindex,dict(status='superseded by owner-directed RTC/PTC purge',current_purge_index=str(current),historical_index=str(old/'ARCHIVE_INDEX.json'),historical_index_sha256=digest(old/'ARCHIVE_INDEX.json'),historical_manifest_sha256=digest(old/'FILES_SHA256SUMS'),old_byte_restoration='No longer available for purged canonical files. Keep historical action/seal unchanged; use explicitly documented upstream reprocessing instead.',science='Scientific report/maps/measurements remain unchanged; new availability does not alter old conclusions.'))
# Check actual current reader behavior for a directly purged path and a prior deduplication counterpart.
try:explain_purged(R/targets[0]['path'])
except FileNotFoundError as e:assert 'INTENTIONALLY PURGED'in str(e)
else:raise AssertionError('purge state not detected')
import point_archive_compaction as pa
oldactions=pa.read(old/'ACTION_MANIFEST.json.gz');a=next(x for x in oldactions if str(Path(x['retained']).relative_to(R))in bytarget)
for function in [pa.resolve,pa.explain_missing]:
 try:function(a['path'],oldactions)
 except FileNotFoundError as e:assert 'INTENTIONALLY PURGED'in str(e)
 else:raise AssertionError('old restoration advice not superseded')
# Full post-operation stat census, without payload hashing or following symlinks/mounts.
rows=[];links=[];errors=[];dev=R.stat().st_dev;top=collections.Counter();allocated=collections.Counter();dirs=0
for parent,ds,fs in os.walk(R,followlinks=False,onerror=lambda e:errors.append(str(e))):
 for n in list(ds):
  p=Path(parent)/n;s=p.lstat()
  if p.is_symlink():links.append(str(p.relative_to(R)));ds.remove(n)
  elif s.st_dev!=dev:errors.append('skipped mount '+str(p));ds.remove(n)
  else:dirs+=1
 for n in fs:
  p=Path(parent)/n;s=p.lstat();rel=str(p.relative_to(R))
  if p.is_symlink():links.append(rel);continue
  if not stat.S_ISREG(s.st_mode)or s.st_dev!=dev:errors.append('nonregular/mount '+rel);continue
  row=dict(path=rel,**identity(p));rows.append(row);key=rel.split('/')[0];top[key]+=s.st_size;allocated[key]+=s.st_blocks*512
save(V/'INVENTORY_AFTER.json.gz',rows)
allowed_new={x['companion']for x in pres if x['status']=='pass'}|{str(p.relative_to(R))for p in [pathsfile,current,newindex]}
beforepaths={r['path']for r in before};newpaths={r['path']for r in rows}-beforepaths;assert newpaths==allowed_new,(newpaths-allowed_new,allowed_new-newpaths)
oldtotal=sum(r['bytes']for r in before);aftertotal=sum(r['bytes']for r in rows);newbytes=sum(r['bytes']for r in rows if r['path']in allowed_new);assert aftertotal==oldtotal-receipt['logical_bytes']+newbytes
b=int(receipt['before_df'].splitlines()[1].split()[3])*1024;e=int(receipt['after_df'].splitlines()[1].split()[3])*1024
s=dict(status='pass',files_before=len(before),files_after=len(rows),directories_after=dirs,symlinks_after=len(links),traversal_errors=errors,removed_files_absent=len(targets),all_original_non_target_files_unchanged=original_retained,protected_original_netcdf_files_unchanged=protected_raw_files,all_original_symlinks_unchanged=True,retained_mixed_companions_verified=len(pres),new_metadata_paths=[str(p.relative_to(R))for p in [pathsfile,current,newindex]],before_logical_bytes=oldtotal,deleted_logical_bytes=receipt['logical_bytes'],new_retained_companion_and_index_bytes=newbytes,after_logical_bytes=aftertotal,net_logical_reduction_bytes=oldtotal-aftertotal,after_allocated_file_blocks_bytes=sum(r['blocks']for r in rows),observed_purge_free_space_gain_bytes=e-b,free_immediately_after_bytes=e,by_top_level=[dict(path=k,bytes=v,blocks=allocated[k])for k,v in top.most_common()],post_purge_readers='direct purge and prior deduplication canonical path both explicitly require reprocessing; no automatic recovery',seconds=time.monotonic()-start,qualification='No raw payload hash or numerical rerun. All original non-target stat identities including size/mtime/ctime/inode unchanged; mixed companions freshly hashed. df delta measures deletion interval; companion preservation and committed records consumed space before that interval.')
save(V/'POST_PURGE_VERIFICATION.json',s);save(V/'CURRENT_PURGE_INDEX_COPY.json',index)
print(json.dumps({k:v for k,v in s.items()if k!='by_top_level'},indent=2))
