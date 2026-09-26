from pathlib import Path
import json,gzip,hashlib,tarfile,io,time
R=Path('/Users/gwilson/work_toltec/local_data');V=Path('/private/tmp/citlali-point-additive-20260923/validation/local_data_rtc_ptc_purge_2026-09-26')
r=json.load(gzip.open(V/'INVENTORY_BEFORE.json.gz','rt'));a=json.load(gzip.open(V/'RECOVERY_RECORD_CANDIDATES.json.gz','rt'));chosen={x['path']:x for x in a}
for x in r:
 if Path(x['path']).name in {'FILES_SHA256SUMS','SHA256SUMS'}:chosen[x['path']]=x
index=[];unique=set();errors=[];total=0;start=time.monotonic()
with tarfile.open(V/'RECOVERY_RECORDS.tar.gz','w:gz',compresslevel=6) as tar:
 for i,x in enumerate(chosen.values()):
  p=R/x['path']
  try:
   s=p.lstat();assert not p.is_symlink() and s.st_size==x['bytes'] and s.st_mtime_ns==x['mtime_ns'] and s.st_ino==x['ino'];b=p.read_bytes();h=hashlib.sha256(b).hexdigest();member='sha256/'+h
   if h not in unique:
    info=tarfile.TarInfo(member);info.size=len(b);info.mode=0o600;info.mtime=0;tar.addfile(info,io.BytesIO(b));unique.add(h);total+=len(b)
   index.append(dict(path=x['path'],sha256=h,bytes=len(b),member=member))
  except Exception as e:errors.append(dict(path=x['path'],error=str(e)))
  if (i+1)%20000==0:print('Records',i+1,'unique',len(unique),flush=True)
with gzip.open(V/'RECOVERY_RECORD_INDEX.json.gz','wt') as f:json.dump(index,f,separators=(',',':'))
(V/'RECOVERY_PRESERVATION.json').write_text(json.dumps(dict(files=len(index),unique_contents=len(unique),original_logical_bytes=sum(x['bytes'] for x in index),deduplicated_logical_bytes=total,compressed_bytes=(V/'RECOVERY_RECORDS.tar.gz').stat().st_size,errors=errors,seconds=time.monotonic()-start,selection='All compact <=2MB md/yaml/yml/toml/sh and selected JSON/table/manifest records except array/operator records and dependency/build trees; all original FILES_SHA256SUMS regardless size. Nonselected large diagnostics/maps/inputs remain in place and are not targeted. Exact per-path preservation index; duplicate contents stored once.'),indent=2)+'\n')
print('DONE',len(index),len(unique),total,(V/'RECOVERY_RECORDS.tar.gz').stat().st_size,'errors',len(errors),flush=True)
