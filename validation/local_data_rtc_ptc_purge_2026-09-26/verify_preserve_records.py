from pathlib import Path
import gzip,json,tarfile,hashlib,time,shutil
V=Path('/private/tmp/citlali-point-additive-20260923/validation/local_data_rtc_ptc_purge_2026-09-26');start=time.monotonic();index=json.load(gzip.open(V/'RECOVERY_RECORD_INDEX.json.gz','rt'));expected={r['member']:r for r in index};seen=set()
with tarfile.open(V/'RECOVERY_RECORDS.tar.gz','r|gz')as tar:
 for m in tar:
  assert m.isfile()and m.name in expected and m.name not in seen
  row=expected[m.name];assert m.size==row['bytes'];f=tar.extractfile(m);h=hashlib.sha256()
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  assert h.hexdigest()==row['sha256'];seen.add(m.name)
assert seen==set(expected)
parts=[];whole=hashlib.sha256()
with open(V/'RECOVERY_RECORDS.tar.gz','rb')as f:
 for i in range(10):
  b=f.read(48<<20)
  if not b:break
  p=V/f'RECOVERY_RECORDS.tar.gz.part{i:02d}';p.write_bytes(b);h=hashlib.sha256(p.read_bytes()).hexdigest();whole.update(p.read_bytes());parts.append(dict(path=p.name,bytes=len(b),sha256=h))
assert sum(p['bytes']for p in parts)==(V/'RECOVERY_RECORDS.tar.gz').stat().st_size
r=dict(status='pass',record_paths=len(index),unique_record_payloads=len(seen),all_original_record_hashes_verified=True,archive_sha256=whole.hexdigest(),parts=parts,seconds=time.monotonic()-start,reassemble='Concatenate parts in numeric order; original tar member sha256/<digest> is mapped to each relative local_data path by RECOVERY_RECORD_INDEX.json.gz. Do not extract into local_data automatically.')
(V/'DURABLE_PRESERVATION_CHECK.json').write_text(json.dumps(r,indent=2)+'\n')
# Only discard the just-created unsplit staging copy, after every retained part is verified.
(V/'RECOVERY_RECORDS.tar.gz').unlink()
for n in ['inventory_all_local_data.py','inventory_netcdf_headers.py','preserve_local_data_records.py','verify_preserve_records.py','build_purge_targets.py','build_local_observations.py']:shutil.copy2(Path('/private/tmp')/n,V/n)
print('Verified',len(index),'records in',len(parts),'parts; bytes',sum(p['bytes']for p in parts),flush=True)
