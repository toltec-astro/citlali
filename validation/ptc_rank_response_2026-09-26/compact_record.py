"""Preserve compact experiment first; then remove its enumerated inactive payloads."""
import argparse,gzip,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-point-152430-a2000-ptc-rank-20260926')
REPO=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor')
DEST=REPO/'validation/ptc_rank_response_2026-09-26/compact'
SUFFIXES={'.f64','.f32','.u32','.u8','.u16','.i32','.i64','.u64','.bits'}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def protect():
 for d in ROOT.glob('rank-*/freeze/dependencies.json'):
  for b in read(d)['inputs']:
   p=Path(b['path']);assert not p.resolve().is_relative_to(ROOT)
   assert p.is_file() and digest(p)==b['sha256'],str(p)
def compact():
 assert not DEST.exists() and read(ROOT/'EXECUTION.json')['status']!='running'
 assert (ROOT/'report/matrix.json').exists();protect();DEST.mkdir()
 kept=[];removed=[]
 for p in sorted(ROOT.rglob('*')):
  assert not p.is_symlink(),str(p)
  if not p.is_file():continue
  rel=p.relative_to(ROOT)
  if p.suffix in SUFFIXES:
   assert rel.parts[0] in ('rank-5','rank-10','rank-15') and rel.parts[1] in ('upstream','M0','initial','run'),str(p)
   removed.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)));continue
  q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True)
  if p.suffix=='.fits' or (p.suffix=='.yaml' and p.stat().st_size>100000):
   raw=p.read_bytes();q=Path(str(q)+'.gz');q.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0));assert gzip.decompress(q.read_bytes())==raw
   kept.append(dict(source=str(rel),source_sha256=hashlib.sha256(raw).hexdigest(),path=str(q.relative_to(DEST)),sha256=digest(q),bytes=q.stat().st_size,encoding='gzip'))
  else:
   shutil.copyfile(p,q);assert digest(p)==digest(q)
   kept.append(dict(source=str(rel),source_sha256=digest(p),path=str(q.relative_to(DEST)),sha256=digest(q),bytes=q.stat().st_size,encoding='identity'))
 save(DEST.parent/'COMPACT_MANIFEST.json',dict(files=kept,bytes=sum(r['bytes'] for r in kept),old_evidence_referenced_not_copied=True))
 save(DEST.parent/'NATIVE_REMOVALS.json',dict(root=str(ROOT),files=removed,bytes=sum(r['bytes'] for r in removed),applied=False))
 print(json.dumps(dict(retained_files=len(kept),retained_bytes=sum(r['bytes'] for r in kept),native_files=len(removed),native_bytes=sum(r['bytes'] for r in removed))))
def clean():
 assert read(ROOT/'EXECUTION.json')['status']!='running';protect()
 manifest=read(DEST.parent/'COMPACT_MANIFEST.json');plan=read(DEST.parent/'NATIVE_REMOVALS.json');assert not plan['applied']
 # Compact files and exact deletion plan must already be durable tracked history.
 for rel in ('COMPACT_MANIFEST.json','NATIVE_REMOVALS.json'):
  subprocess.run(['git','ls-files','--error-unmatch',str(DEST.parent/rel)],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
 assert not subprocess.check_output(['git','status','--porcelain','--',str(DEST)],cwd=REPO,text=True).strip()
 for row in manifest['files']:
  p=DEST/row['path'];assert p.is_file() and digest(p)==row['sha256']
 for row in plan['files']:
  p=Path(row['path']);assert p.resolve().is_relative_to(ROOT) and not p.is_symlink() and p.suffix in SUFFIXES
  assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
 before=shutil.disk_usage(ROOT).free
 for row in plan['files']:Path(row['path']).unlink()
 protect();plan.update(applied=True,protected_inputs_unchanged=True,free_space_delta_bytes=shutil.disk_usage(ROOT).free-before,compact_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
 save(DEST.parent/'CLEANUP.json',plan);print(json.dumps({k:v for k,v in plan.items() if k!='files'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['compact','clean']);a=p.parse_args();compact() if a.mode=='compact' else clean()
