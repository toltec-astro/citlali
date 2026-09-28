"""Preserve completed two-stage evidence and enumerate scratch; never deletes."""
from pathlib import Path
import json,gzip,hashlib,shutil
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-two-stage-20260928')
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');D=W/'validation/rtc_two_stage_2026-09-28';DEST=D/'compact'
SUFFIXES={'.f64','.f32','.u32','.u8','.u16','.i32','.i64','.u64','.bits'}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,o):
 b=(json.dumps(o,indent=2)+'\n').encode();p.write_bytes(gzip.compress(b,compresslevel=6,mtime=0) if p.suffix=='.gz' else b)
def main():
 names=['B-REF-rep1','Q-REF','Q-C-core-plus','Q-C-core-minus','Q-C-shoulder']
 assert sorted(p.parent.name for p in ROOT.glob('*/LAUNCH.json'))==sorted(names)
 for n in names:
  assert json.loads((ROOT/n/'LAUNCH.json').read_text())['status']=='completed'
  assert json.loads((ROOT/n/'EXECUTION.json').read_text())['completed_cycle']==5
 protected=json.loads((ROOT/'PROTECTED_INPUTS.json').read_text())
 for r in protected:assert digest(Path(r['path']))==r['sha256'],r['path']
 assert not DEST.exists();DEST.mkdir();kept=[];scratch=[]
 for p in sorted(ROOT.rglob('*')):
  assert not p.is_symlink(),str(p)
  if not p.is_file():continue
  rel=p.relative_to(ROOT);h=digest(p);n=p.stat().st_size
  if p.suffix in SUFFIXES:
   assert rel.parts[0] in names and rel.parts[1] in ('upstream','M0','initial','run','native'),str(p)
   scratch.append(dict(path=str(p),bytes=n,sha256=h));continue
  q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);encoding='identity'
  if p.suffix=='.fits' or p.suffix=='.yaml' and n>100000:
   q=Path(str(q)+'.gz');q.write_bytes(gzip.compress(p.read_bytes(),compresslevel=6,mtime=0));assert hashlib.sha256(gzip.decompress(q.read_bytes())).hexdigest()==h;encoding='gzip'
  else:shutil.copyfile(p,q);assert digest(q)==h
  kept.append(dict(source=str(rel),source_sha256=h,path=str(q.relative_to(DEST)),sha256=digest(q),bytes=q.stat().st_size,encoding=encoding))
 save(D/'COMPACT_MANIFEST.json',dict(files=kept,bytes=sum(r['bytes'] for r in kept),scope='five completed trajectories, full B/Q population and fixed windows, all FITS/receipts/configuration/logs, saved comparison and response controls; native arrays enumerated only'))
 save(D/'NATIVE_SCRATCH_PROPOSAL.json.gz',dict(root=str(ROOT),files=scratch,bytes=sum(r['bytes'] for r in scratch),applied=False,owner_approval=False,operation='none; proposed native scratch list only; no deletion authority'))
 save(D/'PROTECTED_INPUTS.json',protected)
 print(json.dumps(dict(kept=len(kept),compact_bytes=sum(r['bytes'] for r in kept),scratch_files=len(scratch),scratch_bytes=sum(r['bytes'] for r in scratch),deleted=0,protected_unchanged=len(protected))))
if __name__=='__main__':main()
