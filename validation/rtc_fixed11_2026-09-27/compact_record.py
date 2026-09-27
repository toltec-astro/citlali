"""Exact bounded compact preservation, then committed-manifest native cleanup."""
import argparse,gzip,hashlib,json,shutil,subprocess
from pathlib import Path
import psutil
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-fixed11-20260927')
DEST=Path(__file__).parent/'compact'
SUFFIXES={'.f64','.f32','.u32','.u8','.u16','.i32','.i64','.u64','.bits'}
CHECK={'cycle-01','cycle-05'}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def save(p,o):
 b=(json.dumps(o,indent=2)+'\n').encode();p.write_bytes(gzip.compress(b,compresslevel=6,mtime=0) if p.suffix=='.gz' else b)
def inactive():
 assert ROOT.resolve()==ROOT and ROOT.is_dir()
 assert read(ROOT/'COMPARISON.json')['status']=='completed'
 for p in psutil.process_iter(['pid','cmdline']):
  if p.pid==psutil.Process().pid:continue
  a=p.info['cmdline'] or []
  if not a:continue
  is_work=Path(a[0]).name=='citlali' or any(Path(x).name=='point_pipeline.py' for x in a[:3])
  assert not(is_work and any(str(ROOT) in x for x in a)),p.info

def protect_prior():
 prior=REPO/'validation/rtc_cost_response_2026-09-27'
 sealed=read(ROOT/'PRIOR_PRESERVATION.json')
 assert digest(prior/'COMPACT_MANIFEST.json')==sealed['sha256']
 for r in read(prior/'COMPACT_MANIFEST.json')['files']:assert digest(prior/'compact'/r['path'])==r['sha256']
def protect():
 protect_prior()
 unique={}
 for d in ROOT.glob('*-REF-rep*/freeze/dependencies.json'):
  for r in read(d)['inputs']:
   p=Path(r['path']);assert not p.resolve().is_relative_to(ROOT)
   assert p.is_file() and (p not in unique or unique[p]==r['sha256']);unique[p]=r['sha256']
 for p,h in unique.items():assert digest(p)==h,str(p)
 return [dict(path=str(p),sha256=h,bytes=p.stat().st_size) for p,h in sorted(unique.items())]

def redundant(rel):
 top=rel.parts[0]
 return (top in ('B-REF-rep2','B-REF-rep3','C-REF-rep2','C-REF-rep3')) and ('products' in rel.parts or rel.suffix=='.fits')

def necessary_fits(rel):
 parts=rel.parts
 if 'freeze' in parts or 'endpoint' in parts or any(x.startswith('checkpoint-') for x in parts):return True
 if '/M0/products/' in '/'+str(rel):return rel.name=='map.fits'
 if 'initial' in parts:return rel.name in ('total.fits','response.fits','candidate.fits')
 if any(x in CHECK for x in parts):return rel.name in ('total.fits','response.fits','candidate.fits')
 return False

def compact():
 inactive();assert not DEST.exists();protected=protect()
 for out in ROOT.glob('*-REF-rep*'):
  if not out.is_dir():continue
  if out.name!='C-REF-rep1':assert read(out/'REGRESSION.json')['pass_']
  if out.name!='C-REF-rep1':assert read(out/'NATIVE_REGRESSION.json')['pass_']
 DEST.mkdir();kept=[];removed=[];hash_only=[]
 for p in sorted(ROOT.rglob('*')):
  assert not p.is_symlink(),str(p)
  if not p.is_file():continue
  rel=p.relative_to(ROOT);n=p.stat().st_size;h=digest(p)
  if p.suffix in SUFFIXES:
   assert rel.parts[0].startswith(('A-','B-','C-')) and rel.parts[1] in ('upstream','M0','initial','run','native'),str(p)
   removed.append(dict(path=str(p),bytes=n,sha256=h));continue
  if redundant(rel) or (p.suffix=='.fits' and not necessary_fits(rel)):
   hash_only.append(dict(path=str(rel),bytes=n,sha256=h,reason='redundant regression-verified repeat or non-checkpoint product; all-cycle numerical summaries retained'));continue
  q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);encoding='identity'
  if p.suffix=='.fits' or p.suffix=='.yaml' and n>100000:
   raw=p.read_bytes();q=Path(str(q)+'.gz');q.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0));assert hashlib.sha256(gzip.decompress(q.read_bytes())).hexdigest()==h;encoding='gzip'
  else:shutil.copyfile(p,q);assert digest(q)==h
  kept.append(dict(source=str(rel),source_sha256=h,path=str(q.relative_to(DEST)),sha256=digest(q),bytes=q.stat().st_size,encoding=encoding))
 save(DEST.parent/'COMPACT_MANIFEST.json',dict(files=kept,bytes=sum(x['bytes'] for x in kept),old_reference_controls_referenced_in_place=True,scope='Fixed11 B/C populations, source, maps and timing; B1/C1 plus C injections retained; redundant repeat payloads hash-only'))
 save(DEST.parent/'HASH_ONLY.json.gz',dict(files=hash_only,bytes=sum(x['bytes'] for x in hash_only)))
 save(DEST.parent/'NATIVE_REMOVALS.json.gz',dict(root=str(ROOT),files=removed,bytes=sum(x['bytes'] for x in removed),applied=False))
 save(DEST.parent/'PROTECTED_INPUTS.json',protected)
 print(json.dumps(dict(retained_files=len(kept),retained_bytes=sum(x['bytes'] for x in kept),native_files=len(removed),native_bytes=sum(x['bytes'] for x in removed),hash_only_files=len(hash_only))))

def clean(expected):
 inactive();protected=protect();assert protected==read(DEST.parent/'PROTECTED_INPUTS.json')
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==expected
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
 review=read(ROOT/'FINAL-REVIEW.json');assert review['source_revision']==expected and review['verdict']=='pass'
 for f in ('COMPACT_MANIFEST.json','NATIVE_REMOVALS.json.gz','PROTECTED_INPUTS.json','HASH_ONLY.json.gz'):
  subprocess.run(['git','ls-files','--error-unmatch',str(DEST.parent/f)],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
 assert not (DEST.parent/'CLEANUP.json').exists()
 manifest=read(DEST.parent/'COMPACT_MANIFEST.json');plan=read(DEST.parent/'NATIVE_REMOVALS.json.gz');assert not plan['applied']
 for r in manifest['files']:assert digest(DEST/r['path'])==r['sha256'],r['path']
 for r in plan['files']:
  p=Path(r['path']);assert p.resolve().is_relative_to(ROOT) and not p.is_symlink() and p.suffix in SUFFIXES
  assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],str(p)
 before=shutil.disk_usage(ROOT).free
 for r in plan['files']:Path(r['path']).unlink()
 assert protect()==protected
 save(DEST.parent/'CLEANUP.json',dict(applied=True,files=len(plan['files']),bytes=plan['bytes'],compact_commit=expected,plan_sha256=digest(DEST.parent/'NATIVE_REMOVALS.json.gz'),manifest_sha256=digest(DEST.parent/'COMPACT_MANIFEST.json'),all_removed=all(not Path(r['path']).exists() for r in plan['files']),protected_inputs_unchanged=True,prior_compact_unchanged=True,free_space_delta_bytes=shutil.disk_usage(ROOT).free-before))
 print((DEST.parent/'CLEANUP.json').read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['compact','clean']);p.add_argument('--expected-head');a=p.parse_args();compact() if a.mode=='compact' else clean(a.expected_head)
