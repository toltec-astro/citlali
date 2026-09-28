"""Exact bounded compact preservation, then committed-manifest native cleanup."""
import argparse,gzip,hashlib,json,shutil,subprocess
from pathlib import Path
import psutil
REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928')
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
 launches=list(ROOT.glob('*/LAUNCH.json'));assert len(launches) in (1,4)
 for p in launches:
  status=read(p.parent/'EXECUTION.json')['status'];assert status=='completed' or len(launches)==1 and p.parent.name=='N14-REF-rep1' and status=='source_rejected'
 if len(launches)==1 and read(launches[0].parent/'EXECUTION.json')['status']=='completed':assert read(launches[0].parent/'SHAPES.json')['available'] is False
 for p in psutil.process_iter(['pid','cmdline']):
  if p.pid==psutil.Process().pid:continue
  a=p.info['cmdline'] or []
  if not a:continue
  is_work=Path(a[0]).name=='citlali' or any(Path(x).name=='point_pipeline.py' for x in a[:3])
  assert not(is_work and any(str(ROOT) in x for x in a)),p.info

def protect():
 protected=read(ROOT/'PROTECTED_INPUTS.json')
 for r in protected:
  p=Path(r['path']);assert not p.resolve().is_relative_to(ROOT)
  assert digest(p)==r['sha256'],str(p)
 return protected

def redundant(rel):
 return False

def necessary_fits(rel):
 parts=rel.parts
 if 'freeze' in parts or 'endpoint' in parts or any(x.startswith('checkpoint-') for x in parts):return True
 if '/M0/products/' in '/'+str(rel):return rel.name=='map.fits'
 if 'initial' in parts:return rel.name in ('total.fits','response.fits','candidate.fits')
 if any(x in CHECK for x in parts):return rel.name in ('total.fits','response.fits','candidate.fits')
 return False

def compact():
 inactive();assert not DEST.exists();protected=protect()
 for name in ('N14-REF-rep1','N14-C-core-plus','N14-C-core-minus','N14-C-shoulder'):
  out=ROOT/name
  if not out.exists():continue
  if read(out/'EXECUTION.json')['status']=='source_rejected':continue
  summary=read(out/'run/BASE-POINT/REF/a1100/SUMMARY.json');assert len(summary['histories'])==6
 DEST.mkdir();kept=[];removed=[];hash_only=[]
 for p in sorted(ROOT.rglob('*')):
  assert not p.is_symlink(),str(p)
  if not p.is_file():continue
  rel=p.relative_to(ROOT);n=p.stat().st_size;h=digest(p)
  if p.suffix in SUFFIXES:
   assert rel.parts[0].startswith('N14-') and rel.parts[1] in ('upstream','M0','initial','run','native'),str(p)
   removed.append(dict(path=str(p),bytes=n,sha256=h));continue
  if redundant(rel) or (p.suffix=='.fits' and not necessary_fits(rel)):
   hash_only.append(dict(path=str(rel),bytes=n,sha256=h,reason='redundant regression-verified repeat or non-checkpoint product; all-cycle numerical summaries retained'));continue
  q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);encoding='identity'
  if p.suffix=='.fits' or p.suffix=='.yaml' and n>100000:
   raw=p.read_bytes();q=Path(str(q)+'.gz');q.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0));assert hashlib.sha256(gzip.decompress(q.read_bytes())).hexdigest()==h;encoding='gzip'
  else:shutil.copyfile(p,q);assert digest(q)==h
  kept.append(dict(source=str(rel),source_sha256=h,path=str(q.relative_to(DEST)),sha256=digest(q),bytes=q.stat().st_size,encoding=encoding))
 save(DEST.parent/'COMPACT_MANIFEST.json',dict(files=kept,bytes=sum(x['bytes'] for x in kept),old_reference_controls_referenced_in_place=True,scope='POINT152430 a1100 networks1–4 consequence event chains, bounded excerpts, exposure masks, source response, checkpoints and timing; intermediate FITS hash-only'))
 save(DEST.parent/'HASH_ONLY.json.gz',dict(files=hash_only,bytes=sum(x['bytes'] for x in hash_only)))
 save(DEST.parent/'NATIVE_REMOVALS.json.gz',dict(root=str(ROOT),files=removed,bytes=sum(x['bytes'] for x in removed),applied=False))
 save(DEST.parent/'PROTECTED_INPUTS.json',protected)
 print(json.dumps(dict(retained_files=len(kept),retained_bytes=sum(x['bytes'] for x in kept),native_files=len(removed),native_bytes=sum(x['bytes'] for x in removed),hash_only_files=len(hash_only))))

def clean(expected):
 inactive();protected=protect();assert protected==read(DEST.parent/'PROTECTED_INPUTS.json')
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==expected
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
 review=read(ROOT/'FINAL-REVIEW.json');assert review['source_revision']==expected and review['verdict']=='pass'
 assert read(DEST.parent/'REPORT_ISOLATION.json')['pass_']
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
 remaining=[p for p in ROOT.rglob('*') if p.is_file()]
 assert not any(p.suffix in SUFFIXES for p in remaining)
 save(DEST.parent/'CLEANUP.json',dict(applied=True,files=len(plan['files']),bytes=plan['bytes'],compact_commit=expected,plan_sha256=digest(DEST.parent/'NATIVE_REMOVALS.json.gz'),manifest_sha256=digest(DEST.parent/'COMPACT_MANIFEST.json'),all_removed=all(not Path(r['path']).exists() for r in plan['files']),protected_inputs_unchanged=True,prior_compact_unchanged=True,free_space_delta_bytes=shutil.disk_usage(ROOT).free-before,remaining_files=len(remaining),remaining_bytes=sum(p.stat().st_size for p in remaining),remaining_metadata_FITS_logs=True,exhaustive_remaining_native_verification_count=1))
 print((DEST.parent/'CLEANUP.json').read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['compact','clean']);p.add_argument('--expected-head');a=p.parse_args();compact() if a.mode=='compact' else clean(a.expected_head)
