"""Compare the one new B replay to the immutable completed B control."""
from pathlib import Path
import json,gzip,sys
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(W/'tools/timestream_successor'))
from inspect_forward_map import digest,require
from point_additive_diagnostics import save_json
from compare_point_pipeline import compare
from point_rtc_diagnostics import yread
D=W/'validation/rtc_step_consequences_2026-09-28'
OLD=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928')
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-retained-residuals-20260928');out=R/'B-REF-rep1';ref=OLD/'N14-REF-rep2'
read=lambda p:json.loads(p.read_text())
# Authenticate retained old products against tracked sealed old records first.
oldhash=json.loads(gzip.decompress((D/'HASH_ONLY.json.gz').read_bytes()))
print(type(oldhash).__name__,flush=True)
if isinstance(oldhash,dict): oldhash=oldhash['files']
checks=[]
for q in oldhash:
 if q['path'].startswith(ref.name+'/'):
  require(digest(OLD/q['path'])==q['sha256'],'sealed previous product changed: '+q['path'])
for p in (D/'compact/N14-REF-rep2').rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(D/'compact/N14-REF-rep2');original=ref/rel
 if p.suffix=='.gz':
  original=ref/str(rel)[:-3]
  if original.exists():require(original.read_bytes()==gzip.decompress(p.read_bytes()),'compact original mismatch '+str(rel))
 elif original.exists():require(digest(original)==digest(p),'compact original mismatch '+str(rel))
# Failed T1 route: only completed upstream/M0 can be compared. Never call this
# full M0-T5 equivalence or infer an endpoint result from a partial trajectory.
from astropy.io import fits
import numpy as np
planes=[]
for rel in ('M0/products/a1100/map.fits','run/BASE-POINT/REF/a1100/initial/candidate.fits'):
 with fits.open(out/rel) as a,fits.open(ref/rel) as b:
  require([h.name for h in a]==[h.name for h in b],'different plane inventory')
  for x,z in zip(a,b):
   equal=bool(np.array_equal(x.data,z.data,equal_nan=True));planes.append(dict(path=rel,plane=x.name,exact=equal))
for name in ('context','C0','BASE'):
 with np.load(out/f'freeze/a1100/{name}.npz') as a,np.load(ref/f'freeze/a1100/{name}.npz') as b:
  require(set(a.files)==set(b.files),'different frozen field inventory')
  for key in a.files:planes.append(dict(path=name,field=key,exact=bool(np.array_equal(a[key],b[key],equal_nan=True))))
a=read(out/'run/BASE-POINT/REF/a1100/initial/state.json');b=read(ref/'run/BASE-POINT/REF/a1100/initial/state.json')
# Compare scientific record only; paths/parent publication identity are run-local.
for key in ('measurement','admission'):
 planes.append(dict(path='initial/state.json',field=key,exact=a[key]==b[key]))
result=dict(completed_M0_exact=all(q['exact'] for q in planes),planes=planes,full_M0_T5_equivalence='unavailable: T1 native request rejected',endpoint='not executed',candidate_blocked=True)
save_json(R/'REGRESSION.json',result)
plan=json.loads(gzip.decompress((D/'NATIVE_REMOVALS.json.gz').read_bytes()))['files']
prefix=str(ref)+'/'
science={q['path'][len(prefix):]:q['sha256'] for q in plan if q['path'].startswith(prefix) and q['path'][len(prefix):].startswith(('upstream/','M0/'))}
for rel,expected in science.items():
 p=out/rel;checks.append(dict(path=rel,exact=p.exists() and digest(p)==expected))
for nw in (1,2,3,4):
 base=Path(f'upstream/products/network{nw}')
 for name in ('events.yaml','event-decisions.yaml','support-decisions.yaml'):
  checks.append(dict(path=str(base/name),exact=(out/base/name).read_bytes()==(ref/base/name).read_bytes()))
 a=yread(out/base/'receipt.yaml');b=yread(ref/base/'receipt.yaml')
 for key in ('candidate_count','event_count','admitted_jump_groups','transient_excluded_pair_cells'):
  checks.append(dict(network=nw,field=key,exact=a[key]==b[key]))
 for key in ('realized_detectors','speed_support_comparison','speed_support_authority'):
  checks.append(dict(network=nw,field=key,exact=a['donor_continuity'][key]==b['donor_continuity'][key]))
 a=yread(out/base/'jump-exclusions.yaml');b=yread(ref/base/'jump-exclusions.yaml')
 for v in (a,b):v.pop('processing_generation',None)
 checks.append(dict(network=nw,field='jump excluding generation',exact=a==b))
r=dict(pass_=all(q['exact'] for q in checks),checks=checks,science_hashes=len(science),prior_manifest_sha256=digest(D/'NATIVE_REMOVALS.json.gz'))
save_json(R/'NATIVE_REGRESSION.json',r)
print(json.dumps(dict(M0_exact=result['completed_M0_exact'],native=r['pass_'],hashes=len(science),failures=[q for q in checks if not q['exact']][:30])),flush=True)
require(result['completed_M0_exact'] and r['pass_'],'partial B products differ; retain failed gate')
