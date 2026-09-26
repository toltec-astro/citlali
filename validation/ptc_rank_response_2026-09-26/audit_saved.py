"""Audit saved native rank receipts and coverage without processing observations."""
import argparse,json,re,sys,gzip
from pathlib import Path
import numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/timestream_successor'))
from point_fruit_diagnostics import load
from compare_point_pipeline import available
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
rows=[];segments=[]
for rank in (5,10,15):
 for branch in ('REF','C-core-plus','C-core-minus','C-shoulder'):
  route=root/f'rank-{rank}/run/BASE-POINT/{branch}/a2000';s=json.loads((route/'SUMMARY.json').read_text());assert s['ready'] and s['completed_cycle']==10 and not s['m10_applied']
  for k in range(11):
   rel=f'cycle-{k:02d}/native/products/total.fits' if k else None
   actual=route/rel if k else root/f'rank-{rank}/M0/products/a2000/map.fits' if branch=='REF' else root/f'rank-{rank}/initial/{branch}/native/products/total.fits'
   ref=root/f'rank-10/run/BASE-POINT/REF/a2000/{rel}' if k else root/'rank-10/M0/products/a2000/map.fits'
   x,h=load(available(actual));y,g=load(available(ref))
   same={n:bool(np.array_equal(x[n],y[n],equal_nan=True)) for n in ('HITS','NORMALIZATION','SCI_SUPPORT','NORM_SUPPORT')};assert all(same.values())
   assert all(h[n]==g[n] for n in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2'))
   rows.append(dict(rank=rank,branch=branch,cycle=k,grid_and_coverage_equal_rank10_REF=True,science_pixels=int(x['SCI_SUPPORT'].sum()),hits=int(x['HITS'].sum())))
   if k or branch!='REF' or rank!=10:
    receipt=actual.parent/'receipt.yaml';receipt=receipt if receipt.exists() else Path(str(receipt)+'.gz')
    if not receipt.exists() and rank!=10 and branch=='REF' and k==0:
     receipt=root/f'rank-{rank}/M0/products/attempt.yaml';receipt=receipt if receipt.exists() else Path(str(receipt)+'.gz')
    r=yaml.safe_load(gzip.decompress(receipt.read_bytes()) if receipt.suffix=='.gz' else receipt.read_text())
    for z in r['segments']:
     assert z['rank']==z['realized_rank']==rank and z['learning_performed'] and z['converged'] and z['input_eligibility_changed']==0 and z['application_failed_times']==0
     assert z['projection_support_counts'][0]==z.get('injection_projection_support_counts',[0])[0]==0
     segments.append(dict(rank=rank,branch=branch,cycle=k,network=z['source_segment']['network'],rank_verified=True,fresh_learning=True,eligibility_unchanged=True))
errors=[]
for f in root.rglob('run.log'):
 for line in f.read_text().splitlines():
  if re.search(r'\[(error|critical)\]',line,re.I):errors.append(dict(path=str(f.relative_to(root)),line=line))
assert not errors
(root/'SAVED-AUDIT.json').write_text(json.dumps(dict(pass_=True,maps=rows,segments=segments,native_error_messages=errors,scope='all 132 maps; all 131 fresh residual/M0 native invocations; rank10 original M0 separately bound by its PTC receipts'),indent=2)+'\n')
print(json.dumps(dict(maps=len(rows),native_segments=len(segments),native_error_messages=len(errors))))
