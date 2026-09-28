"""Verify originals/controls, then submit actual bound requests to native contract-only route."""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,tempfile,time
import yaml
W=Path(__file__).resolve().parents[2];D=Path(__file__).parent
OLD=W/'validation/rtc_step_consequences_2026-09-28'
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928')
FAILED=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-retained-residuals-20260928/B-REF-rep1')
def read(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def binding(p):return dict(path=str(p),sha256=digest(p))
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n');return binding(p)
def main(binary,result):
 start=time.monotonic();deps=read(FAILED/'freeze/dependencies.json')['inputs']
 for q in deps:assert digest(Path(q['path']))==q['sha256'],q['path']
 manifests=read(OLD/'COMPACT_MANIFEST.json')['files'];known={q['source']:q['source_sha256'] for q in manifests};known.update({q['path']:q['sha256'] for q in json.loads(gzip.decompress((OLD/'HASH_ONLY.json.gz').read_bytes()))['files']});checked=[]
 for name in ('N14-REF-rep2','N14-C-core-plus','N14-C-core-minus','N14-C-shoulder'):
  base=ROOT/name;s=read(base/'run/BASE-POINT/REF/a1100/SUMMARY.json');assert s['ready'] and s['completed_cycle']==5
  paths=[base/'freeze/prospective.json',base/'freeze/measurement-geometry.json',base/'SHAPES.json',base/'run/BASE-POINT/REF/a1100/SUMMARY.json',base/'M0/products/a1100/map.fits']
  paths += [base/f'freeze/a1100/{x}.npz' for x in ('context','C0','BASE')]
  for k in range(1,6):paths += [base/f'run/BASE-POINT/REF/a1100/cycle-{k:02d}/native/products/total.fits',base/f'run/BASE-POINT/REF/a1100/cycle-{k:02d}/candidate.fits']
  paths += [base/'run/BASE-POINT/REF/a1100/initial/candidate.fits']
  for p in paths:
   rel=str(p.relative_to(ROOT));assert digest(p)==known[rel],rel;checked.append(binding(p))
  p=read(base/'freeze/prospective.json');assert (p['observation'],p['subobservation'],p['scan'],p['array'],p['networks'],p['feedback_count'],p['ptc']['rank'],p['experimental_method'])==(152430,0,2,'a1100',[1,2,3,4],5,10,'warm-observed-als')
  shapes=read(base/'SHAPES.json');refshapes=read(ROOT/'N14-REF-rep2/SHAPES.json')
  for branch in ('C-core-plus','C-core-minus','C-shoulder'):
   assert shapes['rasters'][branch]['sha256']==refshapes['rasters'][branch]['sha256']
   path=base/f'freeze/shapes/{branch}.fits';assert digest(path)==refshapes['rasters'][branch]['sha256'];checked.append(binding(path))
 rows=[]
 with tempfile.TemporaryDirectory(prefix='bound-fruit-contract-') as tmp:
  t=Path(tmp)
  for cycle in range(1,6):
   q=read(ROOT/f'N14-REF-rep2/run/BASE-POINT/REF/a1100/cycle-{cycle:02d}/request.json');owner=read(Path(q['owner_binding']['path']));p=read(Path(owner['prospective']['path']));p['campaign_scope']='rtc-retained-residuals-v1'
   owner['prospective']=write(t/f'prospective-{cycle}.json',p);q['owner_binding']=write(t/f'owner-{cycle}.json',owner)
   previous=read(Path(q['previous_state']['path']));previous['owner_sha256']=q['owner_binding']['sha256'];q['previous_state']=write(t/f'previous-{cycle}.json',previous)
   # Candidate, total, predecessor receipt, MAP owner/plan, full network receipts
   # and original per-network config stay exactly bound to the sealed control.
   req=write(t/f'request-{cycle}.json',q);out=t/f'validation-{cycle}';config=write(t/f'development-{cycle}.json',dict(schema='citlali-development-v1',terminal='point-fruit-validate',input=req,output=str(out)))
   run=subprocess.run([str(binary),config['path']],capture_output=True,text=True,timeout=60);assert run.returncode==0,run.stdout+run.stderr
   receipt=yaml.safe_load((out/'validation.yaml').read_text());assert receipt['outcome']=='request-contract-accepted' and receipt['cycle_index']==cycle and not receipt['scientific_processing_executed'] and not receipt['detector_arrays_loaded']
   rows.append(dict(cycle=cycle,request=req,accepted=True,receipt=receipt))
  # The exact originally rejected T1 request is accepted unchanged.
  original=FAILED/'run/BASE-POINT/REF/a1100/cycle-01/request.json';out=t/'failed-request-repaired';config=write(t/'failed-request.json',dict(schema='citlali-development-v1',terminal='point-fruit-validate',input=binding(original),output=str(out)))
  run=subprocess.run([str(binary),config['path']],capture_output=True,text=True,timeout=60);assert run.returncode==0,run.stdout+run.stderr
  original_receipt=yaml.safe_load((out/'validation.yaml').read_text())
 record=dict(pass_=True,seconds=time.monotonic()-start,original_dependencies_verified=len(deps),original_bindings=deps,control_bindings=checked,controls_compatible=True,compatibility='same original inputs/profile/rank/5cycles/measurement geometry/frozen rasters; full baseline numerical/discrete equivalence still required before Q',cycles=rows,original_failed_request=binding(original),original_failed_request_validation=original_receipt,scientific_processing_executed=False,limit='request/metadata admission only; actual numerical T1-T5 and realized network support remain Stage2 gates')
 result.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(pass_=True,dependencies=len(deps),controls=len(checked),cycles=len(rows),failed_request_repaired=True,seconds=record['seconds'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binary',required=True,type=Path);p.add_argument('--result',required=True,type=Path);a=p.parse_args();main(a.binary.resolve(),a.result)
