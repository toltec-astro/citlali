"""Only the four frozen Q roles; existing native pipeline performs all science."""
from pathlib import Path
import argparse,json,sys,shutil
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_pipeline_inputs import prepare
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound,verified
from point_step_experiment import consume_retained
from point_speed_experiment import launch
from point_storage import tree_bytes
from inspect_forward_map import require,digest
ROOT=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-two-stage-20260928')
PRIOR=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-retained-residuals-20260928')
D=Path(__file__).parent
ROLES=('REF','C-core-plus','C-core-minus','C-shoulder')
MANIFEST_SHA='a8edbfecdc297e06fd8625fbd1b435ff1b43bae60de852043cac0fcda995f1c5'
def read(p):return json.loads(p.read_text())
def preflight(branch):
 require(branch in ROLES,'foreign Q role')
 require(read(ROOT/'B-REF-rep1/EXECUTION.json')['status']=='completed','B incomplete')
 for name in ('REGRESSION','NATIVE_REGRESSION'):require(read(ROOT/(name+'.json'))['pass_'],'full B gate required')
 require(digest(D/'QUARANTINE.json')==MANIFEST_SHA,'frozen candidate changed')
 manifest=read(D/'QUARANTINE.json')
 for b in [manifest['baseline'],*manifest['full_baseline_gate'],*manifest['selection_evidence']]:verified(b)
 for role in ROLES[:ROLES.index(branch)]:require(read(ROOT/('Q-'+role)/'EXECUTION.json')['status']=='completed','earlier Q role incomplete; no continuation')
 for row in read(ROOT/'PROTECTED_INPUTS.json'):require(digest(Path(row['path']))==row['sha256'],'protected input changed')
 require(not (ROOT/('Q-'+branch)).exists(),'no implicit Q resume or replacement')
 return manifest

def prepare_q(branch):
 preflight(branch);out=ROOT/('Q-'+branch);B=ROOT/'B-REF-rep1';prepare(out,REPO/'validation/rtc_step_consequences_2026-09-28/inputs/reference.json',Path('/Users/gwilson/work_toltec/local_data'))
 p=read(out/'freeze/prospective.json');p.update(schema='citlali-point-speed-pca-v1',profile='POINT-SPEED-PCA-v1',campaign_scope='rtc-retained-residuals-v1',rtc_family='Q',signal_branch=branch,cycles=5,feedback_count=5,experimental_method='warm-observed-als',trajectory_identity=out.name)
 p['scientific_recipe']['PTC']['method']='observed-entry ALS; own preceding converged basis initializer; fully resolved current residual'
 for name in ('freeze/measurement-geometry.json','SHAPES.json'):shutil.copyfile(B/name,out/name)
 shutil.copytree(B/'freeze/shapes',out/'freeze/shapes')
 p['experiment_operands']=[bound(D/'QUARANTINE.json'),bound(out/'freeze/measurement-geometry.json'),bound(out/'SHAPES.json')]+[bound(x) for x in sorted((out/'freeze/shapes').iterdir())]
 if branch!='REF':
  ref=ROOT/'Q-REF';require(read(ref/'EXECUTION.json')['status']=='completed','own Q REF required')
  require(read(ref/'freeze/prospective.json')['rtc_family']=='Q','foreign Q noise')
  p['noise_reference']=dict(root=str(ref),binding=bound(ref/'freeze/prospective.json'),files=[bound(x) for x in sorted((ref/'freeze/a1100').iterdir()) if x.is_file()])
 obs=read(out/'inputs/observation.json')
 for e in obs['networks']:
  path=Path(e['input']['path']);cfg=read(path)
  cfg['rtc_whole_run_quarantine']=bound(D/'QUARANTINE.json')
  cfg['rtc_source_probe']=dict(schema='point-rtc-source-probe-v1',authority='owner-RTC-step-consequences-2026-09-28',branch=branch)
  if branch!='REF':cfg['rtc_source_probe']['raster']=bound(out/f'freeze/shapes/{branch}.fits')
  save_json(path,cfg);e['input']=bound(path)
 save_json(out/'inputs/observation.json',obs);save_json(out/'freeze/generated-inputs.json',[bound(x) for x in sorted((out/'inputs').iterdir()) if x.is_file()]);p['generated_inputs']=bound(out/'freeze/generated-inputs.json')
 budget=read(out/'freeze/budget.json');retained=tree_bytes(ROOT)-tree_bytes(out)+tree_bytes(PRIOR);single=304*budget['native_cells']+2_000_000_000
 require(retained+single<=30_000_000_000,'shared storage reservation ceiling')
 require(shutil.disk_usage(ROOT).free>single+budget['free_space_reserve_bytes'],'free space reserve')
 budget.update(campaign_root=str(ROOT),campaign_retained_bytes=retained,peak_ceiling_bytes=30_000_000_000,metadata_reserve_bytes=100_000_000,maximum_branch_workers=1,single_trajectory_reservation_bytes=single)
 save_json(out/'freeze/budget.json',budget);p['budget']=bound(out/'freeze/budget.json');save_json(out/'freeze/prospective.json',p)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('branch',choices=ROLES);p.add_argument('--review',required=True,type=Path);a=p.parse_args();out=prepare_q(a.branch)
 r=launch(ROOT,out,REPO/'build/bin/citlali',a.review,'unflushed local cache; fixed frozen Q membership and original inputs; no timing-repeat claim',attempt_consumer=consume_retained,child_mode='rtc-once')
 print(json.dumps(dict(run=out.name,status=r['status'],seconds=r['exit_seconds'])))
 require(r['status']=='completed','Q failed: stop; no automatic retry or slot substitution')
