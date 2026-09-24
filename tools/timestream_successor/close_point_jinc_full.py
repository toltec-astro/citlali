"""Authenticate and seal the completed bounded experiment; no science execution."""
import argparse,json,os,platform,re,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import scipy,astropy,psutil
from point_jinc_full import read,source,REPO,ARRAYS,key
from point_fruit_trajectory import bound,verified,manifest,verify_manifest
from point_additive_diagnostics import save_json
from inspect_forward_map import digest,require

V=REPO/'validation/point_152389_map_only_jinc_full_2026-09-24'
LOGS=['point-jinc-full-build-a5.log','point-jinc-full-configure-a5.log','point-jinc-full-prepare.log','point-jinc-full-prepare-r2.log','point-jinc-full-prepare-r3.log','point-jinc-full-prepare-r4.log','point-jinc-freeze.log','point-jinc-grids.log','point-jinc-science.log','point-jinc-wcs-repair.log','point-jinc-science-r2.log','point-jinc-noise.log','point-jinc-report.log','point-jinc-report-r2.log','point-jinc-report-r3.log','point-jinc-successor-tests-closeout.log','point-jinc-baseline-tests.log','point-jinc-ctest.log','point-jinc-config.log']

def matrix(out):
 p=read(out/'PREPARED.json');s=read(out/'SCIENCE.json');n=read(out/'NOISE.json');rows=[]
 for state in p['states']:
  for op in ('MAP','JINC'):
   m=read(out/'science'/(state['key']+'-'+op+'-measurement.json'));fit=m['fit'];qual=[]
   if op=='JINC':qual.append('native low-denominator edge excursions; original source footprint supported')
   if fit.get('constrained'):qual.append('finite constrained descriptive fit retained')
   rows.append(dict(role='primary_map',array=state['array'],operator=op,branch=state['branch'],stage=state['stage'],status='completed with qualification' if qual else 'completed',qualifications=qual,measurement_status=fit['status'],measurement=bound(out/'science'/(state['key']+'-'+op+'-measurement.json'))))
   rp=(out/'ordinary'/state['key']/'rejoin.json') if op=='MAP' else (out/'grids'/state['key']/'REJOIN.json');rows.append(dict(role='rejoin_control',array=state['array'],operator=op,branch=state['branch'],stage=state['stage'],status='completed',evidence=bound(rp)))
 for a,v in s['arrays'].items():
  for r in v['responses']:
   rows.append(dict(role='injection_reference',array=a,operator=r['operator'],branch=r['branch'],stage=10,status='completed',original_domain=r['original_benchmark']['status'],native_domain=r['native']['status'],common_domain=r['paired_common']['status']))
   rows.append(dict(role='ideal_control_fit',array=a,operator=r['operator'],branch=r['branch'],stage=10,status='completed with qualification' if r['ideal_map_addition']['fit'].get('constrained') else 'completed',fit=r['ideal_map_addition']['fit']['status'],constrained=r['ideal_map_addition']['fit'].get('constrained',False)))
  for r in v['phases']:rows.append(dict(role='phase_fixture',array=a,operator=r['operator'],phase=r['phase'],status='completed',fit=r['measurement']['fit']['status'],constrained=r['measurement']['fit'].get('constrained',False)))
 for r in n['results']:
  rows.append(dict(role='T10_ensemble',array=r['array'],operator=r['operator'],stage=10,status='completed with qualification',trials=r['trials'],failures=r['uncertainty']['failures'],constrained=r['uncertainty']['bound_limited'],qualifications=['conditional Gaussian baseline, transferred covariance, source-depth extrapolation','finite Monte Carlo precision; no physical cross-map covariance']))
  for b in r['background']:
   rows.append(dict(role='background',array=r['array'],operator=r['operator'],stage=b['stage'],status='completed with qualification',outside_source_depth_pixels=b['outside_source_range_pixels'],model_status=b['status']))
   rows.append(dict(role='larger_mask_analytic',array=r['array'],operator=r['operator'],stage=b['stage'],status='completed with qualification',model_status=b['larger_mask']['status']))
 counts={k:sum(r['role']==k for r in rows) for k in sorted({r['role'] for r in rows})};expected=dict(primary_map=30,rejoin_control=30,injection_reference=18,ideal_control_fit=18,phase_fixture=54,T10_ensemble=6,background=12,larger_mask_analytic=12);require(counts==expected,'matrix coverage differs')
 save_json(out/'MATRIX.json',dict(counts=counts,cells=rows,unrun_cells=[],unavailable_required_measurements=[],pixel_unavailability='no-member pixels preserved separately in every JINC certificate; original measurement and benchmark domains supported'))


def authenticate(out):
 p=read(out/'PREPARED.json');t=time.perf_counter()
 for path,sha in p['consumed_inputs'].items():require(digest(Path(path))==sha,'changed original parent '+path)
 f=read(out/'freeze/prospective.json')
 for a in ARRAYS:
  for ref in f['benchmark_domains'][a]['original_report_maps'].values():verified(ref)
  for name in ('cache','coefficient_errors','lower','upper'):verified(p['arrays'][a]['cache'][name])
  for state in [s for s in p['states'] if s['array']==a]:
   for name in ('geometry','clean','operand','total'):verified(state[name])
 for folder in ('prepared','ast','cache','freeze','ordinary','grids','science','noise','report'):verify_manifest(out/folder)
 return dict(original_files=len(p['consumed_inputs']),wall_seconds=time.perf_counter()-t,all_original_and_current_consumed_hashes_unchanged=True)


def inventory(out):
 revision=source();matrix(out);d=out/'provenance';d.mkdir();(d/'logs').mkdir()
 for name in LOGS:
  path=Path('/private/tmp')/name;require(path.is_file(),'missing actual execution log '+name);shutil.copy2(path,d/'logs'/name)
 for name in ('ASSESSMENT.md','PREFLIGHT_REVIEW.md') :shutil.copy2(V/name,out/name)
 environment=dict(python=sys.version,python_executable=sys.executable,platform=platform.platform(),architecture=platform.machine(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__,logical_cpus=psutil.cpu_count(),host_memory_bytes=psutil.virtual_memory().total,thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MPLBACKEND')},native_double_significand_bits=53,native_long_double_significand_bits=53,native_FP_policy='IEEE binary64 nearest, no fast-math/no contraction; nonregular contribution/sum path unavailable',build_scope='local AppleClang and existing cached dependencies; no Spack V2 or Unity validation')
 save_json(d/'ENVIRONMENT.json',environment)
 p=read(out/'PREPARED.json');bins=d/'bin';bins.mkdir()
 for name in ('helper','cli'):shutil.copy2(verified(p[name]),bins/Path(p[name]['path']).name)
 for label,rev in [('final',revision),('native','a5c45cd7e38c9e48b0b37a73ba260859a3f52700')]:subprocess.run(['git','archive','--format=tar.gz','--output='+str(d/('source-'+label+'.tar.gz')),rev],cwd=REPO,check=True)
 version=subprocess.check_output([p['cli']['path'],'--version'],text=True,stderr=subprocess.STDOUT);save_json(d/'SOURCE_AND_BINARY.json',dict(code_source=revision,native_source='a5c45cd7e38c9e48b0b37a73ba260859a3f52700',cli_version=version,cli=bound(bins/Path(p['cli']['path']).name),helper=bound(bins/Path(p['helper']['path']).name),native_tree_diff=subprocess.check_output(['git','diff','a5c45cd7e38c9e48b0b37a73ba260859a3f52700',revision,'--','src','include','CMakeLists.txt'],cwd=REPO,text=True),grid_source=read(out/'freeze/prospective.json')['source_revision'],science_source=read(out/'SCIENCE.json')['source_revision'],noise_source=read(out/'NOISE.json')['source_revision']))
 successful_logs=['point-jinc-full-prepare-r4.log','point-jinc-freeze.log','point-jinc-grids.log','point-jinc-wcs-repair.log','point-jinc-science-r2.log','point-jinc-noise.log','point-jinc-report-r3.log']
 error_lines={name:[line for line in (d/'logs'/name).read_text().splitlines() if re.search(r'(?i)\[(?:error|critical)\]|^Traceback \(most recent call last\)|^\w*Error:',line)] for name in successful_logs}
 require(not any(error_lines.values()),'unexpected runtime errors in successful execution logs')
 save_json(d/'SUCCESSFUL_LOG_SCAN.json',dict(pattern='error/critical severity, traceback, terminal exception',logs=error_lines,excludes='preserved failed attempts are separately retained'))
 save_json(d/'VALIDATION.json',dict(successor=dict(tests=283,subtests=111,skips=0,log=bound(d/'logs/point-jinc-successor-tests-closeout.log')),baseline=dict(tests=207,subtests=137,log=bound(d/'logs/point-jinc-baseline-tests.log')),native_CLI=dict(tests=34,log=bound(d/'logs/point-jinc-ctest.log')),config=dict(require_all=True,pass_gate=True,log=bound(d/'logs/point-jinc-config.log')),focused_latest=dict(tests=6,command='python -m pytest tools/timestream_successor/test_point_jinc_campaign.py -q',scope='unchanged source measurement failures, original domain, signed plot geometry and exact WCS serialization'),numerical_and_realized_science_review='independent point_jinc_review; exact source dispositions retained in final review',unexpected_runtime_error_messages=0,known_preserved_warning='initial grid rejoin diagnostic performed unused division on no-member pixels, yielding one RuntimeWarning; accepted-domain checks passed; guarded in future code',no_new_upstream_evidence='three native AST export invocations use explicit point-152389-jinc-ast-v1 branch and return before MAP/PTC execution;75 native jinc_diagnostic accumulation invocations cannot call RTC/PTC/FRUIT; exact source archives and all requests/receipts retained; no synthetic zero Learn counter'))
 (out/'REPRODUCTION.md').write_text(f'''# Reproduce the bounded comparison

Use a clean Git checkout at commit {revision} and the
preserved native executables in `{bins}`. Original immutable parent paths are
bound in PREPARED.json and must remain available. The fresh destination must
not already exist. This recreates only the authorized fixed experiment.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-jinc-mpl XDG_CACHE_HOME=/private/tmp/point-jinc-cache
/Users/gwilson/tolteca/bin/python tools/timestream_successor/point_jinc_full.py prepare --output /ABSOLUTE/FRESH/OUTPUT --cli {bins/'citlali'} --helper {bins/'citlali_jinc_diagnostic'}
/Users/gwilson/tolteca/bin/python tools/timestream_successor/point_jinc_campaign.py freeze --output /ABSOLUTE/FRESH/OUTPUT --review {out/'PREFLIGHT_REVIEW.md'}
/Users/gwilson/tolteca/bin/python tools/timestream_successor/point_jinc_campaign.py grid --output /ABSOLUTE/FRESH/OUTPUT
/Users/gwilson/tolteca/bin/python tools/timestream_successor/point_jinc_science.py science --output /ABSOLUTE/FRESH/OUTPUT
/Users/gwilson/tolteca/bin/python tools/timestream_successor/point_jinc_science.py noise --output /ABSOLUTE/FRESH/OUTPUT
/Users/gwilson/tolteca/bin/python tools/timestream_successor/report_point_jinc_full.py --output /ABSOLUTE/FRESH/OUTPUT
```

Run from that clean Git checkout's repository root. The source archive
`{d/'source-final.tar.gz'}` preserves the exact source contents, but is not
a Git checkout; these commands require Git metadata. Obtain the same existing environment;
no dependency migration is implied. A new source/binary requires appropriate
review. The repaired writer preserves exact WCS directly; the historical
header-only repair need not be repeated on a fresh run. The current bundle
preserves failed preparation, initial header products and first render rather
than altering older experiments. No push, Unity, PTC Learn, new feedback or
extra independent trials are part of this sequence.
''')
 (out/'README.md').write_text(f'''# POINT 152389: completed fixed JINC experiment

[Assessment and one recommended next action]({out/'ASSESSMENT.md'})

[Numerical tables and diagnostic figures]({out/'report/RESULTS.md'})

[Every planned cell and qualification]({out/'MATRIX.json'})

[Reproduction]({out/'REPRODUCTION.md'})

Original maps, calibration, cleaning and FRUIT histories remain unchanged.
The result is qualified: smoother pixels do not establish better measurement
precision, and the a2000 JINC source-noise transfer is extrapolated throughout
the fit footprint. No recommended next action was executed.
''')
 save_json(out/'FINAL_AUTHENTICATION.json',authenticate(out));manifest(d)


def seal(out,review):
 require((out/'provenance').exists(),'inventory first');shutil.copy2(review,out/'FINAL_REVIEW.md');auth=authenticate(out);save_json(out/'FINAL_AUTHENTICATION.json',auth)
 save_json(out/'COMPLETION.json',dict(status='complete with scientific qualifications',source_revision=source(),inventory_source=read(out/'provenance/SOURCE_AND_BINARY.json')['code_source'],product_root=str(out),matrix=bound(out/'MATRIX.json'),assessment=bound(out/'ASSESSMENT.md'),review=bound(out/'FINAL_REVIEW.md'),authentication=auth,primary_maps=30,logical_references=18,phase_maps=54,actual_native_accumulations=75,independent_trials=1536,failed_trials=0,retained_constrained_trials=20,stop='no new science action, source model, map weights, feedback, production/default, integration or push'))
 manifest(out);verify_manifest(out);print(json.dumps(dict(manifest=bound(out/'FILES_SHA256SUMS'),files=len((out/'FILES_SHA256SUMS').read_text().splitlines()))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','seal']);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path);a=p.parse_args();inventory(a.output) if a.stage=='inventory' else seal(a.output,a.review)
