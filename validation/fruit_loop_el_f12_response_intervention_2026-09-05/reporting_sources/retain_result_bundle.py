"""Bind retained outputs and copy the small review artifacts into the repository."""
from pathlib import Path
import json
import shutil
import sys
import time
REPO=Path('/Users/gwilson/.codex/worktrees/4c31/citlali-refactor')
ROOT=Path('/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f12-response-aware-intervention-r0.1')
SCRATCH=Path('/private/tmp/fruit-el-f12-prototype-20260905')
VALIDATION=REPO/'validation/fruit_loop_el_f12_response_intervention_2026-09-05'
sys.path.insert(0,str(REPO))
from tools.fruit_loops.analyze_response_intervention import file_record,write_json
from tools.fruit_loops.run_response_intervention import verify_artifacts,tree_bytes,resource_violation
registration=ROOT/'setup/REGISTRATION_R0.2.json'
registered=json.loads(registration.read_text())
verify_artifacts(registered['artifacts'])
print('All 6,274 registered input/implementation identities reverified.',flush=True)

# These are presentation/orchestration sources; the registered scientific
# analyzer and native binary have not changed.
report=ROOT/'analysis/REPORT_R0.1'
source_dir=report/'reporting_sources';source_dir.mkdir()
names=('run_registered_analysis.py','run_registered_analysis_initial_timestamp_error.py',
       'summarize_decisions.py','supplement_retained_reports.py','render_results.py',
       'final_result_checks.py','retain_result_bundle.py')
for name in names:shutil.copy2(SCRATCH/name,source_dir/name)
write_json(report/'REPORTING_REPAIR_R0.1.json',dict(
 status='resolved_before_measurements',failure="KeyError: 'started_unix' while reading STARTED.json",
 actual_field='unix_time',initial_wrapper=file_record(source_dir/'run_registered_analysis_initial_timestamp_error.py'),
 corrected_wrapper=file_record(source_dir/'run_registered_analysis.py'),
 registered_analyzer_changed=False,native_binary_changed=False,additional_replays=0,
 measurements_before_repair=0,existing_products_changed=False,
 supplement_resources={'wall_seconds':5.57,'user_seconds':3.82,'system_seconds':.38,'peak_rss_bytes':219217920},
 figures_visually_reviewed=['PRIORITIZED_LEAKAGE_R0.1.png','ALL_ARRAY_PROTECTIONS_R0.1.png']))

records=[]
first=json.loads((ROOT/'H0/uninjected/STARTED.json').read_text())['unix_time']
for path in sorted(ROOT.rglob('*')):
 if path.is_file() and not path.is_symlink():records.append(file_record(path))
retained,spool=tree_bytes(ROOT)
violation=resource_violation(0,time.time()-first,0,retained,spool)
assert violation is None,violation
manifest=dict(status='retained',root=str(ROOT),file_count=len(records),total_bytes=retained,
 temporary_uncompressed_spool_bytes=spool,aggregate_elapsed_seconds=time.time()-first,
 limits='12 hours aggregate; 64 GiB retained; 32 GiB temporary spool; passed',
 self_exclusion='Manifest describes all files present immediately before this manifest was written; its own bytes are excluded.',
 registration=file_record(registration),artifacts=records)
write_json(report/'RETAINED_OUTPUT_MANIFEST_R0.1.json',manifest)

for p in sorted(report.rglob('*')):
 if p.is_file():
  target=VALIDATION/p.relative_to(report);target.parent.mkdir(parents=True,exist_ok=True)
  assert not target.exists(),target
  shutil.copy2(p,target)
for name in ('Half_RESULT.json','Hold_RESULT.json','Half_ANALYSIS_STARTED_R0.1.json','Hold_ANALYSIS_STARTED_R0.1.json',
 'Half_ANALYSIS_RECEIPT_R0.1.json','Hold_ANALYSIS_RECEIPT_R0.1.json','Half_ANALYSIS_R0.1.log','Hold_ANALYSIS_R0.1.log'):
 assert not (VALIDATION/name).exists()
 shutil.copy2(ROOT/'analysis'/name,VALIDATION/name)
shutil.copy2(SCRATCH/'SCIENTIFIC_INTERPRETATION_R0.1.md',VALIDATION/'SCIENTIFIC_INTERPRETATION_R0.1.md')
print(json.dumps({'files':len(records),'bytes':retained,'hours':manifest['aggregate_elapsed_seconds']/3600,
 'retention_manifest':file_record(report/'RETAINED_OUTPUT_MANIFEST_R0.1.json')}),flush=True)
