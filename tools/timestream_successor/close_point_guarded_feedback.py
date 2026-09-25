"""Close the fixed guarded comparison without executing scientific operations."""
import argparse,json,re,shutil,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from run_point_guarded_feedback import authenticated,read,REPO,V,ARRAYS,POLICIES,BRANCHES
from run_point_response_cd import source
from point_fruit_trajectory import bound,manifest,verify_manifest
from inspect_forward_map import digest,require
from point_additive_diagnostics import save_json


def parallel_seal(root,workers=12):
    """Hash every payload once; reconcile every nested ledger against those bytes.

    This is the existing sorted SHA256 ledger format, with independent reads
    parallelized. Nested ledgers are assertions, never substitutes for hashing.
    """
    root=root.resolve();target=root/'FILES_SHA256SUMS'
    require(not target.exists(),'already sealed')
    start=time.perf_counter();files=sorted(f for f in root.rglob('*') if f.is_file())
    require(all(not f.is_symlink() and f.resolve().is_relative_to(root) for f in files),'external/symlink payload')
    with ThreadPoolExecutor(max_workers=workers) as pool:hashes=list(pool.map(digest,files))
    actual={f.resolve():sha for f,sha in zip(files,hashes)};nested=0
    for f in files:
        if f.name!='FILES_SHA256SUMS':continue
        for line in f.read_text().splitlines():
            sha,name=line.split('  ',1);p=(f.parent/name).resolve()
            require(p.is_relative_to(f.parent) and actual.get(p)==sha,'nested manifest mismatch: '+str(p))
            nested+=1
    payload=''.join(f'{sha}  {f.relative_to(root)}\n' for f,sha in zip(files,hashes))
    with target.open('x') as stream:stream.write(payload)
    require(target.read_text()==payload,'root ledger serialization mismatch')
    require(files==sorted(f for f in root.rglob('*') if f.is_file() and f!=target),'payload inventory changed during seal')
    return dict(root_manifest=bound(target),files=len(files),bytes=sum(f.stat().st_size for f in files),nested_entries_reconciled=nested,
        workers=workers,all_payloads_rehashed=True,wall_seconds=time.perf_counter()-start)


def census(root):
    p=authenticated(root);s=read(root/'report/SUMMARY.json');verify_manifest(root/'report')
    expected={(a,h,b) for a in ARRAYS for h in POLICIES for b in BRANCHES}
    require({(r['array'],r['policy'],r['branch']) for r in s['matrix']}==expected,'matrix identity')
    require(all(r['status']=='T10 complete' and r['cycle']==10 for r in s['matrix']),'incomplete trajectory')
    requests=list((root/'run').glob('*/*/*/cycle-*/request.json'));receipts=list((root/'run').glob('*/*/*/cycle-*/native/products/receipt.yaml'))
    ideals=list((root/'report').glob('*/*/*/ideal-measurement.json'));endpoints=list((root/'report').glob('*/*/endpoint/learned.json'))
    require(len(requests)==len(receipts)==450 and len(ideals)==27 and len(endpoints)==9,'fixed execution census')
    require(all(read(f)['measurement'] is not None for f in ideals),'unavailable ideal control')
    require(all(read(f.parent/'RESULTS.json')['status']=='available' for f in endpoints),'unavailable endpoint')
    return dict(consumed_originals=len(p['consumed_inputs']),consumed_originals_unchanged=True,trajectories=45,feedback_applications=450,
        ideal_controls=27,endpoint_fits=9,initial_PTC_MAP_calculations=0,fit_cache_note='Preserved publication attempts duplicate files, not scientific fit attempts. Identity caches reused without inference.',
        trajectory_authentication=bound(root/'report/TRAJECTORY_AUTHENTICATION.json'),report_manifest=bound(root/'report/FILES_SHA256SUMS'))


def inventory(root):
    rev=source();auth=census(root);dest=root/'provenance';logs=dest/'logs';logs.mkdir(exist_ok=True)
    for f in sorted(Path('/private/tmp').glob('point-guarded-*.log')):shutil.copy2(f,logs/f.name)
    gates={}
    checks=[('native','native-controls','7 tests'),('CLI','cli','Ran 20 tests'),('focused_ctest','ctest-corrected','100% tests passed'),
        ('baseline','baseline-final','207 passed'),('successor','successor-corrected','304 passed'),('frozen_fixture','frozen-corrected','3 passed'),
        ('configuration','config-final','config preflight reports written under'),('report','report-tests-recovery','8 passed'),('closure_and_reporting','closure-tests','10 passed')]
    for label,name,marker in checks:
        f=logs/f'point-guarded-{name}.log';require(marker in f.read_text(),'missing test evidence '+name);gates[label]=bound(f)
    error=re.compile(r'(?i)\[(?:error|critical)\]|^Traceback \(most recent call last\)|^\w*Error:|RuntimeWarning')
    paths=sorted((root/'run').glob('*/*/*/cycle-*/native/run.log'))
    require(len(paths)==450,'missing native logs')
    errors=[dict(path=str(f),line=line) for f in paths for line in f.read_text().splitlines() if error.search(line)]
    require(not errors,'unexpected native error log')
    save_json(dest/'VALIDATION.json',dict(gates=gates,native_logs_checked=len(paths),unexpected_native_errors=errors,
        limitations='Local cached-dependency environment, not Unity/Spack evidence. Focused9 CTest checks pass; unrelated unbuilt targets prevent a full CTest claim. Broad successor304+1skip and111subtests plus the separately corrected frozen fixture3 tests cover all305 required successor tests at the science revision. Later focused tests cover reporting/closure changes.',
        preserved_failures='Initial test interpreter/fixture setup failures and two report-publication failures are retained. Successful endpoint/ideal fit caches crossed publication retries byte-for-byte; zero repeated scientific fit attempts.'))
    s=read(root/'report/SUMMARY.json')
    revisions=dict(campaign=read(root/'freeze/prospective.json')['source_revision'],numerical_report=s['source_revision'],closure=rev,figures=read(root/'report/FIGURE_REVIEW.json')['source_revision'])
    for label,r in revisions.items():subprocess.run(['git','archive','--format=tar.gz','--output='+str(dest/f'source-{label}.tar.gz'),r],cwd=REPO,check=True)
    save_json(dest/'SOURCE.json',dict(revisions=revisions,native=bound(dest/'citlali'),build_qualification='Exact science binary built from campaign revision using preserved local cached dependencies. No Unity, dependency migration, RTC/CAL or JINC science execution.'))
    save_json(dest/'RUNTIME.json',dict(phases=s['phases'],summed_sequential_feedback_phase_wall_seconds=sum(x['wall_seconds'] for x in s['phases'].values()),
        native_cpu_seconds=sum(x['cpu_seconds'] for x in s['native_runtime']),native_output_bytes=sum(x['output_bytes'] for x in s['native_runtime']),
        report_phase_seconds=s['report_wall_seconds'],figure_phase=read(root/'report/FIGURE_REVIEW.json'),
        report_attempt1=read(dest/'REPORT_ATTEMPT1.json'),report_attempt2=read(dest/'REPORT_ATTEMPT2.json'),
        warning='Report phase excludes its final authentication/manifest; process real290.93s includes them. First attempt508.89s is filesystem-lifetime proxy, not stopwatch. Phase/native walls overlap; native CPU sum is not elapsed time. Full final sealing separately timed.'))
    for name in ('ASSESSMENT.md','REPRODUCTION.md'):shutil.copy2(V/name,root/name)
    (root/'README.md').write_text('# POINT 152389 guarded feedback comparison\n\nStart with ASSESSMENT.md. Numerical report and45-cell matrix: report/REPORT.md; full data: report/SUMMARY.json. See REPRODUCTION.md, FINAL_REVIEW.md and COMPLETION.json. No policy adoption or production change.\n')
    save_json(root/'FINAL_AUTHENTICATION.json',auth);manifest(dest)
    print(json.dumps(auth))


def seal(root,review):
    rev=source();require(review.is_file(),'missing independent review')
    auth=census(root);shutil.copy2(review,root/'FINAL_REVIEW.md')
    save_json(root/'FINAL_AUTHENTICATION.json',auth)
    save_json(root/'COMPLETION.json',dict(status='complete fixed45-trajectory comparison',source_revision=rev,sources=read(root/'provenance/SOURCE.json'),
        review=bound(root/'FINAL_REVIEW.md'),assessment=bound(root/'ASSESSMENT.md'),authentication=auth,
        stop='T10 consumes m9; m10 saved unapplied. No cycle11, new fields/calibration, adaptive uncertainty, RTC/CAL/JINC rerun, production default, Unity, merge or push.'))
    print(json.dumps(parallel_seal(root)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['inventory','seal']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--review',type=Path);args=ap.parse_args()
    inventory(args.output) if args.stage=='inventory' else seal(args.output,args.review)
