"""Authenticate, preserve and seal the bounded POINT expected-source experiment."""
import argparse, json, re, shutil, subprocess, time
from pathlib import Path
from run_point_expected import inputs, read, ARRAYS, POLICIES, REPO, V
from run_point_corr_admission import source
from point_fruit_trajectory import bound, manifest, verify_manifest
from point_additive_diagnostics import save_json
from inspect_forward_map import require


def authenticate(out):
    start=time.perf_counter();p=inputs(out)
    stages={s:verify_manifest(out/s) for s in ('freeze','checks','calibration','validation','screen','report')}
    checks=read(out/'checks/RESULTS.json');screens=read(out/'validation/screens.json')['policies']
    require(len(checks['cases'])==18 and len(checks['bright'])==15,'wrong bounded controls')
    for phase in ('calibration','validation'):
        for a in ARRAYS:
            rows=read(out/phase/f'{a}-trials.json')
            require(len(rows)==1024 and [r['trial'] for r in rows]==list(range(1024)),'missing/replaced field')
            require(all(r['maxima_known'][pol] for r in rows for pol in POLICIES),'unavailable completed campaign')
            require(sum('stress' in r for r in rows)==(256 if phase=='validation' else 0),'wrong stress count')
    require(all(screens[a][pol]['comparison_usable'] for a in ARRAYS for pol in POLICIES),'realized screen disposition changed')
    require(not (out/'run').exists(),'unexpected feedback execution')
    return dict(consumed_originals=len(p['consumed_inputs']),consumed_originals_unchanged=True,stage_manifest_files=stages,
                wall_seconds=time.perf_counter()-start,independent_new_fields=0,feedback_applications=0)


def inventory(out):
    rev=source();auth=authenticate(out);dest=out/'provenance';dest.mkdir();logs=dest/'logs';logs.mkdir()
    for f in sorted(Path('/private/tmp').glob('point-expected-*.log')):shutil.copy2(f,logs/f.name)
    gates={}
    for label,name,marker in [('focused','focused-r3','17 passed'),('successor','successor-r2','300 passed, 111 subtests passed'),
                             ('baseline','baseline','207 passed, 137 subtests passed'),('available_native_CLI','ctest-built','100% tests passed, 0 tests failed out of 38'),
                             ('configuration','config','config preflight reports written under')]:
        f=logs/f'point-expected-{name}.log';require(marker in f.read_text(),'missing successful check '+name);gates[label]=bound(f)
    pattern=r'(?i)\[(?:error|critical)\]|^Traceback \(most recent call last\)|^\w*Error:|RuntimeWarning'
    scans={name:[s for s in (logs/f'point-expected-{name}.log').read_text().splitlines() if re.search(pattern,s)] for name in ('prepare-r2','calibration','validation','screen','report')}
    require(not any(scans.values()),'unexpected accepted-run error/warning')
    save_json(dest/'VALIDATION.json',dict(gates=gates,successful_log_scan=scans,
        limitations='Broad initial CTest could not run unrelated unbuilt _NOT_BUILT targets. It is retained, not counted as a full pass. All38 available frozen-native tests pass; no native rebuild. Python/config checks pass. Local cached environment is supplemental, not Unity/Spack replication.',
        attempts='First prepared bundle preserved at sibling without -r2; reviewed publication/authentication repairs precede calibration. Early test logs retained. No calibration or validation tuning/retry.'))
    p=read(out/'freeze/prospective.json');revisions=dict(campaign=p['source_revision'],report=read(out/'report/REPORT.json')['source_revision'],closure=rev)
    for label,r in revisions.items():subprocess.run(['git','archive','--format=tar.gz','--output='+str(dest/f'source-{label}.tar.gz'),r],cwd=REPO,check=True)
    native=Path('/private/tmp/citlali-point-additive-build-20260923/bin')
    binaries={n:bound(native/n) for n in ('citlali','citlali_point_frozen_apply','citlali_jinc_diagnostic')}
    diff=subprocess.check_output(['git','diff','cfcde23f4f6cd27588924c748c2fec07d8f3007e',rev,'--','src','include','CMakeLists.txt'],cwd=REPO,text=True)
    require(not diff,'unexpected native change')
    save_json(dest/'SOURCE.json',dict(revisions=revisions,regression_binaries=binaries,native_source='ee54efb6d32759003d4249f3a4ad90dbeb9c10d9',
        scientific_native_invocations=0,native_diff_from_work_order_base=diff,source_archive_note='Archives require Git metadata for clean-source execution checks; exact campaign/report sources remain distinct.'))
    first=out.with_name(out.name.removesuffix('-r2'))
    save_json(dest/'FIRST_PREPARATION.json',dict(root=str(first),preserved_unmodified=True,files=[bound(f) for f in sorted(first.rglob('*')) if f.is_file()]))
    shutil.copy2(out/'report/ASSESSMENT.md',out/'ASSESSMENT.md');shutil.copy2(V/'REPRODUCTION.md',out/'REPRODUCTION.md')
    save_json(dest/'RUNTIME.json',dict(checks=read(out/'PREPARED.json'),calibration=read(out/'CALIBRATION.json'),validation=read(out/'VALIDATION.json'),report=read(out/'report/REPORT.json'),
        saved_map_publication='not timed separately; zero fits',caution='Phase times overlap; worker CPU sums and peak RSS are not total campaign wall/memory.'))
    (out/'README.md').write_text('# POINT 152389 expected-source guard\n\nSee ASSESSMENT.md, report/*-saved-candidates.png, report/amplitude-uncertainty.png, report/SUMMARY.json and REPRODUCTION.md.\n\nAll six replayed screens pass. Candidates remain unapplied. No new feedback or production default.\n')
    save_json(out/'FINAL_AUTHENTICATION.json',auth);manifest(dest)
    print(json.dumps(auth))


def seal(out,review):
    require(not (out/'FILES_SHA256SUMS').exists(),'already sealed');verify_manifest(out/'provenance');auth=authenticate(out)
    require(review.is_file(),'missing final independent review');shutil.copy2(review,out/'FINAL_REVIEW.md')
    save_json(out/'FINAL_AUTHENTICATION.json',auth)
    save_json(out/'COMPLETION.json',dict(status='complete bounded expected-source guard; six replayed screens pass',source_revision=source(),
        sources=read(out/'provenance/SOURCE.json'),review=bound(out/'FINAL_REVIEW.md'),assessment=bound(out/'ASSESSMENT.md'),authentication=auth,
        calibration_reused_fields=3072,validation_reused_fields=3072,stress_reuses=768,independent_new_fields=0,feedback_applications=0,
        stop='No trajectories, endpoint uncertainty, RTC/CAL/PTC/MAP/JINC rerun, m10 application, production default, Unity, merge or push.'))
    manifest(out);print(json.dumps(dict(root_manifest=bound(out/'FILES_SHA256SUMS'),files=verify_manifest(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['inventory','seal']);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path);a=p.parse_args()
    inventory(a.output) if a.stage=='inventory' else seal(a.output,a.review)
