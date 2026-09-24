"""Preserve and seal the failed-gate POINT experiment without new science work."""
import argparse, json, os, platform, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import scipy, astropy, psutil
from run_point_corr_admission import read, source, REPO, ARRAYS
from point_fruit_trajectory import bound, verified, manifest, verify_manifest
from point_additive_diagnostics import save_json
from inspect_forward_map import digest, require, read_yaml

V = REPO/'validation/point_152389_correlation_admission_2026-09-24'


def authenticate(out):
    start = time.perf_counter()
    p = read(out/'freeze/prospective.json')
    for name, sha in p['consumed_inputs'].items():
        require(digest(Path(name)) == sha, 'changed consumed original: '+name)
    folders = ['freeze', 'calibration', 'calibration-disposition', 'validation', 'screen', 'report']
    folders += ['run/BASE/REF/'+a for a in ARRAYS]
    counts = {name: verify_manifest(out/name) for name in folders}
    for a in ARRAYS:
        s = read(out/'run/BASE/REF'/a/'SUMMARY.json')
        require(s['ready'] and s['completed_cycle'] == 10 and not s['m10_applied'], 'incomplete BASE replay')
        require(len(s['histories']) == 11, 'wrong BASE history count')
        for h in s['histories'][1:]:
            require(h['ready'] and h['upstream_replay']['all_numerical_files_exact'], 'BASE exact replay failed')
    gates = read(out/'validation/gates.json')['policies']
    require(all(not gates[a][h]['pass'] and gates[a][h]['threshold'] is None for a in ARRAYS for h in ('CORR', 'DIAG')), 'failed-gate disposition changed')
    require(not (out/'run/CORR').exists() and not (out/'run/DIAG').exists(), 'unexpected new-policy execution')
    return dict(original_files=len(p['consumed_inputs']), original_hashes_unchanged=True,
                stage_manifest_files=counts, wall_seconds=time.perf_counter()-start)


def runtime(out):
    rows = []
    for a in ARRAYS:
        for k in range(1, 11):
            d = out/'run/BASE/REF'/a/f'cycle-{k:02d}'
            r = read_yaml(d/'native/products/receipt.yaml')
            run = read(d/'native/RUN.json')
            require(run['exit_code'] == 0 and r['outcome'] == 'success', 'failed native invocation')
            row = dict(array=a, cycle=k, receipt=bound(d/'native/products/receipt.yaml'),
                       run=bound(d/'native/RUN.json'), invocation_wall_seconds=run['wall_seconds'],
                       sampled_peak_RSS_bytes=run['polled_peak_rss_bytes'],
                       sampled_max_OS_threads=max(s['os_threads'] for s in run['samples']),
                       diagnostic_seconds=read(d/'cycle.json')['diagnostic_seconds'],
                       segment_count=len(r['segments']))
            for key in ('cpu_seconds', 'wall_seconds', 'MAP_seconds', 'serialization_seconds', 'verification_seconds'):
                row['native_'+key] = r[key]
            for key in ('preparation_seconds', 'learning_seconds', 'application_seconds', 'loading_projection_serialization_seconds'):
                row['segment_sum_'+key] = sum(s[key] for s in r['segments'])
            rows.append(row)
    sums = {key: sum(r[key] for r in rows) for key in rows[0] if key.endswith('_seconds')}
    return dict(rows=rows, summed_seconds=sums, native_invocations=len(rows),
                BASE_execution=read(out/'FEEDBACK-BASE.json'),
                simulation_calibration=read(out/'CALIBRATION.json'), simulation_validation=read(out/'VALIDATION.json'),
                report_wall_seconds=read(out/'report/REPORT.json')['wall_seconds'],
                caution='Phase totals overlap and must not be added; summed process times are not campaign wall time. OS thread observations do not identify BLAS worker counts. No new-policy Learn or Apply occurred.')


def inventory(out):
    revision = source()
    auth = authenticate(out)
    dest = out/'provenance'
    dest.mkdir()
    logs = dest/'logs'
    logs.mkdir()
    for path in sorted(Path('/private/tmp').glob('point-corr-*.log')):
        shutil.copy2(path, logs/path.name)
    required = {'successor': ('point-corr-successor-final.log', '293 passed, 111 subtests passed'),
                'baseline': ('point-corr-baseline-final.log', '207 passed, 137 subtests passed'),
                'native_CLI': ('point-corr-ctest.log', '100% tests passed'),
                'configuration': ('point-corr-config.log', 'config preflight reports written under')}
    checks = {}
    for key, (name, marker) in required.items():
        require(marker in (logs/name).read_text(), 'missing successful gate: '+name)
        checks[key] = bound(logs/name)
    successful = ['point-corr-prepare.log', 'point-corr-calibration.log', 'point-corr-validation.log', 'point-corr-screen.log', 'point-corr-BASE.log', 'point-corr-report.log']
    pattern = r'(?i)\[(?:error|critical)\]|^Traceback \(most recent call last\)|^\w*Error:'
    scans = {name: [s for s in (logs/name).read_text().splitlines() if re.search(pattern, s)] for name in successful}
    native_errors = {str(f.relative_to(out)): [s for s in f.read_text().splitlines() if re.search(pattern, s)] for f in (out/'run/BASE').rglob('run.log')}
    require(not any(scans.values()) and not any(native_errors.values()), 'unexpected successful-execution error log')
    save_json(dest/'VALIDATION.json', dict(gates=checks, successor_tests=293, successor_subtests=111,
              baseline_tests=207, baseline_subtests=137, native_CLI_tests=34, config_require_all=True,
              successful_log_scan=scans, native_log_scan=native_errors,
              retained_attempts='Initial numerical-fixture FP warnings and one test-fixture HDU-name failure were corrected before accepted tests. Initial calibration thresholds are explicitly invalidated; all raw trials and the amendment are retained.',
              interpretation='Expected blank-fit solver failures are trial outcomes, not successful fits or hidden runtime failures.'))
    p = read(out/'freeze/prospective.json')
    (dest/'bin').mkdir()
    binary = verified(p['binary'])
    shutil.copy2(binary, dest/'bin/citlali')
    revisions = dict(native_and_calibration=p['source_revision'], validation=read(out/'VALIDATION.json')['source_revision'],
                     BASE_and_screen=read(out/'FEEDBACK-BASE.json')['source_revision'], report=read(out/'report/REPORT.json')['source_revision'], closeout=revision)
    for label, rev in revisions.items():
        subprocess.run(['git', 'archive', '--format=tar.gz', '--output='+str(dest/('source-'+label+'.tar.gz')), rev], cwd=REPO, check=True)
    diff = subprocess.check_output(['git', 'diff', p['source_revision'], revision, '--', 'src', 'include', 'CMakeLists.txt'], cwd=REPO, text=True)
    require(not diff, 'native source changed after frozen build')
    save_json(dest/'SOURCE_AND_BINARY.json', dict(revisions=revisions, binary=bound(dest/'bin/citlali'),
              version=p['version'], native_tree_diff=diff, archive_scope='Exact source snapshots; clean Git metadata is needed by execution commands.'))
    for name in ('CMakeCache.txt', 'compile_commands.json'):
        path = binary.parent.parent/name
        if path.exists(): shutil.copy2(path, dest/name)
    save_json(dest/'ENVIRONMENT.json', dict(python=sys.version, python_executable=sys.executable, platform=platform.platform(),
              numpy=np.__version__, scipy=scipy.__version__, astropy=astropy.__version__, logical_CPUs=psutil.cpu_count(),
              memory_bytes=psutil.virtual_memory().total, frozen_host=p['host'],
              requested_threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
              actual_BLAS_threads='Apple Accelerate internal count not observable; OS-thread samples retained in every native RUN.json',
              build_scope='Local cached AppleClang environment, supplemental validation; no Unity/Spack V2 claim or dependency installation. Frozen executable version preserves inherited dependency identity including its dirty label.'))
    save_json(dest/'RUNTIME.json', runtime(out))
    shutil.copy2(out/'report/ASSESSMENT.md', out/'ASSESSMENT.md')
    shutil.copy2(out/'report/MATRIX.json', out/'MATRIX.json')
    shutil.copy2(V/'REPRODUCTION.md', out/'REPRODUCTION.md')
    (out/'README.md').write_text(f'# POINT 152389: completed failed-gate experiment\n\n[Assessment and figures]({out/"ASSESSMENT.md"})\n\n[Completed and gate-blocked cells]({out/"MATRIX.json"})\n\n[Reproduction]({out/"REPRODUCTION.md"})\n\nNo CORR/DIAG trajectory was launched. All three BASE controls completed T10. One next action is recommended only.\n')
    save_json(out/'FINAL_AUTHENTICATION.json', auth)
    manifest(dest)
    print(json.dumps(dict(inventory_source=revision, authentication=auth)))


def seal(out, review):
    require(not (out/'FILES_SHA256SUMS').exists(), 'bundle already sealed')
    verify_manifest(out/'provenance')
    auth = authenticate(out)
    require(review.is_file(), 'final independent review missing')
    shutil.copy2(review, out/'FINAL_REVIEW.md')
    save_json(out/'FINAL_AUTHENTICATION.json', auth)
    save_json(out/'COMPLETION.json', dict(status='complete: six new-policy gates unavailable; new trajectories blocked',
              source_revision=source(), sources=read(out/'provenance/SOURCE_AND_BINARY.json'),
              assessment=bound(out/'ASSESSMENT.md'), review=bound(out/'FINAL_REVIEW.md'), matrix=bound(out/'MATRIX.json'),
              authentication=auth, independent_fields=6144, stress_field_reuses=768,
              BASE_replay_trajectories=3, BASE_feedback_applications=30, new_policy_feedback_applications=0,
              stop='No new thresholds, solver change, additional draws, new-policy feedback, m10 application, cycle 11, production/default change, Unity, integration or push.'))
    manifest(out)
    count = verify_manifest(out)
    print(json.dumps(dict(manifest=bound(out/'FILES_SHA256SUMS'), files=count)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['inventory', 'seal'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    inventory(args.output) if args.stage == 'inventory' else seal(args.output, args.review)
