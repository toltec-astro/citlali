"""One foreground, exact-manifest cleanup; no science, retries or recursive deletion."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

D = Path(__file__).resolve().parent
W = D.parents[2]
ROOT = Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-scratch-cleanup-20260929')
PYTHON = '/Users/gwilson/tolteca/bin/python'
SUFFIXES = {'.f64', '.f32', '.u32', '.u8', '.u16', '.i32', '.i64', '.u64', '.bits'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(p):
    raw = p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix == '.gz' else raw)


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, value):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def validate_entries(plan, protected):
    root = Path(plan['root'])
    require(root.is_dir() and root.resolve() == root, 'noncanonical campaign root')
    seen = set()
    for r in plan['files']:
        p = Path(r['path'])
        require(p.is_absolute() and p.resolve() == p and p.is_relative_to(root), 'foreign or symlinked path')
        rel = p.relative_to(root)
        require(len(rel.parts) >= 3 and rel.parts[1] in {'upstream', 'M0', 'initial', 'run', 'native'}, 'foreign stage')
        require(p.suffix in SUFFIXES, 'not native scratch')
        require(str(p) not in seen and str(p) not in protected, 'duplicate or protected target')
        require(p.is_file(), 'missing target; no automatic partial-cleanup resume')
        seen.add(str(p))
    require(sum(r['bytes'] for r in plan['files']) == plan['bytes'], 'manifest byte total differs')
    return seen


def verify_native(plans):
    for plan in plans:
        for r in plan['files']:
            p = Path(r['path'])
            require(p.stat().st_size == r['bytes'] and digest(p) == r['sha256'], 'native changed: ' + str(p))


def inactive(plans):
    import psutil
    for plan in plans:
        root = Path(plan['root'])
        expected = {'B-REF-rep1': 'failed'} if 'retained-residuals' in root.name else {
            'B-REF-rep1': 'completed', 'Q-REF': 'completed', 'Q-C-core-plus': 'completed',
            'Q-C-core-minus': 'completed', 'Q-C-shoulder': 'completed'}
        launches = {p.parent.name: read(p)['status'] for p in root.glob('*/LAUNCH.json')}
        require(launches == expected, 'campaign is not sealed: ' + str(root))
    for proc in psutil.process_iter(['pid', 'uids']):
        if proc.pid == os.getpid() or proc.info['uids'].real != os.getuid():
            continue
        try:
            cmd = proc.cmdline()
            require(not any(str(plan['root']) in arg for plan in plans for arg in cmd),
                    'another process references a cleanup campaign: ' + str(proc.pid))
            if cmd and (Path(cmd[0]).name == 'citlali' or any(Path(a).name.startswith('point_') for a in cmd[:3])):
                cwd = Path(proc.cwd())
                require(not any(cwd.is_relative_to(plan['root']) for plan in plans), 'producer cwd in campaign')
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        # AccessDenied on our own process inspection fails closed.


def verify_protected(protected):
    for path, sha in protected.items():
        require(digest(Path(path)) == sha, 'protected input changed: ' + path)


def report_isolation(evidence, destination):
    # Reuse the exact saved verifier. Redirect only its final receipt so historical
    # evidence (also protected by the later campaign) remains byte-identical.
    source = (evidence / 'verify_saved_report.py').read_text()
    needle = "(D/'REPORT_ISOLATION.json').write_text"
    require(source.count(needle) == 1, 'unknown isolation verifier layout')
    source = source.replace(needle, '(Path(' + repr(str(destination)) + ')).write_text')
    code = "__file__=" + repr(str(evidence / 'verify_saved_report.py')) + '\n' + source
    subprocess.run([PYTHON, '-c', code], check=True)
    result = read(destination)
    require(result['pass_'] and result['positive_denial_control'], 'saved report isolation failed')
    return result


def verify_review():
    review = read(D / 'REVIEW.json')
    require(review['verdict'] == 'pass', 'cleanup review not passed')
    rev = review['candidate_sha']
    subprocess.run(['git', 'merge-base', '--is-ancestor', rev, 'HEAD'], cwd=W, check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', rev, 'HEAD'], cwd=W, text=True).splitlines()
    allowed = {str((D / name).relative_to(W)) for name in ('REVIEW.json', 'HANDOFF.md')}
    require(set(changed) <= allowed, 'unreviewed changes after cleanup candidate')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip(), 'dirty worktree')
    require(review['script_sha256'] == digest(Path(__file__)), 'reviewed cleanup script differs')
    require(review['approval_sha256'] == digest(D / 'OWNER_APPROVAL.json'), 'reviewed approval differs')
    return review


def main():
    review = verify_review()
    approval = read(D / 'OWNER_APPROVAL.json')
    require(approval['authorized'] is True, 'owner approval absent')
    ROOT.mkdir(exist_ok=True)
    require(ROOT.resolve() == ROOT, 'noncanonical cleanup output root')
    # Exclusive creation prevents duplicate launch or accidental retry after interruption.
    with (ROOT / 'LAUNCH.json').open('x') as f:
        json.dump({'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat(),
                   'reviewed_candidate': review['candidate_sha']}, f)
    state = dict(state='RUNNING', phase='preflight', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                 deleted_files=0, deleted_bytes=0, log=str(ROOT / 'cleanup.log'),
                 receipt=str(ROOT / 'CLEANUP.json'), exit_status=None, science_launched=False)
    def update(phase=None):
        if phase:
            state['phase'] = phase
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        save(ROOT / 'STATUS.json', state)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    start = time.monotonic()
    try:
        update()
        plans = []
        protected = {}
        for spec in approval['plans']:
            p = Path(spec['manifest'])
            require(digest(p) == spec['manifest_sha256'], 'approved manifest changed')
            plan = read(p)
            require(not plan['applied'] and plan['root'] == spec['root'] and len(plan['files']) == spec['files'] and plan['bytes'] == spec['bytes'], 'approval scope differs')
            plans.append(plan)
            for row in read(Path(spec['evidence_directory']) / 'PROTECTED_INPUTS.json'):
                require(row['path'] not in protected or protected[row['path']] == row['sha256'], 'inconsistent protection digest')
                protected[row['path']] = row['sha256']
        targets = set()
        for plan in plans:
            entries = validate_entries(plan, protected)
            require(not targets.intersection(entries), 'overlapping cleanup manifests')
            targets.update(entries)
        require(len(targets) == approval['files'] and sum(p['bytes'] for p in plans) == approval['bytes'], 'approval total differs')
        inactive(plans)
        verify_protected(protected)
        keep = {}
        for spec in approval['plans']:
            evidence = Path(spec['evidence_directory'])
            for row in read(evidence / 'COMPACT_MANIFEST.json')['files']:
                packed = evidence / 'compact' / row['path']
                require(digest(packed) == row['sha256'], 'compact digest mismatch')
                raw = gzip.decompress(packed.read_bytes()) if row['encoding'] == 'gzip' else packed.read_bytes()
                require(hashlib.sha256(raw).hexdigest() == row['source_sha256'], 'decoded compact mismatch')
            for p in Path(spec['root']).rglob('*'):
                require(not p.is_symlink(), 'symlink in campaign')
                if p.is_file() and str(p) not in targets:
                    keep[str(p)] = digest(p)
            report_isolation(evidence, ROOT / (evidence.name + '-before-isolation.json'))
        update('verify-native-hashes')
        verify_native(plans)
        inactive(plans)
        update('delete-exact-manifests')
        free_before = __import__('shutil').disk_usage(ROOT).free
        with (ROOT / 'DELETED.jsonl').open('x') as journal:
            for plan in plans:
                for row in plan['files']:
                    p = Path(row['path'])
                    require(p.resolve() == p and p.stat().st_size == row['bytes'], 'target changed before unlink')
                    p.unlink()  # Sole deletion operation; no recursive directory removal.
                    state['deleted_files'] += 1
                    state['deleted_bytes'] += row['bytes']
                    journal.write(json.dumps({'path': str(p), 'bytes': row['bytes']}) + '\n')
                    if state['deleted_files'] % 1000 == 0:
                        journal.flush()
                        update()
        update('verify-preservation-and-reporting')
        verify_protected(protected)
        for path, sha in keep.items():
            require(digest(Path(path)) == sha, 'non-native campaign evidence changed')
        remaining = []
        # One exhaustive post-deletion inventory per campaign.
        for plan in plans:
            paths = [p for p in Path(plan['root']).rglob('*') if p.is_file()]
            require(not any(p.suffix in SUFFIXES for p in paths), 'unexpected remaining native scratch')
            remaining.append(dict(root=plan['root'], files=len(paths), bytes=sum(p.stat().st_size for p in paths)))
        after = []
        for spec in approval['plans']:
            evidence = Path(spec['evidence_directory'])
            after.append(report_isolation(evidence, ROOT / (evidence.name + '-after-isolation.json')))
        require(state['deleted_files'] == approval['files'] and state['deleted_bytes'] == approval['bytes'], 'removal accounting mismatch')
        receipt = dict(applied=True, files=state['deleted_files'], bytes=state['deleted_bytes'],
                       protected_files_unchanged=len(protected), non_native_files_unchanged=len(keep),
                       remaining=remaining, retained_shared_bytes=sum(r['bytes'] for r in remaining),
                       cap_bytes=30_000_000_000, remaining_cap_headroom=30_000_000_000-sum(r['bytes'] for r in remaining),
                       free_space_delta_bytes=__import__('shutil').disk_usage(ROOT).free-free_before,
                       reviewed_candidate=review['candidate_sha'], approval_sha256=digest(D / 'OWNER_APPROVAL.json'),
                       manifest_bindings=approval['plans'], post_cleanup_exact_report_artifacts=sum(r['exact_artifacts'] for r in after),
                       seconds=time.monotonic()-start, exhaustive_post_inventory_per_campaign=1,
                       next='Owner completion check; full transfer capacity preflight still required. No scientific launch.')
        save(ROOT / 'CLEANUP.json', receipt)
        state.update(state='COMPLETE_PENDING_OWNER_REVIEW', exit_status=0)
        update('complete')
    except BaseException as exc:
        state.update(state='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                     exit_status=130 if isinstance(exc, KeyboardInterrupt) else 1, error=repr(exc))
        state['interruption_accounting'] = 'Confirmed counters/journal only; interruption between unlink and journal requires inspection, never automatic retry.'
        update('stopped-possible-partial-cleanup' if state['deleted_files'] or state['phase'] == 'delete-exact-manifests' else 'stopped-before-deletion')
        raise


if __name__ == '__main__':
    # Log all numerical/report subprocess output once, never stream it to a monitor.
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'cleanup.log').open('x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            main()
        except BaseException as exc:
            existing = read(ROOT / 'STATUS.json') if (ROOT / 'STATUS.json').exists() else {}
            if existing.get('pid') != os.getpid():
                save(ROOT / 'STATUS.json', dict(state='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                    phase='startup-before-deletion', pid=os.getpid(), error=repr(exc), deleted_files=0,
                    deleted_bytes=0, exit_status=130 if isinstance(exc, KeyboardInterrupt) else 1,
                    updated_at=datetime.now(timezone.utc).isoformat(), log=str(ROOT / 'cleanup.log')))
            raise
