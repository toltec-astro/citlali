"""Explicit, manifest-controlled duplicate compaction for POINT 152389 only.

No numerical generation, fitting, candidate selection or implicit restoration.
Original scientific manifests remain historical. Paths are absolute locators;
SHA256 binds identity. GB means 10**9 bytes; GiB means 2**30 bytes.
"""
import argparse, concurrent.futures, gzip, hashlib, json, os, shutil, stat, subprocess, time
from pathlib import Path

BASE=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs')
OUT=BASE/'successor-point-152389-importance-and-archive-compaction-20260926'
ROOTS=(BASE/'successor-point-152389-guarded-feedback-comparison-20260925',BASE/'successor-point-152389-response-robustness-cd-20260923')
CLASSES={'cleaned.f64','operand.f64','injection.f64','operator-pixel.i32','causes.u8','model-support.u8'}

def require(ok,why):
    if not ok:raise ValueError(why)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()

def read(path):
    p=Path(path);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())

def save(path,value):
    p=Path(path);data=(json.dumps(value,indent=2)+'\n').encode()
    p.write_bytes(gzip.compress(data,mtime=0) if p.suffix=='.gz' else data)

def identity(path):
    p=Path(path);require(p.resolve()==p,'symlink/alias path rejected: '+str(p));s=p.lstat()
    require(stat.S_ISREG(s.st_mode),'not a regular file: '+str(p))
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,nlink=s.st_nlink)

def check_scope(path,roots=ROOTS):
    p=Path(path);require(p.name in CLASSES,'protected/unclassified product: '+str(p))
    require(any(p.is_relative_to(r/'run') for r in roots),'protected/out-of-scope path: '+str(p))
    require('native/products' in str(p),'not an experiment native product: '+str(p))
    return p

def validate_closure(actions,roots=ROOTS):
    targets={a['path'] for a in actions};require(len(targets)==len(actions),'duplicate action')
    for a in actions:
        check_scope(a['path'],roots);check_scope(a['retained'],roots)
        require(a['retained'] not in targets,'retained dependency scheduled for deletion')
        require(a['path']!=a['retained'],'self replacement')
        require(len(a['sha256'])==64 and a['bytes']>=0,'invalid identity')
    return True

def changed_rejection(path,expected):
    require(identity(path)==expected,'file changed since inventory: '+str(path))

def verify_file(row):
    path,expected,h=row;changed_rejection(path,expected)
    require(digest(path)==h,'byte identity mismatch: '+str(path));changed_rejection(path,expected)
    return path

def freeze(out=OUT):
    require(not (out/'ACTION_MANIFEST.json.gz').exists(),'action manifest already frozen')
    groups=read(out/'DUPLICATE_CANDIDATES.json.gz');actions=[];checks={}
    for g in groups:
        keep=g['retained'];kp=keep['path'];ks={k:keep[k] for k in ('device','inode','bytes','mtime_ns','ctime_ns','nlink')}
        changed_rejection(kp,ks);checks[kp]=(kp,ks,g['sha256'])
        for row in g['duplicates']:
            p=row['path'];s={k:row[k] for k in ks};changed_rejection(p,s);checks[p]=(p,s,g['sha256'])
            actions.append(dict(path=p,retained=kp,sha256=g['sha256'],bytes=g['bytes'],classification='inactive byte-identical native duplicate',product_class=g['class'],identity=s,retained_identity=ks,recovery='explicit restore from retained exact bytes; no numerical regeneration',verification='FRESH_DUPLICATE_VERIFICATION.json'))
    validate_closure(actions);save(out/'ACTION_MANIFEST.json.gz',actions)
    start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n,_ in enumerate(pool.map(verify_file,checks.values()),1):
            if n%10000==0:print('hashed',n,'files',flush=True)
    summary=dict(status='dry run verified; no removal',action_manifest_sha256=digest(out/'ACTION_MANIFEST.json.gz'),unique_files_hashed=len(checks),targets=len(actions),logical_bytes=sum(a['bytes'] for a in actions),retained_canonical_files=len(checks)-len(actions),seconds=time.monotonic()-start,closure='all replacements retained directly; no replacement chain; exact source bytes recoverable',scientific_applications=0,fits=0,synthetic_trials=0)
    save(out/'FRESH_DUPLICATE_VERIFICATION.json',summary);print(json.dumps(summary),flush=True)

def resolve(path,actions):
    from local_data_purge import explain_purged
    explain_purged(path)
    p=Path(path);by_path={a['path']:a for a in actions}
    if p.exists():
        identity(p)
        if str(p) in by_path:h=by_path[str(p)]['sha256']
        else:
            root=next((r for r in ROOTS if p.is_relative_to(r)),None)
            require(root is not None,'path outside compact archive')
            ledger={n:h for h,n in (line.split('  ',1) for line in (root/'FILES_SHA256SUMS').read_text().splitlines())}
            require(str(p.relative_to(root)) in ledger,'artifact has no historical identity')
            h=ledger[str(p.relative_to(root))]
        require(digest(p)==h,'corrupt retained artifact')
        return dict(status='retained',path=str(p),sha256=h)
    if str(p) not in by_path:raise FileNotFoundError('missing/corrupt, not intentionally compacted: '+str(p))
    a=by_path[str(p)];q=Path(a['retained']);explain_purged(q);require(q.is_file() and digest(q)==a['sha256'],'missing/corrupt retained counterpart')
    return dict(status='deduplicated; explicit restore required for historical layout',path=str(q),original_path=str(p),sha256=a['sha256'])

def restore(path,actions):
    a=next((a for a in actions if a['path']==str(Path(path))),None);require(a is not None,'path not in compact archive')
    p=Path(a['path']);require(not p.exists() and not p.is_symlink(),'restore destination exists')
    q=resolve(p,actions);require(p.parent.is_dir() and p.parent.resolve()==p.parent,'unsafe/missing parent')
    with Path(q['path']).open('rb') as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
    require(digest(p)==a['sha256'],'restoration digest mismatch');return dict(status='restored',path=str(p),bytes=p.stat().st_size)

def explain_missing(path,actions=None):
    """Fail explicitly; never change a scientific reader into a generator."""
    from local_data_purge import explain_purged
    explain_purged(path)
    p=Path(path)
    if actions is None:
        if not any(p.is_relative_to(r) for r in ROOTS) or not (OUT/'ACTION_MANIFEST.json.gz').exists():
            raise FileNotFoundError('missing/corrupt artifact: '+str(p))
        actions=read(OUT/'ACTION_MANIFEST.json.gz')
    row=next((a for a in actions if a['path']==str(p)),None)
    if row is None:raise FileNotFoundError('missing/corrupt artifact: '+str(p))
    explain_purged(row['retained'])
    raise FileNotFoundError('INTENTIONALLY COMPACTED; explicit restoration required. '
        +'Use point_archive_compaction.py restore '+str(p)+'; retained exact bytes: '+row['retained'])

def apply(out=OUT):
    actions=read(out/'ACTION_MANIFEST.json.gz');v=read(out/'FRESH_DUPLICATE_VERIFICATION.json');review=read(out/'MANIFEST_REVIEW.json')
    repo=Path(__file__).resolve().parents[2]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    require(review['source_revision']==revision,'unreviewed source revision')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),'source must remain clean before removal')
    require(digest(out/'ACTION_MANIFEST.json.gz')==v['action_manifest_sha256']==review['action_manifest_sha256'],'manifest binding changed')
    require(review['disposition']=='pass','independent manifest review required');validate_closure(actions)
    require(not (out/'REMOVAL_RECEIPT.json').exists(),'no automatic application retry')
    for a in actions:
        changed_rejection(a['path'],a['identity']);changed_rejection(a['retained'],a['retained_identity'])
        require(a['identity']['nlink']==a['retained_identity']['nlink']==1,'hardlinked payload needs separate disposition')
    record=dict(status='applying owner-authorized reviewed manifest',files=0,logical_bytes=0,manifest_sha256=v['action_manifest_sha256'],free_space_before=subprocess.check_output(['df','-k',str(out)],text=True),started_epoch=time.time(),authorization='2026-09-26 owner scientific-importance/archive-compaction directive, section6; no routine additional approval')
    record.update(source_revision=revision,review_sha256=digest(out/'MANIFEST_REVIEW.json'))
    save(out/'REMOVAL_RECEIPT.json',record)
    try:
        with (out/'REMOVAL_LOG.jsonl').open('x') as log:
            for i,a in enumerate(actions):
                changed_rejection(a['path'],a['identity']);changed_rejection(a['retained'],a['retained_identity'])
                log.write(json.dumps({'path':a['path'],'sha256':a['sha256'],'retained':a['retained']})+'\n');log.flush()
                Path(a['path']).unlink();record['files']+=1;record['logical_bytes']+=a['bytes']
                if i%10000==0:print('removed',i+1,'verified duplicate files',flush=True)
            os.fsync(log.fileno())
        for a in actions:
            require(not Path(a['path']).exists(),'removed target still exists');changed_rejection(a['retained'],a['retained_identity'])
        record.update(status='completed',free_space_after=subprocess.check_output(['df','-k',str(out)],text=True),seconds=time.time()-record['started_epoch'],retained_counterparts_unchanged=True,all_targets_absent=True)
    except BaseException as exc:
        record.update(status='stopped; no automatic retry',error=repr(exc));save(out/'REMOVAL_RECEIPT.json',record);raise
    save(out/'REMOVAL_RECEIPT.json',record);print(json.dumps(record),flush=True)

def importance_ratio(effect,sigma,units,scale_units,applicability):
    require(units==scale_units,'measurement/scale units differ')
    require(applicability in ('conditional_reference','historical_context','matched_endpoint','unavailable'),'unknown scale applicability')
    if applicability=='unavailable':return None
    require(sigma is not None and sigma>0,'nonpositive/missing scale')
    return abs(effect)/sigma

def fraction(effect,denominator,kind):
    require(kind in ('own_reference_source','ideal_injected_increment'),'explicit denominator required')
    require(denominator!=0,'zero reference');return 100*effect/denominator

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['freeze','apply','status','restore']);ap.add_argument('path',nargs='?');args=ap.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='apply':apply()
    else:
        require(args.path is not None,'one explicit path required');a=read(OUT/'ACTION_MANIFEST.json.gz');print(json.dumps((resolve if args.command=='status' else restore)(args.path,a),indent=2))
