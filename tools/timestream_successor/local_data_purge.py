"""Bounded September 26 local_data purge; no scientific processing or auto-recovery."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json,os,stat,subprocess,time
ROOT=Path('/Users/gwilson/work_toltec/local_data')
WORK=Path('/private/tmp/citlali-point-additive-20260923')
RECORD=WORK/'validation/local_data_rtc_ptc_purge_2026-09-26'

def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)
def save(p,x):
    with (gzip.open(p,'wt') if str(p).endswith('.gz') else open(p,'w')) as f:json.dump(x,f,indent=None if str(p).endswith('.gz') else 2)
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def identity(p,root=None):
    p=Path(p);root=ROOT if root is None else Path(root)
    if not p.is_relative_to(root) or p.resolve()!=p:raise ValueError('outside root or symlink traversal: '+str(p))
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_dev!=root.stat().st_dev:raise ValueError('nonregular/mount target')
    return dict(bytes=s.st_size,blocks=s.st_blocks*512,dev=s.st_dev,ino=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def unchanged(row,root=None):
    root=ROOT if root is None else root
    now=identity(root/row['path'],root)
    if now!={k:row[k]for k in now}:raise ValueError('changed target: '+row['path'])
def slice_indices(v):
    import numpy as np
    if not v.shape:yield Ellipsis;return
    step=max(1,min(v.shape[0],(4<<20)//max(1,int(np.prod(v.shape[1:]))*8)))
    for i in range(0,v.shape[0],step):yield (slice(i,min(i+step,v.shape[0])),)+(slice(None),)*(len(v.shape)-1)
def data_digest(v):
    import numpy as np
    h=hashlib.sha256()
    for s in slice_indices(v):
        a=np.asarray(v[s]);h.update(json.dumps(a.tolist(),ensure_ascii=True).encode() if a.dtype.kind in 'OUS' else a.tobytes())
    return h.hexdigest()
def attributes(obj):return {k:repr(obj.getncattr(k))for k in obj.ncattrs()}
def separate_one(row):
    import netCDF4
    src=ROOT/row['path'];dst=ROOT/row['retained_companion'];created=False
    try:
        unchanged(row)
        if dst.exists() or dst.is_symlink():raise ValueError('companion already exists; no overwrite')
        with netCDF4.Dataset(src) as a:
            if a.groups:raise ValueError('nested groups need separate mixed-container review')
            a.set_auto_maskandscale(False);a.set_auto_chartostring(False)
            kept=row['retained_variables'];drop=row['removed_variables'];dim=row['sample_dimension']
            if set(kept)|set(drop)!=set(a.variables) or any(dim in a[k].dimensions for k in kept):raise ValueError('schema changed')
            original_attrs=attributes(a);bindings={}
            original_schema=dict(dimensions={k:dict(length=len(v),unlimited=v.isunlimited())for k,v in a.dimensions.items()},variables={k:dict(dimensions=list(v.dimensions),shape=list(v.shape),dtype=str(v.dtype),attributes=attributes(v))for k,v in a.variables.items()})
            with netCDF4.Dataset(dst,'w',format='NETCDF4') as b:
                created=True;b.setncatts({k:a.getncattr(k)for k in a.ncattrs()})
                b.setncattr('LOCAL_PURGE_ORIGINAL_PATH',str(src));b.setncattr('LOCAL_PURGE_NOTICE','Non-timestream variables preserved; original RTC/PTC samples deliberately discarded under 2026-09-26 directive. Not an original full payload.')
                used={d for k in kept for d in a[k].dimensions}
                for k in used:b.createDimension(k,len(a.dimensions[k]))
                for k in kept:
                    v=a[k];kwargs={}
                    if '_FillValue'in v.ncattrs():kwargs['fill_value']=v.getncattr('_FillValue')
                    if v.dtype!=str:kwargs.update(zlib=True,complevel=1,shuffle=False)
                    w=b.createVariable(k,v.datatype,v.dimensions,**kwargs);w.set_auto_maskandscale(False);w.set_auto_chartostring(False)
                    w.setncatts({z:v.getncattr(z)for z in v.ncattrs()if z!='_FillValue'})
                    for sl in slice_indices(v):w[sl]=v[sl]
                    bindings[k]=dict(data_sha256=data_digest(v),attributes=attributes(v),shape=list(v.shape),dtype=str(v.dtype))
            with netCDF4.Dataset(dst) as b:
                b.set_auto_maskandscale(False);b.set_auto_chartostring(False)
                if set(b.variables)!=set(kept):raise ValueError('retained variable set mismatch')
                for k,ref in bindings.items():
                    v=b[k]
                    if data_digest(v)!=ref['data_sha256'] or attributes(v)!=ref['attributes'] or list(v.shape)!=ref['shape'] or str(v.dtype)!=ref['dtype']:raise ValueError('retained data/attribute mismatch: '+k)
                if any(attributes(b).get(k)!=v for k,v in original_attrs.items()):raise ValueError('global attributes mismatch')
        unchanged(row)
        return dict(path=row['path'],companion=row['retained_companion'],status='pass',sha256=digest(dst),identity=identity(dst),retained_variables=bindings,removed_variables=drop,original_global_attributes=original_attrs,original_schema=original_schema)
    except Exception as e:
        if created:dst.unlink(missing_ok=True)
        return dict(path=row['path'],status='retained mixed-container exception',error=str(e))
def explain_purged(path):
    """Recognize an executed purge; never offer nonexistent checkpoint restoration."""
    path=Path(path)
    if not path.is_relative_to(ROOT) or path.exists():return
    index=ROOT/'LOCAL_DATA_PURGE_2026-09-26.json'
    if not index.exists():return
    info=read(index)
    if info.get('status')!='completed':return
    paths=ROOT/info['paths_file']
    if digest(paths)!=info['paths_sha256']:raise ValueError('Current purge index identity mismatch')
    if str(path.relative_to(ROOT)) in set(read(paths)):
        raise FileNotFoundError('INTENTIONALLY PURGED RTC/PTC product under 2026-09-26 owner directive: '+str(path)+'; saved restart state deliberately discarded. Reprocessing from documented upstream inputs requires an explicit separate operation. No automatic regeneration. Record: '+info['durable_git_record'])

def separate():
    from concurrent.futures import ProcessPoolExecutor
    rows=[r for r in read(RECORD/'TARGETS_DRAFT.json.gz')if 'retained_companion'in r];start=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(separate_one,rows)):
            results.append(r)
            if (i+1)%50==0:print('Separated',i+1,'/',len(rows),'exceptions',sum(x['status']!='pass'for x in results),flush=True)
    save(RECORD/'MIXED_CONTAINER_PRESERVATION.json.gz',results)
    print('Separation completed',len(results),'seconds',time.monotonic()-start,flush=True)
def freeze():
    rows=read(RECORD/'TARGETS_DRAFT.json.gz');kept={r['path']:r for r in read(RECORD/'MIXED_CONTAINER_PRESERVATION.json.gz')};ok=[];exceptions=[]
    for r in rows:
        if 'retained_companion'in r:
            if kept[r['path']]['status']!='pass':exceptions.append(kept[r['path']]);continue
            r['preservation_sha256']=kept[r['path']]['sha256']
        unchanged(r);ok.append(r)
    save(RECORD/'TARGETS.json.gz',ok);save(RECORD/'PURGE_EXCEPTIONS.json',exceptions)
    save(RECORD/'TARGET_SUMMARY.json',dict(files=len(ok),logical_bytes=sum(r['bytes']for r in ok),allocated_file_blocks_bytes=sum(r['blocks']for r in ok),classes=dict(collections.Counter(r['classification']for r in ok)),sha256=digest(RECORD/'TARGETS.json.gz'),no_numerical_regeneration_required=True))
def apply():
    receipt=RECORD/'EXECUTION_RECEIPT.json'
    if receipt.exists():raise RuntimeError('Execution receipt exists; no automatic retry')
    rows=read(RECORD/'TARGETS.json.gz');review=read(RECORD/'SAFETY_APPROVAL.json');head=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip()
    if review['source_revision']!=head or review['target_sha256']!=digest(RECORD/'TARGETS.json.gz')or review['status']!='pass':raise RuntimeError('Exact review binding mismatch')
    # Review itself is a detached additive record; scientific/recovery/source files must already be committed.
    if subprocess.check_output(['git','-C',str(WORK),'diff','HEAD','--name-only'],text=True).strip():raise RuntimeError('Tracked changes after reviewed commit')
    active=read(RECORD/'ACTIVE_USE_CHECK.json')
    if active['target_open_paths'] or time.time()-active['epoch']>300 or active['incomplete']:raise RuntimeError('Live or stale/incomplete active-use check')
    for r in rows:unchanged(r)
    companions=read(RECORD/'MIXED_CONTAINER_PRESERVATION.json.gz')
    for r in companions:
        if r['status']=='pass':
            now=identity(ROOT/r['companion'])
            if now!=r['identity']or digest(ROOT/r['companion'])!=r['sha256']:raise RuntimeError('Retained companion changed')
    state=dict(status='started',source_revision=head,target_sha256=digest(RECORD/'TARGETS.json.gz'),review_sha256=digest(RECORD/'SAFETY_APPROVAL.json'),files=len(rows),logical_bytes=sum(r['bytes']for r in rows),before_df=subprocess.check_output(['df','-k',str(ROOT)],text=True),started_epoch=time.time())
    save(receipt,state);removed=0;failed=[]
    with open(RECORD/'DELETION_LOG.jsonl','w')as log:
        for r in rows:
            try:
                unchanged(r);log.write(json.dumps(dict(path=r['path'],bytes=r['bytes'],action='unlink intent'))+'\n');log.flush();(ROOT/r['path']).unlink();removed+=1
            except Exception as e:failed.append(dict(path=r['path'],error=str(e)));break
    # Only resulting empty target-parent directories; never sweep other empty trees.
    empty=[]
    for p in sorted({(ROOT/r['path']).parent for r in rows},key=lambda p:-len(p.parts)):
        try:p.rmdir();empty.append(str(p.relative_to(ROOT)))
        except OSError:pass
    state.update(status='completed'if not failed else 'stopped on changed/failed target',removed_files=removed,failures=failed,removed_empty_directories=empty,after_df=subprocess.check_output(['df','-k',str(ROOT)],text=True),seconds=time.time()-state['started_epoch']);save(receipt,state)
    print(json.dumps(state,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['separate','freeze','apply']);args=parser.parse_args();globals()[args.command]()
