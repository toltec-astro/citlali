"""Enumerate or delete only this independent campaign's inactive native arrays."""
import argparse,hashlib,json,os,time,shutil,gzip
from pathlib import Path
SUFFIXES={'.f64','.u8','.u16','.i32','.i64','.u64'}
NAME='successor-point-152430-independent-a2000-20260926'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def record(p,out):return dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest(p))
def safe(out):
    assert out.name==NAME and out.resolve()==out and not out.is_symlink()
    assert json.loads((out/'freeze/prospective.json').read_text())['observation']==152430
    assert (out/'report/matrix.json').is_file()
def verify_inputs(out):
    for r in json.loads((out/'freeze/dependencies.json').read_text())['inputs']:
        assert digest(Path(r['path']))==r['sha256'],r['path']
def plan(out):
    safe(out);verify_inputs(out);targets=[];retained=[];compressed=[]
    for p in sorted((out/'upstream').rglob('*.yaml')):
        if p.stat().st_size<100_000:continue
        q=out/'metadata_archive'/p.relative_to(out);q=q.with_suffix(q.suffix+'.gz');q.parent.mkdir(parents=True,exist_ok=True)
        payload=p.read_bytes();q.write_bytes(gzip.compress(payload,compresslevel=9,mtime=0));assert gzip.decompress(q.read_bytes())==payload
        compressed.append(dict(original=record(p,out),archive=record(q,out),representation='lossless gzip native provenance; not a runnable checkpoint'))
    for p in sorted((out/'M0').rglob('*.yaml')):
        if p.stat().st_size<100_000:continue
        q=out/'metadata_archive'/p.relative_to(out);q=q.with_suffix(q.suffix+'.gz');q.parent.mkdir(parents=True,exist_ok=True)
        payload=p.read_bytes();q.write_bytes(gzip.compress(payload,compresslevel=9,mtime=0));assert gzip.decompress(q.read_bytes())==payload
        compressed.append(dict(original=record(p,out),archive=record(q,out),representation='lossless gzip native provenance; not a runnable checkpoint'))
    originals={r['original']['path'] for r in compressed}
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        assert not p.is_symlink() and p.resolve().is_relative_to(out)
        rel=p.relative_to(out)
        if (rel.parts[0] in ('upstream','initial','run','M0') and p.suffix in SUFFIXES) or str(rel) in originals:targets.append(record(p,out))
        elif rel.parts[0]!='cleanup':retained.append(record(p,out))
    d=out/'cleanup';d.mkdir(exist_ok=True)
    manifest=dict(schema='point152430-enumerated-inactive-native-purge-v1',root=str(out),source='Only this campaign native numeric outputs, after terminal matrix and compact evidence verification. Raw, tune, calibration and outside-tree inputs protected.',targets=targets,retained=retained,compressed_provenance=compressed,remove_bytes=sum(r['bytes'] for r in targets),retain_bytes=sum(r['bytes'] for r in retained),created_epoch=time.time())
    (d/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ('targets','retained','compressed_provenance')}))
def apply(out,review):
    safe(out);d=out/'cleanup';p=d/'manifest.json';r=json.loads(review.read_text());assert r['disposition']=='pass' and r['manifest_sha256']==digest(p)
    assert not (d/'execution.json').exists();m=json.loads(p.read_text());assert m['root']==str(out)
    verify_inputs(out)
    for row in m['retained']:
        q=out/row['path'];assert q.is_file() and digest(q)==row['sha256'],str(q)
    compressed={r['original']['path']:r for r in m.get('compressed_provenance',[])}
    for row in m['targets']:
        q=out/row['path'];assert q.resolve().is_relative_to(out) and not q.is_symlink() and (q.suffix in SUFFIXES or row['path'] in compressed) and q.relative_to(out).parts[0] in ('upstream','initial','run','M0')
        assert q.stat().st_size==row['bytes'] and digest(q)==row['sha256'],str(q)
        if row['path'] in compressed:assert hashlib.sha256(gzip.decompress((out/compressed[row['path']]['archive']['path']).read_bytes())).hexdigest()==row['sha256']
    start=time.perf_counter();free=shutil.disk_usage(out).free
    with (d/'removed.jsonl').open('x') as log:
        for row in m['targets']:
            (out/row['path']).unlink();log.write(json.dumps(row)+'\n');log.flush()
    verify_inputs(out)
    for row in m['retained']:assert digest(out/row['path'])==row['sha256']
    receipt=dict(manifest_sha256=digest(p),review_sha256=digest(review),removed_files=len(m['targets']),removed_logical_bytes=m['remove_bytes'],retained_files_verified=len(m['retained']),all_protected_inputs_unchanged=True,free_space_gain_bytes=shutil.disk_usage(out).free-free,seconds=time.perf_counter()-start,final_bytes=sum(q.stat().st_size for q in out.rglob('*') if q.is_file()))
    (d/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','apply']);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path);a=p.parse_args();plan(a.output) if a.action=='plan' else apply(a.output,a.review)
