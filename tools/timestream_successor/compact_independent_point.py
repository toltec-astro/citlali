"""Remove only inactive native numeric payloads from this completed POINT run.
Maps, fitted measurements, masks, exact inputs/configuration and source survive.
"""
import argparse,json,hashlib,time,gzip
from pathlib import Path
NAME='successor-point-152430-a2000-no-upper-coverage-veto-20260926'
SUFFIXES={'.f64','.f32','.u32','.u8','.u16','.i32','.i64','.u64','.bits'}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def clean(out,apply=False):
    assert out.name==NAME and out.resolve()==out and not out.is_symlink()
    assert (out/'report/matrix.json').is_file()
    for p in out.glob('PHASE-*.json'):assert json.loads(p.read_text())['status']!='running'
    protected=json.loads((out/'freeze/dependencies.json').read_text())['inputs']
    for r in protected:assert digest(Path(r['path']))==r['sha256'] and not Path(r['path']).resolve().is_relative_to(out)
    targets=[]
    for root in ('upstream','M0','initial','run'):
        for p in (out/root).rglob('*'):
            assert not p.is_symlink(),str(p)
            if p.is_file() and p.suffix in SUFFIXES:
                assert p.resolve().is_relative_to(out)
                targets.append((p,p.stat().st_size))
    result=dict(root=str(out),files=len(targets),bytes=sum(n for p,n in targets),scope='only run-owned native numeric payload suffixes under upstream/M0/initial/run; no external input or retained map',applied=apply)
    if apply:
        assert not (out/'CLEANUP.json').exists()
        for p,n in targets:assert p.stat().st_size==n;p.unlink()
        compressed=[]
        for root in ('upstream','M0','initial','run'):
            for p in sorted((out/root).rglob('*.yaml')):
                if p.stat().st_size<100_000:continue
                raw=p.read_bytes();q=p.with_suffix(p.suffix+'.gz');assert not q.exists()
                q.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0));assert gzip.decompress(q.read_bytes())==raw
                compressed.append(dict(path=str(p.relative_to(out)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),compressed_path=str(q.relative_to(out)),compressed_bytes=q.stat().st_size))
                p.unlink()
        result['lossless_large_metadata']=compressed
        result['representation']='large current-run native YAML is lossless gzip at same path plus .gz; binary native numeric payloads intentionally removed; raw reprocessing required for restart'

        for r in protected:assert digest(Path(r['path']))==r['sha256']
        result.update(protected_inputs_unchanged=True,retained_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
        (out/'CLEANUP.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
def compress_maps(out):
    assert (out/'CLEANUP.json').is_file() and not (out/'MAP_COMPRESSION.json').exists()
    assert out.resolve()==out.absolute() and not out.is_symlink()
    assert out.name==NAME or out.resolve()==Path(__file__).resolve().parents[2]/'validation/point_152430_no_upper_coverage_veto_2026-09-26/compact'
    count=0;before=0;after=0
    for p in sorted(out.rglob('*.fits')):
        assert not p.is_symlink() and p.resolve().is_relative_to(out.resolve())
        raw=p.read_bytes();q=p.with_suffix('.fits.gz');assert not q.exists()
        packed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(packed)==raw
        q.write_bytes(packed);assert gzip.decompress(q.read_bytes())==raw
        before+=len(raw);after+=len(packed);count+=1;p.unlink()
    result=dict(files=count,original_bytes=before,compressed_bytes=after,representation='lossless individual FITS gzip; all original decompressed bytes verified before removing uncompressed copy; original paths/hashes retained in producer bindings',numeric_values_changed=False)
    (out/'MAP_COMPRESSION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--apply',action='store_true');p.add_argument('--compress-maps',action='store_true');a=p.parse_args();compress_maps(a.output) if a.compress_maps else clean(a.output,a.apply)
