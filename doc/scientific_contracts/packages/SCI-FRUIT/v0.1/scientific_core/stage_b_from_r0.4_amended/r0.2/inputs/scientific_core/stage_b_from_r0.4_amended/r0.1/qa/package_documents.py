"""Bind exact new document artifacts; package only this output, never Stage A."""
from pathlib import Path
import gzip,hashlib,io,tarfile
root=Path(__file__).resolve().parents[1]
archive=root/'SCI-FRUIT-v0.1-stage-b-r0.1-owner-review.tar.gz'
manifest=root/'ARTIFACT_MANIFEST.md';sidecar=root/'ARTIFACT_MANIFEST.sha256'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
excluded={manifest,sidecar,archive,Path(str(archive)+'.sha256')}
files=sorted(p for p in root.rglob('*') if p.is_file() and p not in excluded and not any(str(p.relative_to(root)).startswith(x) for x in ['qa/cache/','qa/build/','qa/renders/','qa/__pycache__/']))
assert all(not p.is_symlink() for p in files)
def role(p):
    rel=p.relative_to(root).as_posix()
    if rel.startswith('src/common/'):return 'Canonical conditional scientific authority'
    if rel=='src/PTC_APPLICATION_REFERENCE.tex':return 'Exact permitted frozen quotation under cover'
    if rel.startswith('src/'):return 'View wrapper or shared typesetting'
    if rel.startswith('pdf/'):return 'Compiled owner-review view'
    if rel.startswith('review/'):return 'Bounded independent consistency review'
    if rel.startswith('qa/'):return 'Document verification tool or evidence only'
    return 'Document navigation, decision, identity or verification record'
lines=['# SCI-FRUIT v0.1 / Stage B r0.1 — exact artifact manifest','','Status: owner-review draft; not frozen. Numerical methods/routes: unavailable_pending_separate_owner_approval. Scientific read-set identity is bound in SOURCE_IDENTITIES.md. This manifest adds no scientific authority.','', 'The rows bind every regular delivery payload. This manifest does not hash itself; ARTIFACT_MANIFEST.sha256 binds its exact bytes. The archive contains these payloads and this manifest/sidecar exactly once, without symlinks or extra paths. Archive/digest are external delivery controls, not recursively embedded. Caches, build intermediates and rendered PNGs are excluded.','', '| File | Role | Bytes | SHA-256 |','| --- | --- | ---: | --- |']
for p in files:lines.append(f'| `{p.relative_to(root).as_posix()}` | {role(p)} | {p.stat().st_size} | `{sha(p)}` |')
manifest.write_text('\n'.join(lines)+'\n');sidecar.write_text(f'{sha(manifest)}  ARTIFACT_MANIFEST.md\n')
members=files+[manifest,sidecar]
with archive.open('wb') as raw:
    with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=1788739200) as gz:
        with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT) as tf:
            for p in sorted(members):
                b=p.read_bytes();info=tarfile.TarInfo('r0.1/'+p.relative_to(root).as_posix());info.size=len(b);info.mode=0o644;info.mtime=1788739200
                tf.addfile(info,io.BytesIO(b))
Path(str(archive)+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
with tarfile.open(archive,'r:gz') as tf:
    actual=tf.getmembers();expected={'r0.1/'+p.relative_to(root).as_posix():p for p in members}
    assert len(actual)==len(expected) and {m.name for m in actual}==set(expected)
    for m in actual:
        assert m.isfile() and tf.extractfile(m).read()==expected[m.name].read_bytes(),m.name
assert sidecar.read_text()==f'{sha(manifest)}  ARTIFACT_MANIFEST.md\n'
print(f'{len(files)} delivery payloads; {len(members)} exact archive members; archive parity passed')
print(f'{archive.name}: {archive.stat().st_size} bytes; SHA-256 {sha(archive)}')
