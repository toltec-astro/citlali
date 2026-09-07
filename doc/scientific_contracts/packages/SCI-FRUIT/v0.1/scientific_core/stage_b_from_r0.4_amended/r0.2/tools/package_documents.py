"""Package exact r0.2 delivery; preserved historical archives remain opaque files."""
from pathlib import Path
import argparse,gzip,hashlib,io,json,re,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'SCI-FRUIT-v0.1-stage-b-r0.2-owner-review.tar.gz'
MANIFEST=ROOT/'ARTIFACT_MANIFEST.md';SIDE=ROOT/'ARTIFACT_MANIFEST.sha256';ASIDE=Path(str(ARCHIVE)+'.sha256')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def payloads():
    out=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if 'qa'==rel.parts[0] or '__pycache__' in rel.parts:continue
        if p in {ARCHIVE,ASIDE,MANIFEST,SIDE}:continue
        if p.is_file():assert not p.is_symlink();out.append(p)
    return sorted(out)
def role(p):
    rel=p.relative_to(ROOT).as_posix()
    if rel.startswith('inputs/'):return 'Exact preserved authority or historical input; not a new r0.2 reading product'
    if rel.startswith('src/common/') or rel=='src/core_body.tex':return 'Sole normative scientific-core source'
    if rel.startswith('src/'):return 'Distinct document view, shared layout or deterministic digest binding'
    if rel.startswith('pdf/'):return 'Completed r0.2 owner-review PDF'
    if rel.startswith('review/'):return 'Bounded independent consistency review; advisory'
    if rel.startswith('evidence/'):return 'Prospective conformance record/index; numerical comparisons blocked; evidence artifacts not produced'
    if rel.startswith('identities/'):return 'Exact source/input or typesetting-resource identity inventory'
    if rel.startswith('tools/') or rel.startswith('reports/'):return 'Document-only build, verification or evidence'
    return 'Document navigation, authority ledger, derived crosswalk or disposition proposal'
def verify():
    assert SIDE.read_text()==f'{sha(MANIFEST)}  {MANIFEST.name}\n'
    rows=re.findall(r'^\| `([^`]+)` \| [^|]+ \| (\d+) \| `([0-9a-f]{64})` \|$',MANIFEST.read_text(),re.M)
    assert len(rows)==len(payloads())
    for rel,size,digest in rows:
        p=ROOT/rel;assert p.stat().st_size==int(size) and sha(p)==digest,rel
    assert ASIDE.read_text()==f'{sha(ARCHIVE)}  {ARCHIVE.name}\n'
    expected={'r0.2/'+p.relative_to(ROOT).as_posix():p for p in payloads()+[MANIFEST,SIDE]}
    with tarfile.open(ARCHIVE,'r:gz') as tf:
        members=tf.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
        for m in members:assert m.isfile() and tf.extractfile(m).read()==expected[m.name].read_bytes(),m.name
    print(json.dumps({'delivery_payloads':len(rows),'archive_members':len(expected),'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':sha(ARCHIVE),'archive_exact_member_byte_parity':True},indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    if args.verify_only:verify();return
    subprocess.run([sys.executable,str(ROOT/'tools/verify_documents.py')],check=True,stdout=subprocess.DEVNULL)
    checks=json.loads((ROOT/'reports/DOCUMENT_CHECKS.json').read_text())
    assert checks['visual_record_verified'] and checks['correction_review_complete'] and checks['reviewed_snapshot_files_verified']>0
    files=payloads()
    lines=['# Exact delivery manifest - SCI-FRUIT Stage B r0.2','','Status: complete owner-review draft; Stage B not frozen. All numerical methods/routes remain unavailable_pending_separate_owner_approval; numerical comparisons blocked; evidence artifacts not produced.','',f'Sole science: SCI-FRUIT-NORMATIVE-CORE v0.1/r0.2; canonical source-inventory SHA-256 {checks["core_source_inventory_sha256"]}.','', 'Every regular delivery payload appears below once. Manifest self-hash is in ARTIFACT_MANIFEST.sha256. The archive contains these payloads plus manifest/sidecar, without symlinks or extra paths; archive/digest remain external and are not recursively embedded. Task caches, build intermediates, rendered PNGs and the separate external in-progress snapshot are excluded. Historical input archives are opaque member bytes, never relabeled r0.2 products.','', '| Path | Role | Bytes | SHA-256 |','| --- | --- | ---: | --- |']
    for p in files:lines.append(f'| `{p.relative_to(ROOT).as_posix()}` | {role(p)} | {p.stat().st_size} | `{sha(p)}` |')
    MANIFEST.write_text('\n'.join(lines)+'\n');SIDE.write_text(f'{sha(MANIFEST)}  {MANIFEST.name}\n')
    with ARCHIVE.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=1788739200) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT) as tf:
                for p in sorted(files+[MANIFEST,SIDE]):
                    content=p.read_bytes();info=tarfile.TarInfo('r0.2/'+p.relative_to(ROOT).as_posix());info.size=len(content);info.mode=0o644;info.mtime=1788739200;tf.addfile(info,io.BytesIO(content))
    ASIDE.write_text(f'{sha(ARCHIVE)}  {ARCHIVE.name}\n');verify()
if __name__=='__main__':main()
