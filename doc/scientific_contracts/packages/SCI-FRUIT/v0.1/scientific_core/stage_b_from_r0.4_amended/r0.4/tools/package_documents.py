"""Package exact r0.4 documents; outer archive/digest are external sibling files."""
from pathlib import Path, PurePosixPath
import argparse,gzip,hashlib,io,json,re,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[1]
NAME='SCI-FRUIT-v0.1-stage-b-r0.4-owner-review.tar.gz'
ARCHIVE=ROOT/NAME
MANIFEST=ROOT/'ARTIFACT_MANIFEST.md';SIDE=ROOT/'ARTIFACT_MANIFEST.sha256';ASIDE=ROOT/(NAME+'.sha256')
EPOCH=1788825600
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def payloads():
    out=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if rel.parts[0]=='qa' or '__pycache__' in rel.parts:continue
        assert not p.is_symlink(),('symlink',rel)
        if p in {ARCHIVE,ASIDE,MANIFEST,SIDE}:continue
        if p.is_file():
            assert p.stat().st_nlink==1,('hard link',rel)
            out.append(p)
    return sorted(out)
def role(p):
    rel=p.relative_to(ROOT).as_posix()
    if rel.startswith('inputs/'):return 'Exact preserved directive, admitted source or historical input; not a current reading product'
    if rel.startswith('src/common/') or rel=='src/core_body.tex':return 'Complete sole current review-candidate normative-core source'
    if rel.startswith('src/'):return 'Document view, shared layout or deterministic digest binding'
    if rel.startswith('pdf/'):return 'Completed r0.4 review-candidate PDF'
    if rel.startswith('review/'):return 'Author source/semantic review record; prior independent reviews remain historical'
    if rel.startswith('evidence/'):return 'Prospective conformance record/index; comparisons blocked; artifacts not produced'
    if rel.startswith('identities/'):return 'Exact source/input or typesetting-resource identity inventory'
    if rel.startswith('tools/') or rel.startswith('reports/'):return 'Document-only build, verification or QA record'
    return 'Navigation, candidate ledger, derived crosswalk or disposition proposal'
def verify(archive):
    assert SIDE.read_text()==f'{sha(MANIFEST)}  {MANIFEST.name}\n'
    rows=re.findall(r'^\| `([^`]+)` \| [^|]+ \| (\d+) \| `([0-9a-f]{64})` \|$',MANIFEST.read_text(),re.M)
    files=payloads();assert len(rows)==len(files)
    assert {r[0] for r in rows}=={p.relative_to(ROOT).as_posix() for p in files}
    for rel,size,digest in rows:
        p=ROOT/rel;assert p.resolve().is_relative_to(ROOT) and p.stat().st_size==int(size) and sha(p)==digest,rel
    aside=Path(str(archive)+'.sha256')
    assert archive.name==NAME and aside.read_text()==f'{sha(archive)}  {archive.name}\n'
    expected={'r0.4/'+p.relative_to(ROOT).as_posix():p for p in files+[MANIFEST,SIDE]}
    with tarfile.open(archive,'r:gz') as tf:
        members=tf.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
        for m in members:
            path=PurePosixPath(m.name)
            assert not path.is_absolute() and '..' not in path.parts and '.' not in path.parts and path.parts[0]=='r0.4',m.name
            assert m.isfile() and not m.issym() and not m.islnk() and not m.linkname,m.name
            assert tf.extractfile(m).read()==expected[m.name].read_bytes(),m.name
    print(json.dumps({'delivery_payloads':len(rows),'archive_members':len(expected),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'manifest_sha256':sha(MANIFEST),'archive_exact_member_byte_parity':True,'unsafe_member_paths_or_types':0,'external_sidecar_verified':True,'outer_artifacts_excluded_from_members':True},indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');ap.add_argument('--archive',type=Path,default=ARCHIVE);args=ap.parse_args()
    if args.verify_only:verify(args.archive.resolve());return
    assert args.archive.resolve()==ARCHIVE,'External path selection is verification-only.'
    subprocess.run([sys.executable,str(ROOT/'tools/verify_documents.py')],check=True,stdout=subprocess.DEVNULL)
    checks=json.loads((ROOT/'reports/DOCUMENT_CHECKS.json').read_text())
    assert checks['visual_record_verified'] and checks['all_clean_build_pdf_bytes_identical'] and checks['outer_self_links']==0 and checks['new_independent_review_rounds']==0
    files=payloads()
    lines=['# Exact delivery manifest — SCI-FRUIT Stage B r0.4','','Status: complete review-candidate delivery; owner approval, freeze and activation pending. Numerical methods/routes remain unavailable_pending_separate_owner_approval; comparisons blocked; evidence artifacts not produced.','',f'Complete sole current Stage B review-candidate normative core: SCI-FRUIT-NORMATIVE-CORE v0.1/r0.4. Canonical source-inventory SHA-256: `{checks["core_source_inventory_sha256"]}`.','','Every regular payload appears once. ARTIFACT_MANIFEST.sha256 binds this manifest. The archive contains these payloads plus manifest/sidecar. The outer archive and archive-digest sidecar are external sibling artifacts, excluded from members and in-tree links. Task caches, build intermediates, rendered PNGs and Python caches are excluded. Historical input archives remain opaque bytes. Every member is a regular file with a safe relative path; no symlink or hard link is admitted.','','| Path | Role | Bytes | SHA-256 |','| --- | --- | ---: | --- |']
    for p in files:lines.append(f'| `{p.relative_to(ROOT).as_posix()}` | {role(p)} | {p.stat().st_size} | `{sha(p)}` |')
    MANIFEST.write_text('\n'.join(lines)+'\n');SIDE.write_text(f'{sha(MANIFEST)}  {MANIFEST.name}\n')
    with ARCHIVE.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=EPOCH) as gz:
            with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT) as tf:
                for p in sorted(files+[MANIFEST,SIDE]):
                    content=p.read_bytes();info=tarfile.TarInfo('r0.4/'+p.relative_to(ROOT).as_posix());info.size=len(content);info.mode=0o644;info.mtime=EPOCH;tf.addfile(info,io.BytesIO(content))
    ASIDE.write_text(f'{sha(ARCHIVE)}  {ARCHIVE.name}\n');verify(ARCHIVE)
if __name__=='__main__':main()
