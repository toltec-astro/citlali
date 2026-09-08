#!/usr/bin/env python3
"""Seal a document packet and optionally create a deterministic delivery archive.

SOURCE_MANIFEST.json covers every package file except itself and its sidecar.
The sidecar hashes SOURCE_MANIFEST.json. An external archive + sidecar covers
both without a self-hash cycle. Existing archives are never overwritten.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
from build_packet import CORE, CORE_ID, core_digest, sha


def seal(root):
    files={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).as_posix() not in ('SOURCE_MANIFEST.json','SOURCE_MANIFEST.sha256')}
    manifest={
        'identity':'SCI-MAP-v0.2-r0.2-FINAL-DOCUMENT-SOURCE-MANIFEST/2026-09-07',
        'status':'Exact candidate document bytes; not a scientific freeze or numerical-route activation',
        'shared_authority':{'identity':CORE_ID,'sha256':core_digest(root/'src'),'ordered_sources':{p:sha(root/'src'/p) for p in CORE},'algorithm':'For each ordered core path: UTF8(path), NUL, ASCII(raw byte length), NUL, raw bytes, NUL; SHA256 of concatenation.'},
        'files':files,
        'seal_rule':'All package files except this manifest and its SHA256 sidecar are bound above. The sidecar binds this manifest. The external delivery archive and archive sidecar bind both; no file claims its own hash.',
        'authority_limit':'Uniform literal @draft-0.1 and v0.2-draft.1/r0.4 identities retain their historical approvals. Dependent r0.2 source-binding and numerical route remain candidate; no source-closed frozen numerical route is asserted.'
    }
    p=root/'SOURCE_MANIFEST.json'; p.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    (root/'SOURCE_MANIFEST.sha256').write_text(sha(p)+'  SOURCE_MANIFEST.json\n')
    return manifest


def archive(root,path):
    if path.exists() or path.with_name(path.name+'.sha256').exists():
        raise SystemExit('Refusing to overwrite existing archive or sidecar')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=1788739200) as zipped, tarfile.open(fileobj=zipped,mode='w',format=tarfile.PAX_FORMAT) as tar:
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            blob=p.read_bytes(); name='SCI-MAP-v0.2-r0.2/'+p.relative_to(root).as_posix()
            info=tarfile.TarInfo(name);info.size=len(blob);info.mode=0o644;info.mtime=1788739200;info.uid=info.gid=0;info.uname=info.gname=''
            tar.addfile(info,io.BytesIO(blob))
    path.with_name(path.name+'.sha256').write_text(sha(path)+'  '+path.name+'\n')
    # Read archive members without extracting or mutating existing files.
    verified=0
    with tarfile.open(path,'r:gz') as tar:
        names=[]
        for member in tar.getmembers():
            assert member.isfile() and member.name.startswith('SCI-MAP-v0.2-r0.2/')
            rel=member.name.removeprefix('SCI-MAP-v0.2-r0.2/')
            assert '..' not in Path(rel).parts
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==sha(root/rel)
            names.append(rel);verified+=1
        assert len(names)==len(set(names))
        assert set(names)=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    return {'archive':str(path),'sha256':sha(path),'sidecar_sha256':sha(path.with_name(path.name+'.sha256')),'verified_members':verified}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',required=True);parser.add_argument('--archive');parser.add_argument('--archive-only',action='store_true')
    args=parser.parse_args();root=Path(args.package).resolve()
    if not args.archive_only:
        m=seal(root);print(json.dumps({'manifest_sha256':sha(root/'SOURCE_MANIFEST.json'),'bound_files':len(m['files']),'core_sha256':m['shared_authority']['sha256']}))
    if args.archive:print(json.dumps(archive(root,Path(args.archive).resolve())))
