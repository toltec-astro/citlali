#!/usr/bin/env python3
"""Bind exact observation assets, optionally reusing owner-resident raw/TEL files."""
import argparse
import json
from pathlib import Path
import shutil
from ptc_spod import digest, require
from prepare_successor_networks import SUPPLIED


def relative_path(value):
    path=Path(value)
    require(bool(value) and not path.is_absolute() and '..' not in path.parts,
            'expected a contained relative input path: '+value)
    return path


def verify_existing(packet,existing_root):
    """Read-only identity check; symlink targets are recorded, never rewritten."""
    manifest=packet/'EXISTING_INPUTS.json'
    if not manifest.exists():return {}
    require(existing_root is not None,'--existing-root is required for resident inputs')
    entries=json.loads(manifest.read_text())
    require(entries['schema']=='successor-resident-inputs-v1','unknown resident input schema')
    checked={}
    for entry in entries['files']:
        token=entry['token']
        require(token.startswith('@existing/') and token not in checked,'invalid/duplicate resident input token')
        rel=relative_path(entry['relative_path'])
        require(token=='@existing/'+str(rel),'resident token/path mismatch')
        path=existing_root/rel
        require(path.is_file(),'missing resident input: '+str(path))
        require(path.stat().st_size==entry['bytes'],'resident input size mismatch: '+str(path))
        require(digest(path)==entry['sha256'],'resident input SHA256 mismatch: '+str(path))
        checked[token]=dict(path=str(path.absolute()),resolved_path=str(path.resolve()),
                            sha256=entry['sha256'],bytes=entry['bytes'])
    return checked


def stage(reference,packet,existing_data_relative=None):
    base=json.loads(reference.read_text());require(base.get('native_source')=='exact-raw-kids-v1','direct-raw reference required')
    assets=packet/'assets';assets.mkdir(parents=True,exist_ok=True);mapping={}
    def copy_file(ref,destination):
        src=Path(ref['path']);require(digest(src)==ref['sha256'],'changed asset: '+str(src))
        dst=assets/destination;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        require(digest(dst)==ref['sha256'],'copy mismatch')
        mapping[str(src)]=str(dst.relative_to(packet))
    existing=[]
    def resident(ref):
        src=Path(ref['path']);require(digest(src)==ref['sha256'],'changed asset: '+str(src))
        rel=relative_path(existing_data_relative)/src.name
        token='@existing/'+str(rel);mapping[str(src)]=token
        existing.append(dict(token=token,relative_path=str(rel),sha256=ref['sha256'],bytes=src.stat().st_size))
    for r in base['decision_apply']['timing_inputs']:
        if existing_data_relative is None:copy_file(r,Path('raw')/Path(r['path']).name)
        else:resident(r)
    for nw in SUPPLIED:
        p=Path(base['tune']['path']);p=p.with_name(p.name.replace('toltec12_',f'toltec{nw}_',1))
        copy_file(dict(path=str(p),sha256=digest(p)),Path('tunes')/p.name)
    for key in ('telescope','effective_config','ast_acceptance'):
        if key=='telescope' and existing_data_relative is not None:resident(base[key])
        else:copy_file(base[key],Path(key)/Path(base[key]['path']).name)
    manifest=Path(base['manifest']['path']);require(digest(manifest)==base['manifest']['sha256'],'manifest changed')
    if not (assets/'apt').exists():shutil.copytree(manifest.parent,assets/'apt')
    for p in manifest.parent.rglob('*'):
        if p.is_file():require(digest(p)==digest(assets/'apt'/p.relative_to(manifest.parent)),'APT copy mismatch')
    mapping[str(manifest)]=str((assets/'apt'/manifest.name).relative_to(packet))
    # The reviewed array plan is recovered from the exact source at materialization.
    base.pop('filter_plan',None)
    copy_file(base['decision_apply']['processing_provenance'],Path('provenance')/'raw_timestream_provenance.yaml')
    def rebase(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='path':
                    require(v in mapping,'unmapped dependency: '+v);x[k]=mapping[v]
                else:rebase(v)
        elif isinstance(x,list):
            for v in x:rebase(v)
    rebase(base)
    (packet/'reference-template.json').write_text(json.dumps(base,indent=2)+'\n')
    files=sorted(p for p in assets.rglob('*') if p.is_file())
    (packet/'ASSETS_SHA256SUMS').write_text(''.join(digest(p)+'  '+str(p.relative_to(packet))+'\n' for p in files))
    (packet/'ASSET_ORIGINS.json').write_text(json.dumps(dict(reference=str(reference),reference_sha256=digest(reference),
        bound_paths=mapping,asset_bytes=sum(p.stat().st_size for p in files)),indent=2)+'\n')
    if existing_data_relative is not None:
        require(not list(assets.rglob('*.nc')),'resident-input packet must not contain NetCDF payloads')
        (packet/'EXISTING_INPUTS.json').write_text(json.dumps(dict(schema='successor-resident-inputs-v1',
            files=existing),indent=2)+'\n')


def materialize(packet,output,repository,existing_root=None):
    from prepare_successor_networks import prepare
    base=json.loads((packet/'reference-template.json').read_text())
    checked=verify_existing(packet,existing_root)
    def rebase(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='path':
                    if v.startswith('@existing/'):
                        require(v in checked,'unbound resident input: '+v);x[k]=checked[v]['path']
                    else:
                        p=(packet/relative_path(v)).resolve();require(p.is_relative_to(packet.resolve()),'asset path escapes packet');x[k]=str(p)
                else:rebase(v)
        elif isinstance(x,list):
            for v in x:rebase(v)
    rebase(base);ref=output.parent/(output.name+'-reference.json')
    with ref.open('x') as f:json.dump(base,f,indent=2);f.write('\n')
    return prepare(ref,output,repository)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='operation',required=True)
    a=s.add_parser('stage');a.add_argument('--reference',required=True,type=Path);a.add_argument('--packet',required=True,type=Path)
    a.add_argument('--existing-data-relative',help='raw/TEL directory relative to the owner-provided Unity data root; copy only auxiliary metadata')
    a=s.add_parser('materialize');a.add_argument('--packet',required=True,type=Path);a.add_argument('--output',required=True,type=Path)
    a.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[2])
    a.add_argument('--existing-root',type=Path)
    a=s.add_parser('verify-existing');a.add_argument('--packet',required=True,type=Path)
    a.add_argument('--existing-root',required=True,type=Path)
    a=p.parse_args()
    if a.operation=='stage':stage(a.reference,a.packet.resolve(),a.existing_data_relative)
    elif a.operation=='verify-existing':print(json.dumps(verify_existing(a.packet.resolve(),a.existing_root.resolve()),indent=2))
    else:materialize(a.packet.resolve(),a.output.resolve(),a.repository.resolve(),
                     a.existing_root.resolve() if a.existing_root else None)
