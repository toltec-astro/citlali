#!/usr/bin/env python3
"""Verify completed 1/2/4/8/12 runs without executing a reduction.

Each document is parsed once per saved product. Atomic document records permit
resumption after interruption; reuse always rechecks the input's SHA256. This
is a concrete campaign verification product, not a scientific data cache.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import yaml
import compare_successor_outputs as comparator

SETTINGS=(1,2,4,8,12)
EXTENSIONS={'.f64','.i64','.u8','.u16','.yaml'}
SCHEMA='successor-completed-campaign-verification-v1'


def require(ok,reason):
    if not ok:raise ValueError(reason)


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def identity(path):
    s=path.stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def sealed_record(path,value=None):
    """Detect damaged verification checkpoints independently of input hashes."""
    if value is None:
        envelope=json.loads(path.read_text());value=envelope['record']
        require(envelope['sha256']==hashlib.sha256(json.dumps(value,sort_keys=True,
                separators=(',',':'),allow_nan=False).encode()).hexdigest(),'checkpoint integrity failed: '+str(path))
    else:
        atomic_json(path,dict(record=value,sha256=hashlib.sha256(json.dumps(value,sort_keys=True,
                    separators=(',',':'),allow_nan=False).encode()).hexdigest()))
    return value


def tool_binding():
    return dict(verifier_sha256=comparator.sha(Path(__file__)),
                comparator_sha256=comparator.sha(Path(comparator.__file__)),
                python=sys.version,PyYAML=yaml.__version__,libyaml=yaml.__with_libyaml__)


def inventory(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())


def snapshot(root,output):
    """Freeze a verification record; a changed previously checked input stops."""
    root=Path(root);output=Path(output)
    require(not output.resolve().is_relative_to(root.resolve()),'verification output inside scientific product')
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();all_files=inventory(root)
    selected=[r for r in all_files if Path(r).suffix in EXTENSIONS]
    require(selected,'empty scientific product: '+str(root))
    header=dict(schema=SCHEMA,root=str(root),resolved_root=str(root.resolve()),
                tools=tool_binding(),all_files=all_files,scientific_files=selected,
                cal_receipt_sha256=comparator.sha(root/'donor-continuity/cal/receipt.yaml')
                    if (root/'donor-continuity/cal/receipt.yaml').is_file() else None)
    binding=output/'binding.json'
    if binding.exists():
        require(json.loads(binding.read_text())==header,'checkpoint source/tool/inventory changed: '+str(output))
    else:atomic_json(binding,header)
    records=[];reused=0;states={};binary=[]
    binary_checkpoint=output/'binary.json'
    previous_binary={r['path']:r for r in sealed_record(binary_checkpoint)} if binary_checkpoint.exists() else {}
    for number,rel in enumerate(selected,1):
        path=root/rel;before=identity(path);states[rel]=before;digest=comparator.sha(path)
        checkpoint=output/'yaml'/(hashlib.sha256(rel.encode()).hexdigest()+'.json')
        prior=sealed_record(checkpoint) if path.suffix=='.yaml' and checkpoint.exists() else previous_binary.get(rel)
        if prior is not None:
            row=prior
            require(row['path']==rel and row['sha256']==digest and row['bytes']==before[2],
                    'checkpoint input content changed: '+str(path))
            reused+=1
        else:
            row=dict(path=rel,bytes=before[2],sha256=digest)
            if path.suffix=='.yaml':
                atomic_json(output/'progress.json',dict(completed=number-1,total=len(selected),
                    current_file=rel,phase='parsing',reused=reused,seconds=time.monotonic()-started))
                print(json.dumps(dict(product=str(root),parsing=rel,bytes=before[2])),flush=True)
                parsed=comparator.Parser(path,root,retain_subtrees=False).parse()
                row['yaml']={k:v for k,v in parsed.items() if k!='subtrees'}
            require(identity(path)==before,'input changed while verified: '+str(path))
            if path.suffix=='.yaml':sealed_record(checkpoint,row)
        require(identity(path)==before,'input changed while checkpoint rechecked: '+str(path))
        records.append(row)
        if path.suffix!='.yaml':binary.append(row)
        if path.suffix=='.yaml' or number%500==0 or number==len(selected):
            atomic_json(output/'progress.json',dict(completed=number,total=len(selected),last_file=rel,
                reused=reused,seconds=time.monotonic()-started))
            print(json.dumps(dict(product=str(root),checked=number,total=len(selected),last_file=rel)),flush=True)
    require(inventory(root)==all_files,'product inventory changed during verification: '+str(root))
    require(all(identity(root/r)==s for r,s in states.items()),'input changed before snapshot completion')
    # One binary record per product, avoiding thousands of tiny checkpoint files.
    sealed_record(binary_checkpoint,binary)
    result=dict(header,artifacts=records,complete=True,seconds=time.monotonic()-started,
                checkpoints_reused=reused,subtree_details='not retained; complete semantic digest evaluated')
    atomic_json(output/'snapshot.json',result)
    return result


def compare_snapshots(baseline,candidate):
    require(baseline['tools']==candidate['tools']==tool_binding(),'comparison tool mismatch')
    require(baseline['complete'] and candidate['complete'],'incomplete snapshot')
    require(baseline['scientific_files']==candidate['scientific_files'],'scientific file inventory differs')
    require(set(baseline['all_files'])<=set(candidate['all_files']),'baseline ancillary file missing')
    old={r['path']:r for r in baseline['artifacts']};new={r['path']:r for r in candidate['artifacts']}
    rows=[]
    for rel in baseline['scientific_files']:
        a,b=old[rel],new[rel]
        if Path(rel).suffix=='.yaml':
            equal=a['yaml']['semantic_sha256']==b['yaml']['semantic_sha256']
            kind='exact existing YAML semantics'
        else:equal=a['sha256']==b['sha256'];kind='bitwise binary'
        rows.append(dict(path=rel,equal=equal,kind=kind))
    return dict(schema=SCHEMA,baseline=baseline['root'],candidate=candidate['root'],
                tools=tool_binding(),pass_equivalence=all(r['equal'] for r in rows),files=rows)


def verify_network(campaign,output,network,roots):
    began=time.monotonic();baseline=None;results=[]
    for setting in SETTINGS:
        root=Path(roots[str(setting)])/'products'/f'network{network}'
        destination=output/'snapshots'/f'workers-{setting}'/f'network{network}'
        current=snapshot(root,destination)
        if setting==1:baseline=current;continue
        result=compare_snapshots(baseline,current)
        result.update(network=network,workers=setting,
                      baseline_snapshot_sha256=comparator.sha(output/'snapshots/workers-1'/f'network{network}/snapshot.json'),
                      candidate_snapshot_sha256=comparator.sha(destination/'snapshot.json'))
        atomic_json(output/'comparisons'/f'1-{setting}'/f'network{network}.json',result)
        results.append(dict(workers=setting,pass_equivalence=result['pass_equivalence']))
        require(result['pass_equivalence'],f'scientific output mismatch: network{network}, workers{setting}')
    return dict(network=network,seconds=time.monotonic()-began,comparisons=results)


def preflight(campaign,expected):
    """Bind actual completed runs, preserving their exact publication path spelling."""
    roots={};networks=None;reference_identity=None
    for setting in SETTINGS:
        run=campaign/f'workers-{setting}';key=str(setting)
        for name in ('RUN.json','products/observation-receipt.yaml'):
            require(comparator.sha(run/name)==expected['runs'][key][name],
                    f'completed run binding changed: workers{setting}/{name}')
        metrics=json.loads((run/'RUN.json').read_text())
        require(metrics['workers']==setting and metrics['exit_code'] in (0,2),'completed run missing or failed')
        require(metrics['executable_sha256']==expected['executable_sha256'],'runtime binary identity differs')
        published=Path(metrics['command'][1]).parent
        require(published.resolve()==run.resolve(),'run publication path does not resolve to preserved run')
        roots[key]=str(published)
        receipt=yaml.load((run/'products/observation-receipt.yaml').read_text(),Loader=yaml.CSafeLoader)
        require(receipt['source_revision']==expected['runtime_revision'][:9],'runtime source differs')
        require(receipt['exit_code']==metrics['exit_code'],'completion disposition mismatch')
        require(receipt['requested_network_workers']==setting and receipt['requested_internal_threads']==1 and
                receipt['observed_Eigen_threads']==1,'thread binding mismatch')
        ids=[n['network'] for n in receipt['networks']]
        signatures=[dict(network=n['network'],input=n['input'],exit_code=n['exit_code'],state=n['state']) for n in receipt['networks']]
        require(all(n['output']==str(published/'products'/f"network{n['network']}") for n in receipt['networks']),
                'network output publication binding differs')
        require(all(n['exit_code'] in (0,2) for n in receipt['networks']),'failed network in completed run')
        if networks is None:networks=ids;reference_identity=signatures
        require(ids==networks==expected['networks'] and signatures==reference_identity,'population/input/disposition differs')
        require(receipt['started_network_workers']==min(setting,len(ids)),'started worker count differs')
    return networks,roots


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output','expected'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,13),default=6)
    a=p.parse_args();expected=json.loads(a.expected.read_text());networks,roots=preflight(a.campaign,expected)
    a.output.mkdir(parents=True,exist_ok=True)
    binding=dict(schema=SCHEMA,campaign=str(a.campaign.resolve()),expected_sha256=comparator.sha(a.expected),
                 tools=tool_binding(),networks=networks,publication_roots=roots)
    bound=a.output/'binding.json'
    if bound.exists():require(json.loads(bound.read_text())==binding,'verification continuation binding changed')
    else:atomic_json(bound,binding)
    report=dict(binding,state='in-progress',verification_workers=a.workers,networks_completed=[],failures=[])
    atomic_json(a.output/'STATUS.json',report)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(verify_network,a.campaign,a.output,n,roots):n for n in networks}
        for future in as_completed(jobs):
            try:report['networks_completed'].append(future.result())
            except Exception as e:report['failures'].append(dict(network=jobs[future],reason=str(e)))
            atomic_json(a.output/'STATUS.json',report)
    report['state']='FAIL' if report['failures'] else 'PASS'
    atomic_json(a.output/'STATUS.json',report)
    return 1 if report['failures'] else 0


if __name__=='__main__':raise SystemExit(main())
