"""Bounded 152430 PTC-rank experiment; orchestration of shared POINT owners.

Only `run` starts science. `preflight` binds operands; `report` reads compact data.
No resume/retry, transfer, reference-basis reuse, or implicit reconstruction.
"""
import copy
import gzip
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from inspect_forward_map import require, digest, read_yaml
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound, verified
from point_fruit_diagnostics import load, patch_map, learned_state
from point_pipeline_inputs import prepare
from point_reference import initialize, context
from point_guarded_feedback import infer_base, measurement
from point_response_cd import processing_guard
from point_processing import (upstream, execute_branch, owner, request, call,
                              parents, cleanup_cycle, read, size)
from compare_point_pipeline import compare, available
from run_independent_point import initial_control

REPO=Path(__file__).resolve().parents[2]
REFERENCE=REPO/'data/development/bright_point_base_v1/reference.json'
OLD=REPO/'validation/point_152430_no_upper_coverage_veto_2026-09-26/compact'
FREEZE=REPO/'validation/fruit_pipeline_freeze_2026-09-26/compact'
BRANCHES=('REF','C-core-plus','C-core-minus','C-shoulder')
LIMITS=dict(parent=1,trajectory=12,feedback=120,noise=3,endpoint=3,ideal=9)
ABSOLUTE=dict(parent=2,trajectory=14,feedback=140,noise=5,endpoint=5,ideal=11)
COUNTER_LOCK=threading.Lock()


def consume(root,kind,identity):
    """Persist before work; failures consume the budget. No automatic replacements."""
    with COUNTER_LOCK:
        p=root/'attempts.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        require(kind in LIMITS,'unknown operation')
        require(sum(r['kind']==kind for r in rows)<LIMITS[kind],'planned attempt ceiling; repair needs explicit defect disposition')
        require(not any(r['kind']==kind and r['identity']==identity for r in rows),'duplicate operation; no implicit retry')
        with p.open('a') as f:
            f.write(json.dumps(dict(kind=kind,identity=identity,epoch=time.time()))+'\n');f.flush()


def preflight(root,input_root):
    require(not root.exists(),'fresh experiment output required')
    source=read(REFERENCE);root.mkdir(parents=True)
    # Retained operands are explicitly declared before any real-data processing.
    shapes=read(OLD/'SHAPES.json');d=root/'operands';d.mkdir()
    for branch in BRANCHES[1:]:
        p=d/(branch+'.fits');p.write_bytes(gzip.decompress((OLD/f'freeze/shapes/{branch}.fits.gz').read_bytes()))
        require(digest(p)==shapes['rasters'][branch]['sha256'],'retained injection identity mismatch')
    g=read(OLD/'freeze/a2000/geometry.json')
    with np.load(OLD/'freeze/a2000/context.npz') as z:
        geometry=dict(header=g['header'],parameters=z['parameters'].tolist(),major=g['major'],weights=z['q'].tolist())
    save_json(d/'measurement-geometry.json',geometry)
    save_json(d/'SHAPES.json',shapes)
    configs={}
    for rank in (10,5,15):
        out=root/f'rank-{rank}';prepare(out,REFERENCE,input_root)
        p=read(out/'freeze/prospective.json');p.update(schema='citlali-point-ptc-rank-v1',profile=f'point-152430-ptc-rank-{rank}')
        p['ptc']['rank']=rank;p['scientific_recipe']['PTC']['rank']=rank
        configs[str(rank)]=dict(ptc=p['ptc'],scientific_recipe=p['scientific_recipe'])
        save_json(out/'freeze/prospective.json',p)
        budget=read(out/'freeze/budget.json');cells=budget['native_cells']
        # One shared raw/RTC/CAL/PTC carrier, 9 immutable injected parents,
        # REF projection controls retained for 30 stages, at most 3 live cycles.
        budget.update(campaign_root=str(root),peak_ceiling_bytes=256*cells+9*16*cells+30*16*cells+3*48*cells+1_000_000_000,maximum_branch_workers=3)
        save_json(out/'freeze/budget.json',budget);p['budget']=bound(out/'freeze/budget.json')
        shape_dir=out/'freeze/shapes';shape_dir.mkdir()
        for branch in BRANCHES[1:]:shutil.copyfile(d/(branch+'.fits'),shape_dir/(branch+'.fits'))
        save_json(out/'SHAPES.json',shapes)
        p['experiment_operands']=[bound(x) for x in sorted(d.iterdir())]
        save_json(out/'freeze/prospective.json',p)
    # Expand generated binding hashes to their resolved contents before comparing.
    import yaml
    def resolved_inputs(out):
        def normalize(value):
            if isinstance(value,dict):
                if set(value)=={'path','sha256'} and Path(value['path']).is_relative_to(out/'inputs'):
                    target=Path(value['path']);verified(value)
                    return normalize(yaml.safe_load(target.read_text()))
                return {k:normalize(v) for k,v in value.items()}
            if isinstance(value,list):return [normalize(v) for v in value]
            if isinstance(value,str):return value.replace(str(out),'$RANK_OUTPUT')
            return value
        return {p.name:normalize(yaml.safe_load(p.read_text())) for p in sorted((out/'inputs').iterdir())}
    resolved=[resolved_inputs(root/f'rank-{rank}') for rank in (10,5,15)]
    # effective_config_sha256 is redundant with the expanded effective YAML binding.
    for v in resolved:
        def erase_digest(x):
            if isinstance(x,dict):
                x.pop('effective_config_sha256',None)
                for item in x.values():erase_digest(item)
            elif isinstance(x,list):
                for item in x:erase_digest(item)
        erase_digest(v)
    require(resolved[0]==resolved[1]==resolved[2],'resolved native input configuration differs scientifically')
    save_json(root/'RESOLVED-COMMON-INPUTS.json',resolved[0])
    normalized=[]
    for cfg in configs.values():
        c=copy.deepcopy(cfg);c['ptc']['rank']=10;c['scientific_recipe']['PTC']['rank']=10;normalized.append(c)
    require(all(c==normalized[0] for c in normalized),'scientific configuration differs beyond rank')
    # All resolved native input operands/configuration differ only by declared paths
    # and path-dependent digests. Their underlying source/config/RTC files are equal.
    p10=read(root/'rank-10/freeze/prospective.json')
    save_json(root/'PREFLIGHT.json',dict(source_revision=p10['source_revision'],base='5435bdbdb4e6708e884ae4cb2b0fd93e74102ce5',profiles=configs,scientific_difference_only=['ptc.rank'],rank_in_scientific_recipe='same fact recorded for audit',planned=LIMITS,absolute_including_documented_repairs=ABSOLUTE,resource_budget=budget,operands=[bound(p) for p in sorted(d.iterdir())],reference_tolerances=source['reference_tolerances'],array='a2000',networks=[11,12],concurrency=3,library_threads=1,upstream_generations=1,shared_parent='new raw-derived rank10 pre-PTC input carrier; its learned basis is not imported',geometry='retained physical aperture/template only; each rank learns own noise and source mask',runtime_inputs='original manifest objects and explicitly bound raster/geometry operands; old maps only in comparators'))


def initialize_rank(out,root):
    consume(root,'noise',dict(rank=read(out/'freeze/prospective.json')['ptc']['rank']))
    c,g=initialize(out/'M0/products/a2000/map.fits',out/'freeze/a2000')
    if c is not None:
        shutil.copyfile(root/'operands/measurement-geometry.json',out/'freeze/a2000/measurement-geometry.json')
        c=context(out/'freeze/a2000')
        save_json(out/'freeze/M0-measurement.json',measurement(c,c['original'],c['support'],g))
    save_json(out/'UPSTREAM.json',dict(status='accepted' if g['accepted'] else 'source_or_covariance_failed',guard=g,parent=parents(out),parent_generation=False))
    return c,g


def rank_receipts(out,products=None):
    """Compact network/segment population receipts, collected while arrays exist."""
    rank=read(out/'freeze/prospective.json')['ptc']['rank'];rows=[]
    if products is None:
        for nw in (11,12):
            pr=out/f'upstream/products/network{nw}/donor-continuity/ptc'
            r=read_yaml(pr/'receipt.yaml')
            require(r['rank']==rank,'native M0 requested rank differs')
            for s in r['segments']:
                require(s['converged'] and s['basis_columns']==rank,'native M0 rank/learning failed')
                rows.append(dict(network=nw,requested_rank=rank,realized_rank=s['basis_columns'],segment=s))
    else:
        r=read_yaml(products/'receipt.yaml')
        for s in r['segments']:
            require(s['rank']==rank and s['realized_rank']==rank and s['learning_performed'],'native configured rank not freshly learned')
            rows.append(dict(network=s['source_segment']['network'],requested_rank=rank,realized_rank=s['realized_rank'],segment=s))
    return rows


def run_ref(root,binary,rank):
    out=root/f'rank-{rank}'
    if rank==10:
        consume(root,'parent',dict(rank=10));consume(root,'trajectory',dict(rank=10,branch='REF',cycle=0))
        upstream(out,binary)
        require(read(out/'UPSTREAM.json')['status']=='accepted','rank10 initial source/noise gate failed')
        save_json(out/'M0-PTC.json',rank_receipts(out))
    else:
        # Files reference the original shared parent, with no copies of y or r.
        save_json(out/'freeze/parents.json',parents(root/'rank-10'))
        dest=out/'M0';dest.mkdir();op=dest/'owner.json';save_json(op,owner(out,'BASE-POINT','REF'))
        candidate_path=root/'rank-10/run/BASE-POINT/REF/a2000/initial/candidate.fits'
        q=dest/'request.json';save_json(q,request(out,op,'BASE-POINT','REF',0,candidate_path))
        require(call(out,binary,q,dest/'native')==0,'rank REF M0 processing failed')
        products=dest/'native/products';r=read_yaml(products/'receipt.yaml')
        processing_guard(products,r,allow_historical_failure=False,expected_rank=rank)
        save_json(out/'M0-PTC.json',rank_receipts(out,products))
        (dest/'products/a2000').mkdir(parents=True)
        shutil.copyfile(products/'total.fits',dest/'products/a2000/map.fits')
        shutil.copyfile(products/'receipt.yaml',dest/'products/attempt.yaml')
        c,g=initialize_rank(out,root)
        if c is None:return dict(ready=False,completed_cycle=0,stop_reason=g['message'])
        cleanup_cycle(out,products)
    result=execute_branch(out,binary,'BASE-POINT','REF')
    if result['completed_cycle']:
        p=out/f"run/BASE-POINT/REF/a2000/cycle-{result['completed_cycle']:02d}/native/products"
        save_json(out/'LAST-PTC.json',rank_receipts(out,p))
    return result


def run_injection(root,binary,rank,branch):
    out=root/f'rank-{rank}';dest=out/'initial'/branch;dest.mkdir(parents=True)
    c=context(out/'freeze/a2000');op=dest/'owner.json';save_json(op,owner(out,'BASE-POINT',branch))
    q=dest/'request.json';save_json(q,request(out,op,'BASE-POINT',branch,0,out/'run/BASE-POINT/REF/a2000/initial/candidate.fits'))
    require(call(out,binary,q,dest/'native')==0,'injected M0 native failure')
    products=dest/'native/products';receipt=read_yaml(products/'receipt.yaml')
    guards=processing_guard(products,receipt,allow_historical_failure=False,expected_rank=rank)
    s,v,_,_=patch_map(products/'total.fits');m,_=infer_base(c,s,v)
    response=initial_control(out/'M0/products/a2000/map.fits',products/'total.fits',out/f'freeze/shapes/{branch}.fits',c,dest)
    save_json(dest/'INITIAL.json',dict(available=True,total=bound(products/'total.fits'),receipt=bound(products/'receipt.yaml'),measurement=m,guards=guards,response=response))
    cleanup_cycle(out,products,keep_injected=True)
    return execute_branch(out,binary,'BASE-POINT',branch)


def compare_injection(out,branch):
    """Same strict plane/measurement tolerances as REF, without old absolute paths."""
    from astropy.io import fits
    route=Path(f'run/BASE-POINT/{branch}/a2000');a=read(out/route/'SUMMARY.json');b=read(OLD/route/'SUMMARY.json')
    rows=[];failures=[]
    require(a['completed_cycle']==b['completed_cycle']==10 and a['ready'],'rank10 injection incomplete')
    for k,(x,y) in enumerate(zip(a['histories'],b['histories'])):
        where=route/('initial' if k==0 else f'cycle-{k:02d}')
        paths=[(where/'candidate.fits',True),((Path(f'initial/{branch}/native/products/total.fits') if k==0 else where/'native/products/total.fits'),False)]
        for rel,discrete in paths:
            with fits.open(available(out/rel),memmap=False) as af,fits.open(available(OLD/rel),memmap=False) as bf:
                require([h.name for h in af]==[h.name for h in bf],'plane inventory differs')
                for aa,bb in zip(af,bf):
                    exact=discrete or aa.name not in ('PRIMARY','NUMERATOR')
                    ok=np.allclose(aa.data,bb.data,equal_nan=True,atol=0 if exact else 1e-7 if aa.name=='NUMERATOR' else 1e-9,rtol=0 if exact else 1e-11)
                    rows.append(dict(cycle=k,path=str(rel),plane=aa.name,pass_=bool(ok),exact=bool(np.array_equal(aa.data,bb.data,equal_nan=True))))
                    if not ok:failures.append(rows[-1])
        for field in ('measurement','admission','response'):
            if x.get(field)!=y.get(field):failures.append(dict(cycle=k,field=field))
        if k==10 and x.get('ideal')!=y.get('ideal'):failures.append(dict(cycle=k,field='ideal'))
    return dict(pass_=not failures,rows=rows,failures=failures)


def run(root,binary,review):
    from point_pipeline import validate_environment
    require(not (root/'EXECUTION.json').exists(),'campaign already attempted')
    p=read(root/'PREFLIGHT.json');review_data=read(review)
    require(review_data['source_revision']==p['source_revision'] and review_data['verdict']=='pass','independent preflight gate')
    identities=[validate_environment(root/f'rank-{rank}',binary) for rank in (10,5,15)]
    for b in p['operands']:verified(b)
    started=time.perf_counter();record=dict(status='running',identities=identities,preflight_review=bound(review));save_json(root/'EXECUTION.json',record)
    results={}
    def cell(rank,branch):
        try:
            result=run_ref(root,binary,rank) if branch=='REF' else run_injection(root,binary,rank,branch)
            status='completed' if result['ready'] and result['completed_cycle']==10 and (branch!='REF' or result.get('endpoint_status')=='completed') else 'stopped'
            return dict(rank=rank,branch=branch,status=status,completed_cycle=result['completed_cycle'],reason=result.get('stop_reason'))
        except Exception as exc:return dict(rank=rank,branch=branch,status='processing_failed',reason=str(exc))
    try:
        results['10/REF']=cell(10,'REF');save_json(root/'matrix-progress.json',results)
        require(results['10/REF']['status']=='completed','rank10 REF processing gate')
        regression=compare(root/'rank-10',FREEZE,'a2000',p['reference_tolerances']);save_json(root/'REGRESSION-REF.json',regression)
        require(regression['pass_'],'rank10 REF regression gate')
        # Establish all three rank10 injection controls before lower/higher-rank responses.
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={branch:pool.submit(cell,10,branch) for branch in BRANCHES[1:]}
            for branch,f in futures.items():results['10/'+branch]=f.result()
        save_json(root/'matrix-progress.json',results)
        for branch in BRANCHES[1:]:
            require(results['10/'+branch]['status']=='completed','rank10 injection processing gate')
            reg=compare_injection(root/'rank-10',branch);save_json(root/f'REGRESSION-{branch}.json',reg)
            require(reg['pass_'],'rank10 injection wiring regression gate')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={rank:pool.submit(cell,rank,'REF') for rank in (5,15)}
            for rank,f in futures.items():results[f'{rank}/REF']=f.result()
        save_json(root/'matrix-progress.json',results)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={(rank,branch):pool.submit(cell,rank,branch) for rank in (5,15) for branch in BRANCHES[1:] if results[f'{rank}/REF']['status']=='completed'}
            for (rank,branch),f in futures.items():results[f'{rank}/{branch}']=f.result()
        record['status']='completed_matrix'
    except Exception as exc:record.update(status='blocked_gate',reason=str(exc))
    for rank in (5,10,15):
        for branch in BRANCHES:results.setdefault(f'{rank}/{branch}',dict(rank=rank,branch=branch,status='blocked',reason='prerequisite REF/regression gate unavailable'))
    record.update(seconds=time.perf_counter()-started,matrix=list(results.values()));save_json(root/'matrix-progress.json',results);save_json(root/'EXECUTION.json',record)
    return 0 if record['status']=='completed_matrix' else 1
