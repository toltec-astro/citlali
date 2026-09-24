#!/usr/bin/env python3
"""Freeze and execute15 independent saved-Apply pairs; no adaptive execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import numpy as np
import yaml
from point_frozen_shoulder import *
from run_point_response_cd import source
from point_fruit_trajectory import bound, verified, manifest


def freeze(out):
    began=time.perf_counter();revision=source();require(not out.exists(),'fresh output required')
    auth,owner=ancestors();out.mkdir(parents=True);f=out/'freeze';f.mkdir()
    work=dict(schema='point-frozen-shoulder-v1',source_revision=revision,CD_root=str(CD),CD_manifest_sha256=CD_HASH,
        snapshots=list(SNAPSHOTS),arrays=list(ARRAYS),native_execution_source='26af74ac5e59d88a99da1ed4047a7a71a327fca0',
        native_semantics='unchanged ptc_response, saved row-major basis, full observed-entry mask, numerical means held fixed; fresh coefficient solve only',
        numerical_tolerances=TOL,components=list(PRIMARY),mean_centered=list(CENTERED),reverse_T10=list(REVERSE),
        partitions=['retained-brightness','added-support','removed-support'],workers=3,library_threads=1,
        no_observation_learning=True,no_fits=True,no_admission_updates=True,
        figures=['five-snapshot signed projections','T10 signed components/distortion/closure','applied masks/shoulder contours/profiles/linear apertures'],
        identities={},pairs=[],domains='exact C+D COMMON and finite KNOWN, no cropping PTC input; comparison grid/support/norm exact',
        directive=bound(Path(__file__).resolve().parents[2]/'validation/point_152389_frozen_ptc_shoulder_2026-09-24/OWNER_DIRECTIVE.md'))
    work['apply_source_equivalence']={}
    repo=Path(__file__).resolve().parents[2]
    import hashlib
    for path in ['src/citlali/core/pipeline/timestream_ptc_numerics.cpp','include/citlali/core/pipeline/timestream_ptc_numerics.h','src/citlali/core/pipeline/fruit_cycle.cpp','src/citlali/cli/fruit_saved_execution.h']:
        old=subprocess.check_output(['git','show',work['native_execution_source']+':'+path],cwd=repo)
        current=(repo/path).read_bytes();require(current==old,'changed saved Apply semantics '+path)
        work['apply_source_equivalence'][path]=hashlib.sha256(current).hexdigest()
    work['zero_learning_proof']='Harness call graph invokes prepare/ptc_response only. Source-equivalent ptc_response -> private apply does not invoke ptc_learn; no observation Learn/admission/source-fit entry called.'
    for array in ARRAYS:
        ctx=saved_context(auth,owner,array)
        summaries={b:auth.json(CD/f'run/{branch}/{array}/SUMMARY.json') for b,branch in [('R','REF'),('H','C-shoulder')]}
        for k in SNAPSHOTS:
            dest=f/array/f'{k:02d}';dest.mkdir(parents=True)
            pair=dict(array=array,cycle=k,segments=[])
            try:
                receipts={};maps={};headers={};products={}
                for b in ('R','H'):
                    c=summaries[b]['histories'][k];request=auth.json(c['request']['path']);receipt=auth.yaml(c['receipt']['path']);products[b]=Path(c['receipt']['path']).parent
                    require(receipt['cycle_index']==k and receipt['source_revision'].endswith('g26af74ac5'),'foreign saved execution')
                    ledger=auth.yaml(products[b]/'input-manifest.yaml')
                    for p,h in ledger.items():auth.bind(p,h)
                    receipts[b]=receipt;maps[b],headers[b]=auth.fits(c['total']['path'])
                    pair[b]=dict(total=c['total'],receipt=c['receipt'],request=c['request'],measurement=c['measurement'])
                    if k:
                        previous=summaries[b]['histories'][k-1];require(request['candidate']==previous['candidate'],'applied candidate index mismatch')
                        auth.check(request['candidate']['path']);auth.check(previous['total']['path'])
                        pair[b].update(applied_candidate=request['candidate'],previous=previous)
                    for key in ('operator-pixel.i32','output-rows.u64'):auth.check(products[b]/key)
                for n in ['SCI_SUPPORT','NORM_SUPPORT','NORMALIZATION','HITS']:require(np.array_equal(maps['R'][n],maps['H'][n]),'MAP support/normalization incompatible')
                for n in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','RADESYS','EQUINOX']:require(headers['R'][n]==headers['H'][n],'WCS incompatible')
                op=np.fromfile(products['R']/'operator-pixel.i32',dtype='<i4');oph=np.fromfile(products['H']/'operator-pixel.i32',dtype='<i4');require(np.array_equal(op,oph),'MAP operator incompatible')
                require(np.array_equal(np.fromfile(products['R']/'output-rows.u64',dtype='<u8'),np.fromfile(products['H']/'output-rows.u64',dtype='<u8')),'MAP output rows incompatible')
                shape=maps['R']['PRIMARY'].shape;npix=int(np.prod(shape));counts=np.bincount(op[op>=0],minlength=npix).reshape(shape)
                require(np.array_equal(counts,maps['R']['HITS']) and np.array_equal(counts,maps['R']['NORMALIZATION']),'uniform MAP identity')
                pair['shape']=list(shape);pair['comparison']=auth.ref(CD/f'report/{array}/C-shoulder/cycle-{k:02d}.fits');pair['comparison_record']=auth.ref(CD/f'report/{array}/C-shoulder/cycle-{k:02d}.json')
                require(len(receipts['R']['segments'])==len(receipts['H']['segments']),'segment grouping')
                for sr,sh in zip(receipts['R']['segments'],receipts['H']['segments']):
                    s=sr['source_segment'];require(s==sh['source_segment'] and sr['source_root']==sh['source_root'],'parent segment differs')
                    require(sr['converged']==sh['converged'] and sr['rank']==sh['rank'] and sr['input_eligibility_changed']==sh['input_eligibility_changed']==0,'PTC state/support transition')
                    src=Path(sr['source_root']);pr=auth.yaml(src/'receipt.yaml');require(pr['method']=='observed-als' and pr['rank']==10,'PTC recipe')
                    stem=next(n for n in s['files'] if n.endswith('-input.f64')).removesuffix('-input.f64');name=f"network-{s['network']}-{stem}"
                    rd,hd=products['R']/name,products['H']/name;nt,nd,no=s['scheduled_times'],s['detectors'],s['output_detectors'];columns=s['fit_columns_in_output']
                    require(nd>0 and s['basis_rows']==nd and s['basis_columns']==10,'missing saved basis shape')
                    cr=np.fromfile(auth.check(rd/'causes.u8'),dtype='u1').reshape(nt,no);ch=np.fromfile(auth.check(hd/'causes.u8'),dtype='u1').reshape(nt,no);require(np.array_equal(cr,ch),'Apply cause mismatch')
                    require(np.all(cr[:,[j for j in range(no) if j not in columns]]==17),'nonfit cause mismatch')
                    pix=np.full((nt,no),-1,dtype='<i4');retained=cr==0;n=int(retained.sum());require(n==sr['retained']==sh['retained'],'retained count')
                    if n:
                        first=sr['MAP_first_input_row'];require(first==sh['MAP_first_input_row'],'MAP input indexing');pix[retained]=op[first:first+n]
                    p=dest/(name+'-pixel.i32');np.ascontiguousarray(pix[:,columns],dtype='<i4').tofile(p)
                    inputs={n:auth.ref(path) for n,path in dict(y=src/(stem+'-input.f64'),mask=src/(stem+'-eligible.u8'),u=hd/'injection.f64',vr=rd/'operand.f64',vh=hd/'operand.f64',mr=rd/'mean.f64',mh=hd/'mean.f64',br=rd/'basis.f64',bh=hd/'basis.f64',cleanr=rd/'cleaned.f64',cleanh=hd/'cleaned.f64',causer=rd/'causes.u8',causeh=hd/'causes.u8').items()}
                    inputs['pixel']=bound(p)
                    mask=np.fromfile(auth.check(src/(stem+'-eligible.u8')),dtype='u1').reshape(nt,nd).astype(bool)
                    for side in (rd,hd):
                        classes=np.fromfile(auth.check(side/'model-support.u8'),dtype='u1').reshape(nt,nd)
                        require(np.all(classes[mask]!=0),'unsupported model projection in PTC domain')
                        if k:
                            operand=np.fromfile(auth.check(side/'operand.f64'),dtype='<f8').reshape(nt,nd)
                            require(np.array_equal(operand[mask]!=0,classes[mask]==3),'operand support not equivalent to applied positive candidate mask')
                    request=dict(schema='point-frozen-apply-v1',nt=nt,nd=nd,no=no,pixels=npix,rank=10,columns=columns,rank_tolerance=pr['relative_rank_tolerance'],converged_R=sr['converged'],converged_H=sh['converged'],reverse=k==10,inputs=inputs)
                    request_path=dest/(name+'.yaml');request_path.write_text(yaml.safe_dump(request,sort_keys=False));pair['segments'].append(dict(name=name,request=bound(request_path),network=s['network'],scan=s['scan'],inherited_failed=not sr['converged']))
                pair['status']='ready'
            except (ValueError,RuntimeError,KeyError,AssertionError) as exc:pair.update(status='unavailable',reason=str(exc))
            work['pairs'].append(pair)
    work['identities']=auth.consumed;work['freeze_seconds']=time.perf_counter()-began;work['verification_seconds']=auth.seconds
    save_json(f/'prospective.json',work);manifest(f);save_json(out/'FREEZE.json',dict(prospective=bound(f/'prospective.json'),manifest=bound(f/'FILES_SHA256SUMS')))
    print(json.dumps(dict(stage='freeze',ready=sum(p['status']=='ready' for p in work['pairs']),unavailable=[(p['array'],p['cycle'],p.get('reason')) for p in work['pairs'] if p['status']!='ready'],seconds=work['freeze_seconds'])),flush=True)


def run_pair(pair,work,out,binary):
    started=time.perf_counter();dest=out/'pairs'/pair['array']/f"{pair['cycle']:02d}";dest.mkdir(parents=True)
    if pair['status']!='ready':save_json(dest/'result.json',pair);return pair
    result=dict(array=pair['array'],cycle=pair['cycle'],status='running',native=[],checks={});auth=Auth()
    for p,h in work['identities'].items():auth.bind(p,h)
    try:
        # Authentication is repeated for every file actually loaded by this pair.
        owner=auth.json(CD/'freeze/prospective.json');ctx=saved_context(auth,owner,pair['array']);norm,header=auth.fits(pair['R']['total']['path']);other,_=auth.fits(pair['H']['total']['path'])
        comparison,_=auth.fits(pair['comparison']['path']);saved=auth.json(pair['comparison_record']['path']);domain=comparison['COMMON'].astype(bool)&np.isfinite(comparison['KNOWN']);known=comparison['KNOWN'];delta=comparison['DELTA']
        sums={};native_dir=dest/'native';native_dir.mkdir();native_wall=0
        for segment in pair['segments']:
            req=verified(segment['request']);q=read_yaml(req)
            for ref in q['inputs'].values():
                if ref['path'] in work['identities']:auth.check(ref['path'])
                else:verified(ref)
            scratch=dest/('working-'+segment['name']);command=[str(binary),str(req),str(scratch)]
            t=time.perf_counter();p=subprocess.run(command,capture_output=True,text=True);native_wall+=time.perf_counter()-t
            record=dict(segment=segment['name'],command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
            save_json(native_dir/(segment['name']+'-invocation.json'),record)
            require(p.returncode==0,'native saved Apply failed '+segment['name']+': '+p.stderr)
            receipt=read_yaml(scratch/'receipt.yaml');shutil.copyfile(scratch/'receipt.yaml',native_dir/(segment['name']+'.yaml'))
            v=np.fromfile(scratch/'numerators.f64',dtype='<f8').reshape(len(receipt['components']),*pair['shape'])
            for n,a in zip(receipt['components'],v):
                if n not in sums:sums[n]=a.copy()
                else:sums[n]+=a
            result['native'].append(dict(segment=segment['name'],**receipt));shutil.rmtree(scratch)
        diagnostics_began=time.perf_counter()
        support=norm['SCI_SUPPORT'].astype(bool);normalization=norm['NORMALIZATION'];maps={n:np.divide(v,normalization,out=np.full(v.shape,np.nan),where=support) for n,v in sums.items()}
        result['checks']['self_map_R']=close(maps['SELF_R'][support],norm['PRIMARY'][support],'self total R',1e-9)
        result['checks']['self_map_H']=close(maps['SELF_H'][support],other['PRIMARY'][support],'self total H',1e-9)
        result['checks']['self_numerator_R']=close(sums['SELF_R'],norm['NUMERATOR'],'self numerator R',1e-7)
        result['checks']['self_numerator_H']=close(sums['SELF_H'],other['NUMERATOR'],'self numerator H',1e-7)
        result['checks']['injection_map']=close(maps['IDEAL'][domain],known[domain],'independent injection-only map')
        maps['DELTA']=delta;maps['KNOWN']=known
        for name,components in [('CLOSURE',PRIMARY),('CENTERED_CLOSURE',CENTERED),('THREE_CLOSURE',('D','F','S'))]+([('REVERSE_CLOSURE',REVERSE)] if pair['cycle']==10 else []):
            v=sum(maps[n] for n in components);maps[name]=v-delta;result['checks'][name]=close(v[domain],delta[domain],name)
        for prefix,target in [('F','F'),('FC','FC')]:result['checks'][prefix+'_partition']=close(sum(maps[prefix+'_'+n] for n in ('RETAIN','ADD','REMOVE'))[domain],maps[target][domain],prefix+' map partition')
        if pair['cycle']==0:result['checks']['M0_zero_F']=close(maps['F'][domain],np.zeros(domain.sum()),'M0 F',0,0)
        result['measurements']={n:map_measures(v,known,domain,ctx,header) for n,v in maps.items() if not n.endswith('CLOSURE') and not n.startswith('SELF')}
        result['checks']['saved_g']=close(result['measurements']['DELTA']['g'],saved['map_response']['g'],'saved adaptive g',1e-10)
        result['checks']['saved_eta']=close(result['measurements']['DELTA']['eta'],saved['map_response']['eta'],'saved adaptive eta',1e-10)
        result['checks']['linear_sum']={m:close(sum(result['measurements'][n]['linear'][m] for n in PRIMARY),result['measurements']['DELTA']['linear'][m],'linear '+m) for m in result['measurements']['DELTA']['linear']}
        g=result['measurements']['DELTA']['g'];maps['ORTHOGONAL']=delta-g*known
        den=float(np.sum(known[domain]**2));result['cross_inner_products']={a:{b:float(np.sum(maps[a][domain]*maps[b][domain])/den) for b in PRIMARY} for a in PRIMARY}
        cb=pair['H']['measurement']['fit']['parameters'][6]-pair['R']['measurement']['fit']['parameters'][6]
        background=dict(constant_difference=cb,terms={})
        linear=result['measurements']['DELTA']['linear']
        for name,value in linear.items():
            if name=='template':weight=float(ctx['q'].sum());actual=pair['H']['measurement']['frozen_template_projection']-pair['R']['measurement']['frozen_template_projection']
            else:
                i=['aperture_1','aperture_1p5','aperture_2'].index(name);factor=[1,1.5,2][i];weight=float(np.count_nonzero(ctx['radius']<=factor*ctx['major'])*4);actual=pair['H']['measurement']['apertures'][i]['value_mJy_per_beam_arcsec2']-pair['R']['measurement']['apertures'][i]['value_mJy_per_beam_arcsec2']
            term=-weight*cb;check=close(value+term,actual,'saved background-subtracted '+name);background['terms'][name]=dict(raw_difference=value,background_term=term,saved_estimator_difference=actual,check=check)
        result['background']=background;result['admission']=candidate_diagnostics(auth,pair,ctx,dest)
        t=time.perf_counter();write_fits(dest/'components.fits',header,[(n,v,'mJy/nominal beam') for n,v in maps.items()]+[('COMMON',domain.astype('u1'),'1'),('SCI_SUPPORT',support.astype('u1'),'1'),('NORMALIZATION',normalization,'1')],companion='POINT-FROZEN-PTC')
        result['diagnostics_seconds']=t-diagnostics_began
        result.update(status='pass',serialization_seconds=time.perf_counter()-t,native_wall_seconds=native_wall,loading_verification_seconds=auth.seconds,wall_seconds=time.perf_counter()-started)
    except (ValueError,RuntimeError,KeyError,AssertionError) as exc:result.update(status='unavailable',reason=str(exc),wall_seconds=time.perf_counter()-started)
    save_json(dest/'result.json',result);print(json.dumps(dict(array=pair['array'],cycle=pair['cycle'],status=result['status'],reason=result.get('reason'),seconds=result['wall_seconds'])),flush=True);return result


def run(out,binary):
    start=time.perf_counter();revision=source();require(not (out/'pairs').exists(),'fresh execution required');freeze=json.loads((out/'FREEZE.json').read_text());work=json.loads(verified(freeze['prospective']).read_text())
    from point_fruit_trajectory import verify_manifest
    verify_manifest(out/'freeze',freeze['manifest']['sha256']);version=subprocess.check_output([str(binary),'--version'],text=True).strip();require(revision[:9] in version and 'dirty' not in version,'native/source mismatch')
    auth=Auth()
    for p,h in work['identities'].items():auth.bind(p,h);auth.check(p)
    before=auth.seconds
    # Three arrays parallel, each full segment-bounded evaluation sequential; no nested BLAS threads.
    for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):require(os.environ.get(n)=='1','set numerical threads to1')
    def array_run(a):return [run_pair(p,work,out,binary) for p in work['pairs'] if p['array']==a]
    with ThreadPoolExecutor(max_workers=3) as pool:results=sum(list(pool.map(array_run,ARRAYS)),[])
    after=auth.verify_all();verify_manifest(out/'freeze',freeze['manifest']['sha256']);save_json(out/'RUN.json',dict(source_revision=revision,binary=bound(binary),version=version,workers=3,library_threads=1,available_logical_cpus=os.cpu_count(),before_verification_seconds=before,after_verification_seconds=after,wall_seconds=time.perf_counter()-start,passed=sum(r['status']=='pass' for r in results),unavailable=[dict(array=r['array'],cycle=r['cycle'],reason=r.get('reason')) for r in results if r['status']!='pass'],native_learn_calls=0,admission_updates=0,source_fits=0,all_snapshots_independent=True,original_m10_unapplied=True,parent_peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_peak_rss_unit='bytes on macOS; KiB on Linux',memory_scope='individual native process and parent high-water marks separately; aggregate memory not measured'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path);args=p.parse_args()
    if args.action=='freeze':freeze(args.output)
    else:require(args.binary is not None,'binary required');run(args.output,args.binary)
