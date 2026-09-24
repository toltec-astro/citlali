#!/usr/bin/env python3
"""Authenticate the fixed JINC comparison inputs and audit ordinary saved rejoin.

No JINC observation execution, PTC preparation/Learn/Apply, coordinate rounding,
fit or candidate generation. Saved numerical means remain untouched. This is
baseline recovery, not a numerical authorization or the primary comparison.
"""
import argparse
import json
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
from point_frozen_shoulder import ancestors,CD,ARRAYS,close
from point_fruit_trajectory import manifest,verify_manifest,bound
from point_additive_diagnostics import save_json
from inspect_forward_map import require,digest
from run_point_response_cd import source

MATRIX=(('REF',0),('REF',10),('C-core-plus',10),('C-core-minus',10),('C-shoulder',10))


def rejoined_rows(clean,operand,causes,columns):
    require(clean.shape==causes.shape and clean.shape[0]==operand.shape[0] and operand.shape[1]==len(columns),'saved segment shape')
    require(len(set(columns))==len(columns) and all(0<=c<clean.shape[1] for c in columns),'saved fit-column relation')
    restored=np.zeros_like(clean);restored[:,columns]=operand
    valid=causes==0
    require(np.isfinite(clean[valid]).all() and np.isfinite(restored[valid]).all(),'nonfinite admitted saved sample')
    return clean[valid],restored[valid]


def audit_snapshot(auth,array,branch,k,stage,out):
    began=time.perf_counter();products=Path(stage['receipt']['path']).parent;r=auth.yaml(products/'receipt.yaml')
    for p,h in auth.yaml(products/'input-manifest.yaml').items():auth.bind(p,h);auth.check(p)
    for p in products.rglob('*'):
        if p.is_file():auth.check(p)
    for key in ('request','candidate','total','receipt'):auth.check(stage[key]['path'])
    request=auth.json(stage['request']['path']);require(request['cycle_index']==k and request['branch']==branch and request['array']==array,'snapshot identity')
    maps,header=auth.fits(products/'total.fits');pixels=np.fromfile(auth.check(products/'operator-pixel.i32'),dtype='<i4');output=np.fromfile(auth.check(products/'output-rows.u64'),dtype='<u8')
    shape=maps['PRIMARY'].shape;npix=maps['PRIMARY'].size
    sums={n:np.zeros(npix) for n in ('total','clean','operand','normalization')};segments=[];cursor=0
    for s in r['segments']:
        original=s['source_segment'];stem=next(p.removesuffix('-input.f64') for p in original['files'] if p.endswith('-input.f64'));name=f"network-{original['network']}-{stem}"
        nt,no,nd=[original[n] for n in ('scheduled_times','output_detectors','detectors')];columns=original['fit_columns_in_output']
        def read(n,dtype,shape):return np.fromfile(auth.check(products/name/n),dtype=dtype).reshape(shape)
        clean=read('cleaned.f64','<f8',(nt,no));operand=read('operand.f64','<f8',(nt,nd));causes=read('causes.u8','u1',(nt,no))
        z,v=rejoined_rows(clean,operand,causes,columns);require(len(z)==s['retained'],'saved retained count')
        require(s.get('MAP_first_input_row',cursor)==cursor,'saved row indexing')
        p=pixels[cursor:cursor+len(z)];require(len(p)==len(z) and np.all((p>=-1)&(p<npix)),'saved MAP pixel domain');good=p>=0
        require(int(good.sum())==s.get('MAP_admitted',0),'saved MAP admission')
        for key,values in [('total',z+v),('clean',z),('operand',v),('normalization',np.ones(len(z)))]:np.add.at(sums[key],p[good],values[good])
        segments.append(dict(name=name,retained=len(z),map_contributions=int(good.sum()),source=s['source_root'],source_segment=original,
            cleaned=auth.ref(products/name/'cleaned.f64'),operand=auth.ref(products/name/'operand.f64'),causes=auth.ref(products/name/'causes.u8'),
            saved_mean=auth.ref(products/name/'mean.f64'),saved_basis=auth.ref(products/name/'basis.f64')));cursor+=len(z)
    require(cursor==len(pixels)==r['MAP_input_rows'],'complete saved rows')
    checks={};norm=sums['normalization'].reshape(shape)
    require(np.array_equal(norm,maps['NORMALIZATION']) and np.array_equal(norm,maps['HITS']),'normalization/hits replay')
    valid=maps['SCI_SUPPORT'].astype(bool);require(np.array_equal(output,np.flatnonzero(valid)),'saved science rows')
    for key,file in [('total','total.fits'),('clean','residual.fits'),('operand','restored.fits')]:
        old,_=auth.fits(products/file);n=sums[key].reshape(shape);checks[key+'_numerator']=close(n,old['NUMERATOR'],key+' numerator',1e-7,1e-11)
        signal=np.divide(n,norm,out=np.full(shape,np.nan),where=norm>0);checks[key+'_signal']=close(signal[valid],old['PRIMARY'][valid],key+' signal',1e-9,1e-11)
    checks['rejoin']=close(sums['total'],sums['clean']+sums['operand'],'saved rejoin',1e-7,1e-11)
    # Exact archived WCS and continuous AST inputs are bound, not replaced by
    # this one-hot operator's containing-pixel indices. Native AST reconstruction
    # is a separate prerequisite before any eventual JINC observation execution.
    result=dict(array=array,branch=branch,stage=k,checks=checks,normalization_hits_exact=True,output_rows_exact=True,
        quantity='saved cleaned residual plus identical previously subtracted sample operand; mJy/nominal beam',
        removed_mean_state='original saved numerical mean unchanged and not restored',applied_candidate=None if k==0 else request['candidate'],m10_applied=False,
        original_measurement=stage['measurement'],measurement_status='archived values reused after replay; no new fit',
        wcs={key:header[key] for key in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','RADESYS','EQUINOX')},shape=shape,
        coefficient='SCI-PTC:uniform_constant@draft-0.1; dimensionless gamma=1; exact saved payload',
        continuous_AST='original native AST/CAL inputs bound through archived input manifest; no containing-pixel substitution or coordinate regeneration here',
        segments=segments,archived_total=auth.ref(products/'total.fits'),input_manifest=auth.ref(products/'input-manifest.yaml'),
        seconds=time.perf_counter()-began,JINC='not run: numerical parameter/profile authorization absent')
    save_json(out/f'{array}-{branch}-{k:02d}.json',result);return result


def recover(out):
    start=time.perf_counter();cpu=time.process_time();revision=source();require(not out.exists(),'fresh recovery bundle required')
    auth,owner=ancestors();records={};stages=[]
    for root in (CD,CD.parent/'successor-point-152389-post-fruit-ab-20260923',CD.parent/'successor-point-152389-fruit-ten-cycles-20260923'):
        for n in ('COMPLETION.json','REVIEW.md','REGENERATION.md'):
            if (root/n).exists():auth.check(root/n)
    for array in ARRAYS:
        for branch,k in MATRIX:
            if (array,branch) not in records:records[array,branch]=auth.json(CD/f'run/{branch}/{array}/SUMMARY.json')
            h=records[array,branch]['histories'];stage=h[k];require(stage['ready'] and stage['cycle_index']==k,'saved snapshot unavailable')
            req=auth.json(stage['request']['path'])
            if k:require(req['candidate']==h[9]['candidate'],'applied m9 mismatch')
            stages.append((array,branch,k,stage))
    # Freeze recovery scope before numerical replay inspection. This is NOT the
    # scientific comparison's prospective manifest, which needs owner choices.
    out.mkdir();dest=out/'baseline';dest.mkdir()
    save_json(out/'RECOVERY_SCOPE.json',dict(source=revision,matrix=[dict(array=a,branch=b,stage=k,total=s['total']) for a,b,k,s in stages],
        source_order='original C+D adaptive; no REFSEQ input',numerical_authorization='unavailable',observation_jinc_authorized=False,
        replay_tolerances=dict(signal_atol=1e-9,rtol=1e-11,numerator_atol=1e-7),no_new_fit_or_learning=True))
    results=[]
    # Controls first across arrays, then compact, then shoulder. No JINC-dependent work.
    stages.sort(key=lambda r:(MATRIX.index((r[1],r[2])),ARRAYS.index(r[0])))
    for a,b,k,s in stages:
        result=audit_snapshot(auth,a,b,k,s,dest);results.append(result)
        print(json.dumps(dict(array=a,branch=b,stage=k,passed=True,seconds=result['seconds'])),flush=True)
    final_seconds=auth.verify_all();manifest(dest)
    summary=dict(status='baseline recovered; JINC comparison not executed',source_revision=revision,results=results,
        authenticated_originals=auth.consumed,original_hash_seconds=auth.seconds,final_verification_seconds=final_seconds,
        wall_seconds=time.perf_counter()-start,parent_cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource_policy='serial bounded recovery/hash audit; no independent observation accumulations or ensembles yet; host14 logical CPUs/48GiB',
        expected_comparison_maps=30,primary_jinc_maps=0,phase_maps=0,noise_trials=0,
        unavailable=['three numerical kernel parameter sets','exact approved numerical adequacy profile and matching realized certificate'])
    save_json(out/'RECOVERY.json',summary);manifest(out);verify_manifest(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();recover(a.output)
