#!/usr/bin/env python3
"""Report cross-platform numerics, while requiring exact identities/masks/support.

No floating-point acceptance tolerance is selected here. Reported differences
remain review evidence; this is not an automatic scientific equivalence claim.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from ptc_spod import Inputs, digest, require
from compare_successor_outputs import TIMERS, METRICS


def original_parent_binding(learning):
    """Verify the producer's compound identity before separating source from numerics.

    rtc_pipeline_execution hashes each projected channel's solved x/r and
    producer validity, then hashes ordered channel:hash lines. That final digest
    is platform-dependent numerical content, not another raw-file identity.
    Raw/Tune identities and channel relations must still agree exactly.
    """
    require(learning['schema']=='rtc-multidetector-learning-v1',
            'unsupported original parent receipt schema')
    match=re.fullmatch(r'sha256:([0-9a-f]{64}):network=([0-9]+):tune=sha256:([0-9a-f]{64})'
        r':paired-xr:exact-simultaneous-column-projection:sha256:([0-9a-f]{64})',
        learning['original_parent'])
    require(match is not None,'unsupported original parent identity form')
    raw,network,tune,projection=match.groups()
    require(int(network)==learning['network'],'original parent network binding mismatch')
    detectors=learning['detectors'];channels=[d['channel'] for d in detectors]
    require(channels and channels==sorted(set(channels)),
            'original projection channels must be nonempty, unique and ordered')
    require(all(re.fullmatch(r'[0-9a-f]{64}',d['samples_sha256']) for d in detectors),
            'malformed original sample content digest')
    content=''.join(f"{d['channel']}:{d['samples_sha256']}\n" for d in detectors)
    require(hashlib.sha256(content.encode()).hexdigest()==projection,
            'original projection digest does not bind its ordered channel/sample records')
    return dict(raw_sha256=raw,network=int(network),tune_sha256=tune),projection


def receipt_semantics(root, ledger, hashes):
    """Exact scientific mappings, independently verified against each artifact.

    Only solver progress, build provenance and measured costs may differ here.
    Floating-point arrays are assessed separately without an acceptance tolerance.
    """
    calroot=root/'donor-continuity/cal';ptcroot=root/'donor-continuity/ptc'
    cal=ledger.yaml(calroot/'receipt.yaml');ptc=ledger.yaml(ptcroot/'receipt.yaml')
    require(ptc['source_CAL_receipt_sha256']==digest(calroot/'receipt.yaml'),'PTC parent CAL digest mismatch')
    # A copied reference retains its original absolute publication path. The
    # digest above binds the actual parent; only that location may relocate.
    require(Path(ptc['CAL_classification']).parts[-3:]==('donor-continuity','cal','receipt.yaml'),'PTC parent CAL path mismatch')
    statistics={};numeric_representations={}
    progress={'objective','iterations','coefficient_factorizations','coefficient_factor_reuses',
              'apply_factorizations','apply_factor_reuses','working_matrix_bytes'}
    def clean(value,stage,path=()):
        if isinstance(value,list):return [clean(v,stage,path+(str(i),)) for i,v in enumerate(value)]
        if not isinstance(value,dict):return value
        result={}
        for key,v in value.items():
            if key=='files':
                for name,expected in v.items():
                    rel='donor-continuity/'+stage+'/'+name
                    require(hashes.get(rel)==expected,'receipt artifact digest mismatch: '+rel)
                result[key]=sorted(v)
            elif key in TIMERS|METRICS or key=='build_identity':continue
            elif stage=='ptc' and not path and key in ('relative_rank_tolerance','relative_objective_tolerance'):
                # yaml-cpp may emit 1e-05 or 1.0000000000000001e-05 for the
                # same double. PyYAML's YAML 1.1 resolver treats only the latter
                # as a float. Parse these declared numeric config fields, then
                # still require exact double equality (no tolerance relaxation).
                require(type(v) in (str,int,float),'invalid numeric PTC configuration: '+key)
                if isinstance(v,str):
                    require(re.fullmatch(r'[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?',v)
                            is not None,'invalid numeric PTC configuration: '+key)
                value=float(v)
                require(math.isfinite(value),'nonfinite numeric PTC configuration: '+key)
                numeric_representations[key]=dict(parsed_type=type(v).__name__,parsed_value=v)
                result[key]=value
            elif stage=='ptc' and key in progress:
                statistics['/'.join(path+(key,))]=v
            elif key in ('source_CAL_receipt_sha256','CAL_classification'):
                result[key]='verified-own-CAL-receipt'
            else:result[key]=clean(v,stage,path+(key,))
        return result
    return dict(CAL=clean(cal,'cal'),PTC=clean(ptc,'ptc')),statistics,numeric_representations


def compare(before,after):
    binary=lambda r:{str(p.relative_to(r)) for p in r.rglob('*') if p.suffix in ('.f64','.i64','.u8','.u16')}
    require(binary(before)==binary(after),'binary inventory differs')
    ledger=Inputs();a=ledger.yaml(before/'learning-receipt.yaml');b=ledger.yaml(after/'learning-receipt.yaml')
    for k in ('schema','observation','network','rows','native_runs','native_time_sha256',
              'VAL_generation','source_protection','source_protection_authority',
              'native_integration_seconds','cadence_interval_seconds','spectral_estimator','spectral_conventions','spectra'):
        require(a[k]==b[k],'identity/time convention differs: '+k)
    a_source,a_projection=original_parent_binding(a)
    b_source,b_projection=original_parent_binding(b)
    require(a_source==b_source,'original raw/Tune/network source bindings differ')
    require({k:v for k,v in a['explicit_array_lowpass'].items() if k!='path'}==
            {k:v for k,v in b['explicit_array_lowpass'].items() if k!='path'},'filter binding differs')
    # Occurrence entropy is per-load; semantic/envelope hashes and row mappings
    # remain exact. Preserve repeated occurrence relationships in each record.
    def identities(ds):
        entropy={}
        def replace(m):return entropy.setdefault(m.group(0),'apt-occurrence-'+str(len(entropy)))
        ds=[{k:v for k,v in d.items() if k!='samples_sha256'} for d in ds]
        return re.sub(r'apt-v2-occurrence:entropy/[0-9a-f]{64}',replace,json.dumps(ds,sort_keys=True))
    require(identities(a['detectors'])==identities(b['detectors']),'detector/calibration bindings differ')
    original_equal=[d['samples_sha256'] for d in a['detectors']]==[d['samples_sha256'] for d in b['detectors']]
    rows=[];exact=True;changed=0;hashes=[{},{}]
    for rel in sorted(binary(before)):
        old,new=before/rel,after/rel
        hashes[0][rel]=digest(old);hashes[1][rel]=digest(new);equal=hashes[0][rel]==hashes[1][rel]
        item=dict(path=rel,bytes=old.stat().st_size,bitwise_equal=equal)
        if old.suffix!='.f64' or old.name=='native-time.f64':
            item['role']='exact timing/support/validity';exact &= equal
        elif not equal:
            changed+=1;x=np.fromfile(old,'<f8');y=np.fromfile(new,'<f8')
            require(x.shape==y.shape,'numeric shape differs: '+rel)
            finite=np.isfinite(x)&np.isfinite(y)
            special_equal=np.array_equal(np.isnan(x),np.isnan(y)) and np.array_equal(np.isposinf(x),np.isposinf(y)) and np.array_equal(np.isneginf(x),np.isneginf(y))
            exact &= special_equal
            delta=y[finite]-x[finite];base=float(np.sqrt(np.mean(x[finite]**2))) if finite.any() else 0.
            item.update(special_values_match=special_equal,finite_elements=int(finite.sum()),
                unequal_finite_elements=int((x[finite]!=y[finite]).sum()),max_absolute_error=float(abs(delta).max(initial=0)),
                rms_error=float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.,baseline_rms=base,
                relative_rms_error=float(np.sqrt(np.mean(delta**2))/base) if base else None)
        rows.append(item)
    am,astat,arep=receipt_semantics(before,ledger,hashes[0]);bm,bstat,brep=receipt_semantics(after,ledger,hashes[1])
    require(am==bm,'CAL/PTC identity, support, validity, policy or unavailable disposition differs')
    differences=[dict(field=k,baseline=astat.get(k),candidate=bstat.get(k))
                 for k in sorted(astat.keys()|bstat.keys()) if astat.get(k)!=bstat.get(k)]
    ledger.unchanged()
    return dict(exact_identity_timing_mask_support=bool(exact),numeric_files_different=changed,files=rows,
                receipt_semantics_match=True,solver_statistics_differences=differences,
                receipt_numeric_representations=dict(baseline=arep,candidate=brep),
                numerical_disposition='bitwise' if changed==0 and not differences and original_equal else 'reported-cross-platform-differences-require-review-no-tolerance-selected',
                original_pair_content_bindings_match=original_equal,
                original_parent_binding=dict(source_bindings=a_source,source_bindings_match=True,
                    projection_digests_verified=True,baseline_projection_sha256=a_projection,
                    candidate_projection_sha256=b_projection,projection_content_matches=a_projection==b_projection,
                    baseline=a['original_parent'],candidate=b['original_parent']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('baseline','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=compare(a.baseline,a.candidate);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if r['exact_identity_timing_mask_support'] else 1)
