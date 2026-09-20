#!/usr/bin/env python3
"""Report cross-platform numerics, while requiring exact identities/masks/support.

No floating-point acceptance tolerance is selected here. Reported differences
remain review evidence; this is not an automatic scientific equivalence claim.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from ptc_spod import Inputs, digest, require
from compare_successor_outputs import TIMERS, METRICS


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
    statistics={}
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
            elif stage=='ptc' and key in progress:
                statistics['/'.join(path+(key,))]=v
            elif key in ('source_CAL_receipt_sha256','CAL_classification'):
                result[key]='verified-own-CAL-receipt'
            else:result[key]=clean(v,stage,path+(key,))
        return result
    return dict(CAL=clean(cal,'cal'),PTC=clean(ptc,'ptc')),statistics


def compare(before,after):
    binary=lambda r:{str(p.relative_to(r)) for p in r.rglob('*') if p.suffix in ('.f64','.i64','.u8','.u16')}
    require(binary(before)==binary(after),'binary inventory differs')
    ledger=Inputs();a=ledger.yaml(before/'learning-receipt.yaml');b=ledger.yaml(after/'learning-receipt.yaml')
    for k in ('schema','observation','network','rows','native_runs','native_time_sha256',
              'VAL_generation','original_parent','source_protection','source_protection_authority',
              'native_integration_seconds','cadence_interval_seconds','spectral_estimator','spectral_conventions','spectra'):
        require(a[k]==b[k],'identity/time convention differs: '+k)
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
    am,astat=receipt_semantics(before,ledger,hashes[0]);bm,bstat=receipt_semantics(after,ledger,hashes[1])
    require(am==bm,'CAL/PTC identity, support, validity, policy or unavailable disposition differs')
    differences=[dict(field=k,baseline=astat.get(k),candidate=bstat.get(k))
                 for k in sorted(astat.keys()|bstat.keys()) if astat.get(k)!=bstat.get(k)]
    ledger.unchanged()
    return dict(exact_identity_timing_mask_support=bool(exact),numeric_files_different=changed,files=rows,
                receipt_semantics_match=True,solver_statistics_differences=differences,
                numerical_disposition='bitwise' if changed==0 and not differences and original_equal else 'reported-cross-platform-differences-require-review-no-tolerance-selected',
                original_pair_content_bindings_match=original_equal)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('baseline','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=compare(a.baseline,a.candidate);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if r['exact_identity_timing_mask_support'] else 1)
