#!/usr/bin/env python3
"""Report cross-platform numerics, while requiring exact identities/masks/support.

No floating-point acceptance tolerance is selected here. Reported differences
remain review evidence; this is not an automatic scientific equivalence claim.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ptc_spod import Inputs, digest, require


def compare(before,after):
    binary=lambda r:{str(p.relative_to(r)) for p in r.rglob('*') if p.suffix in ('.f64','.i64','.u8','.u16')}
    require(binary(before)==binary(after),'binary inventory differs')
    ledger=Inputs();a=ledger.yaml(before/'learning-receipt.yaml');b=ledger.yaml(after/'learning-receipt.yaml')
    for k in ('observation','network','rows','native_runs','native_time_sha256','source_protection_authority','spectral_estimator','spectral_conventions'):
        require(a[k]==b[k],'identity/time convention differs: '+k)
    require(a['explicit_array_lowpass']['sha256']==b['explicit_array_lowpass']['sha256'],'filter coefficients differ')
    fields=('detector','channel','prior_flxscale','factor_authority','factor_unit')
    require([{k:d[k] for k in fields} for d in a['detectors']]==[{k:d[k] for k in fields} for d in b['detectors']],'detector/calibration identities differ')
    rows=[];exact=True;changed=0
    for rel in sorted(binary(before)):
        old,new=before/rel,after/rel;equal=digest(old)==digest(new)
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
    return dict(exact_identity_timing_mask_support=bool(exact),numeric_files_different=changed,files=rows,
                numerical_disposition='bitwise' if changed==0 else 'reported-cross-platform-differences-require-review-no-tolerance-selected',
                original_pair_content_bindings_match=[d['samples_sha256'] for d in a['detectors']]==[d['samples_sha256'] for d in b['detectors']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('baseline','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=compare(a.baseline,a.candidate);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    raise SystemExit(0 if r['exact_identity_timing_mask_support'] else 1)
