"""Read-only strict cold-start comparator against retained BASE-POINT REF."""
import argparse,json
from pathlib import Path
import numpy as np
from astropy.io import fits
from inspect_forward_map import require


def available(path):
    if path.is_file():return path
    gz=Path(str(path)+'.gz');require(gz.is_file(),'missing saved comparison product: '+str(path));return gz


def compare(actual,reference,array,tolerances,cycles=10,compare_endpoint=True):
    rows=[];failures=[]
    def planes(rel,kind):
        with fits.open(available(actual/rel),memmap=False) as a,fits.open(available(reference/rel),memmap=False) as b:
            require([h.name for h in a]==[h.name for h in b],'plane identities differ '+str(rel))
            for aa,bb in zip(a,b):
                x,y=aa.data,bb.data
                require(x.shape==y.shape,'plane shape differs')
                discrete=kind=='candidate' or aa.name in ('HITS','NORMALIZATION','SUPPORT','SCI_SUPPORT','GEOM_SUPPORT') or x.dtype.kind in 'biu'
                atol=0 if discrete else tolerances['numerator_atol'] if aa.name=='NUMERATOR' else tolerances['signal_atol'];rtol=0 if discrete else tolerances['signal_rtol']
                equal=np.allclose(x,y,atol=atol,rtol=rtol,equal_nan=True)
                finite=np.isfinite(x)&np.isfinite(y);error=float(np.max(abs(x[finite]-y[finite]))) if finite.any() else 0.
                row=dict(path=str(rel),plane=aa.name,exact=bool(np.array_equal(x,y,equal_nan=True)),max_abs=error,atol=atol,rtol=rtol,pass_=bool(equal));rows.append(row)
                if not equal:failures.append(row)
            for key in ('RADESYS','EQUINOX','BUNIT','CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2'):
                require(a[0].header.get(key)==b[0].header.get(key),'header identity differs '+key)
    planes(Path(f'M0/products/{array}/map.fits'),'map')
    route=Path(f'run/BASE-POINT/REF/{array}')
    a=json.loads((actual/route/'SUMMARY.json').read_text());b=json.loads((reference/route/'SUMMARY.json').read_text())
    require(a['completed_cycle']==cycles and b['completed_cycle']>=cycles and a['ready'] and b['ready'] and not a['m10_applied'],'trajectory did not complete frozen endpoint')
    decisions=[]
    for k,(x,y) in enumerate(zip(a['histories'],b['histories'])):
        rel=route/('initial' if k==0 else f'cycle-{k:02d}')
        if k:planes(rel/'native/products/total.fits','map')
        planes(rel/'candidate.fits','candidate')
        # Existing reference requires deterministic measurements/candidates exact.
        for key in ('measurement','admission'):
            same=x[key]==y[key];decisions.append(dict(cycle=k,field=key,exact=same))
            if not same:failures.append(dict(cycle=k,field=key,reason='deterministic record differs'))
        require(x['ready']==y['ready'] and x.get('stop_reason')==y.get('stop_reason'),'discrete continuation differs')
    endpoint=a.get('endpoint_uncertainty')==b.get('endpoint_uncertainty')
    if compare_endpoint and not endpoint:failures.append(dict(field='endpoint_uncertainty',reason='conditional uncertainties differ'))
    # Observation-local numerical state and high-coverage metadata must also agree.
    for name in ('context','C0','BASE'):
        with np.load(actual/f'freeze/{array}/{name}.npz') as x,np.load(reference/f'freeze/{array}/{name}.npz') as y:
            require(set(x.files)==set(y.files),'M0 frozen field inventory differs')
            for key in x.files:
                same=np.array_equal(x[key],y[key],equal_nan=True);decisions.append(dict(initialization=name,field=key,exact=bool(same)))
                if not same:failures.append(dict(initialization=name,field=key,reason='M0 frozen numerical field differs'))
    return dict(pass_=not failures,tolerances=tolerances,planes=rows,decisions=decisions,endpoint_exact=endpoint,failures=failures,maximum_absolute_plane_difference=max(r['max_abs'] for r in rows),scope='M0 and T1–T10, all six MAP planes and candidate planes, accepted source/fit/template/apertures, initial covariance and conditional endpoint; no science rerun')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('actual','reference','config','result'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());r=compare(a.actual,a.reference,c['array'],c['reference_tolerances']);a.result.write_text(json.dumps(r,indent=2)+'\n');raise SystemExit(0 if r['pass_'] else 1)
