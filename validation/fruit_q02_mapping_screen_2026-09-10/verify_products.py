"""Saved-product checks with separate add.at accumulators and direct moments."""
import json
from pathlib import Path
import numpy as np
import netCDF4
from numpy.testing import assert_allclose as close,assert_array_equal as equal
from core import digest,require,write_json,ARRAYS,POLICIES,legacy
from synthetic import CASES,FIELDS

def verify(out,settings,run):
    out=Path(out);synthetic_count=0;real_count=0;checks=[]
    for name in CASES:
        run.check(f'Verify T1 {name}')
        with np.load(out/f'T1_{name}.npz') as z:
            m=z['maps'];q=z['Q'];N=z['numerator'];ss=z['S_sci'];gamma=z['gamma'];e=z['e'];metrics=z['metrics'];t=z['ideal'];xy=[z['x'],z['y']]
            equal(m.shape,(1024,2,2,81));close(m[ss],(N/q)[ss],rtol=1e-12,atol=1e-14)
            require(ss.all(),'T1 required support absent')
            close(np.sum(gamma*e,axis=-1)/e.sum(),1,rtol=1e-12,atol=1e-14)
            group_pixel=np.zeros((16,81));np.add.at(group_pixel,(z['detector'],z['pixel']),1)
            close(q,np.einsum('tsad,dp->tsap',gamma,group_pixel),rtol=1e-12,atol=1e-14)
            close(metrics[:,:,0],np.sum(m[:,0]**2,axis=-1)/81,rtol=1e-12,atol=1e-14)
            close(metrics[:,:,1],np.sum(m[:,0],axis=-1)/81,rtol=1e-12,atol=1e-14)
            delta=m[:,1]-m[:,0];A=.625 if name=='W03b' else 1.25
            close(metrics[:,:,2],A*np.einsum('tap,p->ta',delta,t)/np.dot(t,t)-A,rtol=1e-12,atol=1e-14)
            for axis,x in enumerate(xy):
                reference=np.dot(t,x)/t.sum();center=np.sum(delta*x,axis=-1)/delta.sum(axis=-1)
                close(metrics[:,:,3+axis],center-reference,rtol=1e-12,atol=1e-14,equal_nan=True)
                second=np.sum(delta*(x-center[:,:,None])**2,axis=-1)/delta.sum(axis=-1)
                width=np.sqrt(np.where(second>0,second,np.nan));target=np.sqrt(np.dot(t,(x-reference)**2)/t.sum())
                close(metrics[:,:,5+axis],width-target,rtol=1e-12,atol=1e-14,equal_nan=True)
            close(metrics[:,:,7],np.mean((delta-t)**2,axis=-1),rtol=1e-12,atol=1e-14)
            close(metrics[:,:,8],np.mean((m[:,1]-t)**2,axis=-1),rtol=1e-12,atol=1e-14)
            equal(metrics[:,:,9],(np.max(abs(m[:,0]),axis=-1)>.625).astype(float));synthetic_count+=m.shape[0]*4
        row=json.loads((out/f'T1_{name}.json').read_text());require(row['analytic_U'] is None or row['analytic_U']['passed'],'Analytic benchmark not passed')
        checks.append(dict(case=name,checks='map/normalization/support/independent scalar moments PASS'))
    for item in settings['inputs']:
        obs=item['obsnum'];require(digest(item['path'])==item['sha256'],'Input identity changed during execution')
        gen=np.load(out/f'T2_{obs}_generation.npz');lock=json.loads((out/f'T2_{obs}_GENERATION_LOCK.json').read_text());require(lock['sha256']==digest(out/lock['generation_path']),'Generation mutated')
        prior_path=Path(__file__).resolve().parent.parent/f'fruit_q02_weight_precision_2026-09-09/attempt_01/{obs}_K4_groups.npz'
        with np.load(prior_path) as old:
            for name in ['n','mu','v','e']:close(gen[name],old[name],rtol=1e-12,atol=1e-14,equal_nan=True)
        for a in range(3):
            use=(gen['array']==a)[None,:] & (gen['e']>0)
            close(np.sum(gen['gamma'][use]*gen['e'][use])/gen['e'][use].sum(),1,rtol=1e-12,atol=1e-14)
        with netCDF4.Dataset(item['path'],'r') as f:
            _,erows,_,_=legacy.split_rows(gen['edges']);row_lookup=np.full(3628,-1);row_lookup[erows]=np.arange(len(erows))
            # Independent read of only the admitted evaluation half rows for verification.
            b=np.ma.filled(f['signal'][erows,:],np.nan).astype(float)
            for a,name in enumerate(ARRAYS):
                run.check(f'Verify T2 {obs} {name}')
                prefix=f'T2_{obs}_{name}';pop=np.load(out/(prefix+'_population.npz'));p=pop['pixel'];col=pop['column'];win=pop['window'];v=b[row_lookup[pop['storage_row']],col];npix=len(pop['x']);T=pop['template']
                require(np.isfinite(v).all(),'Saved population has invalid evaluation values')
                group_e=np.zeros(gen['e'].shape,np.int64);np.add.at(group_e,(win,col),1)
                equal(group_e[:,gen['array']==a],gen['e'][:,gen['array']==a])
                for scope in ['full','window0','window1','window2']:
                    selected=np.ones(len(p),bool) if scope=='full' else win==int(scope[-1])
                    for arm in POLICIES:
                        weights=np.ones(selected.sum()) if arm=='U' else gen['gamma'][win[selected],col[selected]]
                        pp=p[selected];vv=v[selected]
                        # Different accumulation primitive, independently from saved numerators.
                        q=np.zeros(npix);N=np.zeros(npix);n=np.zeros(npix,np.int64);q2=np.zeros(npix)
                        np.add.at(q,pp,weights);np.add.at(N,pp,weights*vv);np.add.at(n,pp,1);np.add.at(q2,pp,weights**2)
                        with np.load(out/(prefix+f'_{scope}_{arm}.npz')) as z:
                            close(z['Q'],q,rtol=1e-12,atol=1e-14);close(z['numerator'],N,rtol=1e-12,atol=1e-14);close(z['Q2'],q2,rtol=1e-12,atol=1e-14);equal(z['count'],n)
                            sorted_q=sorted(q[q>0]);nn=len(sorted_q);star=sorted_q[((3*nn//4)+nn)//2] if nn else 0
                            sn=(q>0)&(q>=.01*star);ss=(q>0)&(q>=.1*star)
                            equal(z['S_norm'],sn);equal(z['S_sci'],ss);close(z['map'][ss],N[ss]/q[ss],rtol=1e-12,atol=1e-14)
                            fallback=np.isin(gen['reason'][win[selected],col[selected]],[2,3,4]);fc=np.zeros(npix,int);fq=np.zeros(npix)
                            np.add.at(fc,pp[fallback],1);np.add.at(fq,pp[fallback],weights[fallback]);equal(z['fallback_count'],fc);close(z['fallback_Q'],fq,rtol=1e-12,atol=1e-14)
                            # Independently audit central complete-source and ring moments.
                            summary=json.loads((out/(prefix+'_summary.json')).read_text())['comparisons'][scope]['native'][arm]
                            for region in ['O','O1','O2']:
                                mask=pop[region]
                                if np.all(ss[mask]):
                                    x=N[mask]/q[mask];saved=summary['rings'][region]
                                    close([saved['mean'],saved['power'],saved['centered_power']],[np.sum(x)/len(x),np.dot(x,x)/len(x),np.dot(x-x.mean(),x-x.mean())/len(x)],rtol=1e-12,atol=1e-14)
                            if np.all(ss[pop['C']|pop['O']]):
                                x=N[pop['C']]/q[pop['C']]-(N[pop['O']]/q[pop['O']]).mean();tt=T[pop['C']]
                                close(summary['source']['amplitude'],np.dot(tt,x)/np.dot(tt,tt),rtol=1e-12,atol=1e-14)
                            real_count+=1
                            if scope=='full':
                                for sign in [1,-1]:
                                    with np.load(out/(prefix+f'_response_{arm}_{sign:+d}.npz')) as r:
                                        equal(r['S_sci'],ss);equal(r['S_norm'],sn);close(r['delta'][ss],sign*T[ss],rtol=0,atol=1e-9)
                                        close(r['numerator'][ss]/q[ss]-N[ss]/q[ss],sign*T[ss],rtol=0,atol=1e-9);real_count+=1
                checks.append(dict(obsnum=obs,array=name,checks='independent evaluation accumulation, support, fallback, response and complete-region moments PASS'))
                pop.close()
        gen.close()
    require(synthetic_count==36864 and real_count==72,'Declared map counts not verified')
    result=dict(status='PASS',T1_maps=synthetic_count,T2_maps=real_count,input_hashes='both unchanged',generation_locks='unchanged',reserved_129081='not opened',checks=checks)
    write_json(out/'PRODUCT_VERIFICATION.json',result)
    return result
