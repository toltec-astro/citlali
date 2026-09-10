"""The nine registered T1 laws; regeneration is independent for each trial."""
import time
import numpy as np
from core import (POLICIES,require,from_training,accumulate,moments_map,continuous_bounds,
                  binomial_bounds,write_json,ring_metrics)
CASES=['W00','W01a','W01b','W02','W03a','W03b','W04a','W04b','W05']
FIELDS=['noise_power','null_mean','amplitude_bias','centroid_x_bias','centroid_y_bias',
        'width_x_bias','width_y_bias','response_residual_power','signal_error_power',
        'gross_excursion','null_corr_x','null_corr_y']

def geometry(row):
    yy,xx=np.mgrid[-4:5,-4:5];x,y=xx.ravel().astype(float),yy.ravel().astype(float)
    offsets=np.array([[-.25,-.25],[-.25,.25],[.25,-.25],[.25,.25]])
    bx=np.repeat(x,4)+np.tile(offsets[:,0],81);by=np.repeat(y,4)+np.tile(offsets[:,1],81)
    px=[];py=[];p=[];d=[]
    for det in range(16):
        r=3 if row==3 and det>=8 else 1
        px.append(np.repeat(bx,r));py.append(np.repeat(by,r));p.append(np.repeat(np.repeat(np.arange(81),4),r));d.append(np.full(324*r,det))
    train_x=np.repeat(np.where(np.arange(16)<4,3.25,-4)[:,None],256,axis=1)
    train_y=np.tile((np.arange(256)%5-2)[None,:],(16,1))
    return dict(x=x,y=y,ox=np.concatenate(px),oy=np.concatenate(py),pixel=np.concatenate(p),det=np.concatenate(d),train_x=train_x,train_y=train_y)

def source(x,y,row):
    amplitude=.625 if row==5 else 1.25;w=2.8 if row==8 else 2.;cx=1. if row==8 else 0.
    r2=(x-cx)**2+y*y
    return amplitude*np.exp(-4*np.log(2)*r2/w**2)*(r2<=9)

def generate(row,trial,g):
    rng=lambda stream:np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260909,row,trial,stream])))
    ratio=1 if row==0 else 4 if row==2 else 2
    sd=np.where(np.arange(16)<8,1.,ratio)
    eval_sd=np.where(np.arange(16)<8,1.,4.) if row==5 else sd
    tr=rng(0).standard_normal((16,256))*sd[:,None]
    ev=rng(1).standard_normal(len(g['det']))*eval_sd[g['det']]
    if row==4:
        normals=rng(2).standard_normal(256+324);a=np.empty_like(normals);a[0]=normals[0]
        for j in range(1,len(a)):a[j]=.95*a[j-1]+np.sqrt(1-.95**2)*normals[j]
        tr+=a[:256];ev+=np.tile(a[256:],16)
    if row==6:
        # Streams 3/4: training detector-major, then evaluation detector-major.
        indicators=rng(3).random(tr.size+ev.size)<.01
        signs=np.where(rng(4).random(tr.size+ev.size)<.5,-1.,1.)
        burst=indicators*signs
        tr+=burst[:tr.size].reshape(tr.shape)*8*sd[:,None]*(np.arange(16)[:,None]>=12)
        ev+=burst[tr.size:]*8*eval_sd[g['det']]*(g['det']>=12)
    st=source(g['train_x'],g['train_y'],row);se=source(g['ox'],g['oy'],row)
    states_tr=np.stack([tr,tr+st]);states_ev=np.stack([ev,ev+se])
    if row==7:
        states_tr[:,:, :]*=np.where(np.arange(16)>=12,1.1,1.)[None,:,None]
        states_ev*=np.where(g['det']>=12,1.1,1.)
    return states_tr,states_ev

def ideal(g,row):
    v=source(g['ox'],g['oy'],row)
    return np.bincount(g['pixel'],weights=v,minlength=81)/np.bincount(g['pixel'],minlength=81)

def trial_metrics(maps,t,g,row):
    ans=np.full((2,len(FIELDS)),np.nan);reference=moments_map(t,g['x'],g['y'])
    A=.625 if row==5 else 1.25
    for arm in range(2):
        null,signal=maps[0,arm],maps[1,arm];delta=signal-null
        if not np.isfinite(maps[:,arm]).all():continue
        amp=A*np.dot(t,delta)/np.dot(t,t);mom=moments_map(delta,g['x'],g['y'])
        rings=ring_metrics(null,np.ones(81,bool),(9,9))
        ans[arm]=[np.mean(null**2),null.mean(),amp-A,*(mom-reference),np.mean((delta-t)**2),np.mean((signal-t)**2),float(np.max(np.abs(null))>5*.125),rings['x']['correlation'],rings['y']['correlation']]
    return ans

def summarize(metrics,maps,supports,t,g,row):
    idx={k:i for i,k in enumerate(FIELDS)};n=len(metrics);checks={}
    checks['noise']=continuous_bounds(metrics[:,0,0]-1.1025*metrics[:,1,0])
    A=.625 if row==5 else 1.25;w=2.8 if row==8 else 2.;ref=moments_map(t,g['x'],g['y'])
    for arm,name in enumerate(POLICIES):
        for field,margin in [('amplitude_bias',.02*A),('centroid_x_bias',.02*w),('centroid_y_bias',.02*w),('width_x_bias',.02*ref[2]),('width_y_bias',.02*ref[3])]:
            checks[f'{name}/{field}']=continuous_bounds(metrics[:,arm,idx[field]],margin,True)
    excursions=[]
    for arm in range(2):
        vals=metrics[:,arm,idx['gross_excursion']]
        excursions.append(binomial_bounds(int(np.sum(vals)),n) if np.isfinite(vals).all() else None)
    if all(e is not None for e in excursions):
        lo=excursions[0]['lower']-excursions[1]['upper'];hi=excursions[0]['upper']-excursions[1]['lower']
        checks['excursion_difference']=dict(lower=lo,upper=hi,margin=.01,arms=excursions,disposition='pass' if hi<=.01 else 'failure' if lo>.01 else 'inconclusive')
    else:checks['excursion_difference']=dict(arms=excursions,disposition='inconclusive')
    analytic=None
    if row<4:
        ratio=[1,2,4,2][row];sd=np.where(g['det']<8,1.,ratio)
        count=np.bincount(g['pixel'],minlength=81)
        expectation=float(np.mean(np.bincount(g['pixel'],weights=sd**2,minlength=81)/count**2))
        interval=continuous_bounds(metrics[:,0,0]);ok=interval['available'] and interval['lower']<=expectation<=interval['upper']
        for field in ['disposition','margin','absolute']:interval.pop(field,None)
        analytic=dict(expectation=expectation,interval=interval,passed=ok)
    missing=int((~supports).sum());complete=missing==0
    dispositions=[c['disposition'] for c in checks.values()]
    result='failure' if not complete or 'failure' in dispositions else 'pass' if all(v=='pass' for v in dispositions) else 'inconclusive'
    return dict(case=CASES[row],row=row,trials=n,conditional_disposition=result,support_missing_pixels=missing,checks=checks,analytic_U=analytic,metric_names=FIELDS,metric_means=np.mean(metrics,axis=0),interpretation='Synthetic-law conditional result; no TolTEC policy qualification')

def run_t1(run,settings):
    results=[]
    for row,name in enumerate(CASES):
        run.check(f'T1 {name} start');g=geometry(row);t=ideal(g,row);n=settings['trials']
        maps=np.empty((n,2,2,81));numerators=np.empty_like(maps);q=np.empty_like(maps)
        sn=np.empty(maps.shape,bool);ss=np.empty_like(sn);causes=np.empty(maps.shape,np.uint8)
        gamma=np.empty((n,2,2,16));mus=np.empty((n,2,16));vs=np.empty_like(mus);raw=np.empty_like(mus);reasons=np.empty(mus.shape,np.uint8)
        metrics=np.empty((n,2,len(FIELDS)));times=np.zeros(3)
        e=np.bincount(g['det'],minlength=16);valid=(g['train_x']**2+g['train_y']**2)>9
        for trial in range(n):
            if trial%64==0:run.check(f'T1 {name} trial {trial}')
            train,ev=generate(row,trial,g)
            for state in range(2):
                start=time.monotonic();gen=from_training(train[state].T,valid.T,e);times[0]+=time.monotonic()-start
                require(gen['available'],'Unexpected unavailable T1 generation')
                mus[trial,state]=gen['mu'];vs[trial,state]=gen['v'];raw[trial,state]=gen['raw'];reasons[trial,state]=gen['reason']
                for arm in range(2):
                    weights=np.ones(16) if arm==0 else gen['gamma'];gamma[trial,state,arm]=weights
                    start=time.monotonic();m=accumulate(g['pixel'],ev[state],weights[g['det']],81);times[arm+1]+=time.monotonic()-start
                    maps[trial,state,arm]=m['map'];numerators[trial,state,arm]=m['numerator'];q[trial,state,arm]=m['Q'];sn[trial,state,arm]=m['S_norm'];ss[trial,state,arm]=m['S_sci'];causes[trial,state,arm]=m['cause']
            metrics[trial]=trial_metrics(maps[trial],t,g,row)
        out=run.out/f'T1_{name}.npz'
        np.savez_compressed(out,maps=maps,numerator=numerators,Q=q,S_norm=sn,S_sci=ss,cause=causes,gamma=gamma,training_mu=mus,training_v=vs,training_n=valid.sum(axis=1),raw_weight=raw,fallback_reason=reasons,e=e,metrics=metrics,ideal=t,x=g['x'],y=g['y'],pixel=g['pixel'],detector=g['det'],occurrence_x=g['ox'],occurrence_y=g['oy'],train_x=g['train_x'],train_y=g['train_y'])
        result=summarize(metrics,maps,ss,t,g,row);result['seconds']=dict(zip(['weights','U_mapping','N4U_mapping'],times))
        write_json(run.out/f'T1_{name}.json',result);results.append(result)
        run.check(f'T1 {name} saved: {result["conditional_disposition"]}')
        require(result['analytic_U'] is None or result['analytic_U']['passed'],'T1 analytic benchmark failed; interpretation blocked')
    write_json(run.out/'T1_SUMMARY.json',results)
    return results
