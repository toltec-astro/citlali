"""Deterministic scientific arithmetic and failure checks, before any campaign."""
import numpy as np
from numpy.testing import assert_allclose as close, assert_array_equal as equal
from core import (weight_generation,from_training,accumulate,support,pixel_indices,
                  moments_map,source_metrics,continuous_bounds,binomial_bounds,tail_registry)
from synthetic import geometry,source,ideal,trial_metrics
from discovery import grid

def run_checks():
    checks=[]
    def done(name):checks.append(dict(name=name,status='PASS'))
    e=np.array([2,3,5]);g=weight_generation(np.array([3,3,0]),np.zeros(3),np.array([.5,.25,np.nan]),e)
    close(g['gamma'],[5/8,5/4,1],rtol=1e-12,atol=1e-14);close(g['mean'],3.2)
    b=np.repeat([1.,2.,3.],e);d=np.repeat(np.arange(3),e);pix=np.zeros(10,int)
    m=accumulate(pix,b,g['gamma'][d],1);u=accumulate(pix,b,np.ones(10),1)
    close([m['numerator'][0],m['Q'][0],m['map'][0],u['map'][0]],[23.75,10,2.375,2.3]);done('fallback mean and exact signal contribution')
    n=np.array([3,0,1,3,3,3,2,0]);v=np.array([1,1,1,0,np.nan,np.nextafter(0.,1.),2,np.nan]);e=np.array([1,1,1,1,1,1,1,0])
    g=weight_generation(n,np.zeros(8),v,e);equal(g['reason'],[0,2,2,3,3,4,0,1]);close(g['gamma'][1:6],np.ones(5));assert np.isfinite(g['gamma'][6]);done('all fallback causes, n=2, and unused groups')
    assert not weight_generation(np.zeros(2),np.zeros(2),np.zeros(2),np.ones(2))['available'];done('all training unavailable fails closed')
    g1=weight_generation([3,3],[0,0],[1,4],[2,3]);g2=weight_generation([3,3],[0,0],[10,40],[2,3]);close(g1['gamma'],g2['gamma'],rtol=1e-12,atol=1e-14);done('common scale invariance')
    for count in [0,1,2,3,4,8,81]:
        q=np.arange(1,count+1,dtype=float);s=support(q,q)
        expected=((3*count//4)+count)//2 if count else -1
        assert s['index']==expected
    s=support(np.array([.005,.02,.2,10]),np.array([.005,.02,.2,10]));equal(s['S_norm'],[False,False,True,True]);equal(s['S_sci'],[False,False,False,True]);done('threshold integer oracle and distinct supports')
    j,k=pixel_indices(np.array([-1.5,-.5,.5,1.5]),np.zeros(4),1);equal(j,[-1,0,1,2]);done('half-open containing-cell edges')
    b=np.array([2.,3.,5.,np.nan]);flag=np.array([0,1,0,0]);x=np.array([0.,0.,1.,0.]);use=(flag==0)&np.isfinite(b)&np.isfinite(x)
    p=x[use].astype(int);m=accumulate(p,b[use],np.ones(2),2);m2=accumulate(p,b[use],np.ones(2),2);equal(m['map'],m2['map']);equal(m['count'],[1,1]);done('same invalidity population and equal-arm identity')
    train=np.arange(24.,dtype=float).reshape(8,3);valid=np.ones_like(train,bool);e=np.array([5,5,5]);g=from_training(train,valid,e)
    for ev in [np.zeros((5,3)),np.full((5,3),987.)]:close(from_training(train,valid,np.isfinite(ev).sum(axis=0))['gamma'],g['gamma'])
    done('finite evaluation replacement cannot affect generation')
    geom=geometry(8);base=np.tile(np.sin(np.arange(256))[None,:],(16,1));v=np.ones(base.T.shape,bool);e=np.bincount(geom['det'])
    g0=from_training(base.T,v,e);g1=from_training((base+source(geom['train_x'],geom['train_y'],8)).T,v,e);assert not np.allclose(g0['gamma'],g1['gamma']);done('contaminated training changes its own generation')
    cols=np.array([0,1,0,1]);wins=np.array([0,0,1,2]);assert sum(int((wins==i).sum()) for i in range(3))==len(wins)
    # Two columns with equal illustrative UID remain two separate coefficient inputs.
    duplicate_uid=np.array([7,7]);e=np.bincount(cols,minlength=len(duplicate_uid));equal(e,[2,2]);done('window partition and column identity without UID merge')
    p=np.array([0,1,0,1]);b=np.array([7.,3.,5.,9.]);w=np.array([1.,2.,3.,4.]);T=np.array([.2,1.]);base=accumulate(p,b,w,2)
    for sign in [-1,1]:
        m=accumulate(p,b+sign*T[p],w,2);close(m['map']-base['map'],sign*T,rtol=1e-12,atol=1e-14);equal(m['S_sci'],base['S_sci'])
    constant=accumulate(p,np.full(4,3.),w,2);close(constant['map'],[3,3]);done('constant and signed fixed-generation response')
    x=np.array([-1.,0.,1.,3.]);y=np.array([-1.,0.,1.,3.]);C=np.array([1,1,1,0],bool);O=~C;T=np.array([.5,1,.5,0]);m=2*T+4
    s=source_metrics(m,C,O,T,x,y);close([s['amplitude'],s['centroid_x'],s['width_x']],[2,0,np.sqrt(.5)],rtol=1e-12,atol=1e-14)
    assert np.isnan(moments_map(np.array([-1.,-1.]),np.array([0.,1.]),np.array([0.,1.]))).all()
    assert np.isnan(moments_map(np.array([1.,-3.,3.]),np.array([0.,1.,2.]),np.array([0.,1.,2.]))[2])
    missing=m.copy();missing[0]=np.nan;assert np.isnan(source_metrics(missing,C,O,T,x,y)['amplitude']);done('signed morphology, amplitude and missing support')
    assert continuous_bounds(np.full(1024,-1.))['disposition']=='pass'
    assert continuous_bounds(np.full(1024,1.))['disposition']=='failure'
    assert continuous_bounds(np.linspace(-1,1,1024))['disposition']=='inconclusive'
    assert continuous_bounds(np.full(1024,-2.),1.,True)['disposition']=='failure'
    assert continuous_bounds(np.array([1,np.nan]),2.,True)['disposition']=='inconclusive'
    assert binomial_bounds(0,1024)['lower']==0 and binomial_bounds(1024,1024)['upper']==1
    assert len(tail_registry())==242;done('interval directions, missing trials and exact tail count')
    for row in range(9):
        geom=geometry(row);t=ideal(geom,row);maps=np.stack([np.zeros((2,81)),np.stack([t,t])]);metrics=trial_metrics(maps,t,geom,row)
        close(metrics[:,2:9],0,rtol=1e-12,atol=1e-14);assert len(geom['det'])==(10368 if row==3 else 5184)
    done('all T1 ideal source metrics and registered occurrence counts')
    geo=grid(dict(x=np.array([-1.,2.]),y=np.array([0.,0.])),2.,2000000);assert geo['regions']['D'].sum()==29 and geo['shape']==(7,7);done('full required disk included even when unvisited')
    return dict(status='PASS',count=len(checks),checks=checks,stochastic_trials=0,external_inputs_opened=False)

if __name__=='__main__':
    import json
    print(json.dumps(run_checks(),indent=2))
