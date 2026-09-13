"""Deterministic analytic implementation fixtures; no benchmark products read."""
import numpy as np
from common import HERE,gaussian,model_jac,write
from profiled import shape_and_derivatives,linear_profile,fit_source,QSCALE

def check_jac(fun,jac,q,steps):
    finite=np.column_stack([(fun(q+np.eye(len(q))[i]*steps[i])-fun(q-np.eye(len(q))[i]*steps[i]))/(2*steps[i]) for i in range(len(q))])
    relative=np.linalg.norm(jac-finite)/max(np.linalg.norm(finite),1e-30)
    assert relative<2e-7,relative
    return float(relative)

def main():
    yy,xx=np.mgrid[-48:50:2,-48:50:2];x=xx.ravel();y=yy.ravel();domain=np.hypot(x,y)<=44
    x=x[domain];y=y[domain]
    p=np.array([73.,7.3,-4.1,np.log(15.),np.log(9.),.71,2.,-1.4,.8])
    d=gaussian(p,x,y,True)+.7*np.sin(x*.37)*np.cos(y*.29)
    q=np.array([-2.1,3.2,np.log(19.),np.log(6.5),-.31])
    v=linear_profile(q,x,y,d)
    jr=check_jac(lambda q:linear_profile(q,x,y,d)['residual'],v['jacobian'],q,QSCALE*1e-5)
    jp=check_jac(lambda p:gaussian(p,x,y,True),model_jac(p,x,y),p,np.array([1,10,10,1,1,1,1,1,1])*1e-5)
    # Independent unrestricted least-squares reference at a deliberately wrong geometry.
    c=np.linalg.lstsq(v['design'],d,rcond=None)[0]
    np.testing.assert_allclose(c,v['coefficients'],rtol=1e-11,atol=1e-11)
    full=linear_profile(p[1:6],x,y,gaussian(p,x,y,True))
    np.testing.assert_allclose(full['coefficients'],p[[0,6,7,8]],rtol=1e-11,atol=1e-11)
    negative=p.copy();negative[0]=-73
    nv=linear_profile(p[1:6],x,y,gaussian(negative,x,y,True))
    np.testing.assert_allclose(nv['coefficients'],negative[[0,6,7,8]],rtol=1e-11,atol=1e-11)
    switched=p.copy();switched[3:5]=p[[4,3]];switched[5]=p[5]+np.pi/2
    np.testing.assert_allclose(gaussian(p,x,y,True),gaussian(switched,x,y,True),rtol=1e-12,atol=1e-12)
    fits=[]
    for name,params in [('elliptical',p),('circular',np.r_[p[:3],np.log(13.),np.log(13.),p[5:]])]:
        z=gaussian(params,x,y,True)
        a=fit_source(z,x,y,np.ones(len(x),bool))
        f=a['selected_fit'];assert a['available']
        assert abs(f['peak']/params[0]-1)<1e-6
        assert np.linalg.norm(np.array(f['centroid'])-params[1:3])<1e-5
        assert f['sse']<1e-10
        fits.append(dict(name=name,sse=f['sse'],seconds=a['seconds'],starts=a['starts_attempted']))
    r=dict(profile_derivative_relative_error=jr,original_derivative_relative_error=jp,
        signed_coefficient_check=True,width_orientation_equivalence=True,analytic_fits=fits,
        benchmark_reads=0)
    write(HERE/'IMPLEMENTATION_TESTS.json',r);print(r)
if __name__=='__main__':main()
