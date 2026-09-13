"""Experimental terminal readout: profiled free Gaussian plus plane.

The public fitter accepts only map values, coordinates and a domain mask.
It receives no saved fits, labels, injected truth, case or arm identity.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from common import FWHM,coherent_start
LOW=np.array([-80.,-80.,np.log(4.),np.log(4.),-np.pi])
HIGH=np.array([80.,80.,np.log(60.),np.log(60.),np.pi])
QSCALE=np.array([10.,10.,1.,1.,1.])
WIDTH_STARTS=(6.,18.,42.)
SETTINGS=dict(method='trf',max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-8,
              x_scale=QSCALE.tolist(),loss='linear',tr_solver='exact')
OBJECTIVE_ATOL=1e-8
OBJECTIVE_RTOL=1e-10
FIRST_ORDER_TOL=1e-6
NEAR_COST_RTOL=1e-6

def mm(a,b):
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        c=np.matmul(a,b)
    if not np.isfinite(c).all():raise FloatingPointError('nonfinite matrix product')
    return c

def shape_and_derivatives(q,x,y):
    cx,cy,lx,ly,theta=q;c,s=np.cos(theta),np.sin(theta)
    u=c*(x-cx)+s*(y-cy);v=-s*(x-cx)+c*(y-cy)
    ix=(FWHM/np.exp(lx))**2;iy=(FWHM/np.exp(ly))**2
    g=np.exp(-.5*(u*u*ix+v*v*iy))
    dg=g[:,None]*np.column_stack([u*c*ix-v*s*iy,u*s*ix+v*c*iy,u*u*ix,v*v*iy,u*v*(iy-ix)])
    return g,dg

def linear_profile(q,x,y,d):
    g,dg=shape_and_derivatives(q,x,y)
    design=np.column_stack([g,np.ones(len(x)),x/90,y/90])
    U,s,Vt=np.linalg.svd(design,full_matrices=False)
    keep=s>np.finfo(float).eps*max(design.shape)*s[0]
    inv=np.divide(1.,s,out=np.zeros_like(s),where=keep)
    coeff=mm(Vt.T,mm(U.T,d)*inv)
    residual=np.sum(design*coeff,axis=1)-d
    # Full derivative of D(q) c_hat(q)-d, including coefficient changes.
    t=coeff[0]*dg
    jac=t-mm(U[:,keep],mm(U[:,keep].T,t))
    jac-=mm(U,Vt[:,0]*inv)[:,None]*np.sum(dg*residual[:,None],axis=0)[None,:]
    return dict(q=np.array(q),coefficients=coeff,design=design,residual=residual,
                jacobian=jac,rank=int(keep.sum()),singular_values=s,
                condition_number=float(s[0]/s[-1]) if s[-1]>0 else None)

class Profile:
    def __init__(self,x,y,d,scale):
        self.x=x;self.y=y;self.d=d;self.scale=scale
        self.cached=None;self.evaluations=0;self.fun_calls=0;self.jac_calls=0
    def get(self,q):
        if self.cached is None or not np.array_equal(q,self.cached['q']):
            self.cached=linear_profile(q,self.x,self.y,self.d);self.evaluations+=1
        return self.cached
    def fun(self,q):
        self.fun_calls+=1
        return self.get(q)['residual']/self.scale
    def jac(self,q):
        self.jac_calls+=1
        return self.get(q)['jacobian']/self.scale

def fit_record(p,sse,pixels,status,nfev,successful_starts):
    p=np.array(p);widths=np.exp(p[3:5]);order=np.argsort(widths)[::-1]
    boundary=bool(np.any(np.abs(p[1:3])>=79.999) or np.any(widths<=4.001) or np.any(widths>=59.999))
    return dict(parameters=p,peak=float(p[0]),centroid=p[1:3],widths=widths[order],
        angle_rad=float((p[5]+(np.pi/2 if order[0] else 0))%np.pi),
        gaussian_integral=float(p[0]*2*np.pi*np.prod(widths/FWHM)),background=p[6:9],
        residual_rms=float(np.sqrt(sse/pixels)),sse=float(sse),boundary_rejection=boundary,
        solver_status=int(status),nfev=int(nfev),successful_starts=int(successful_starts))

def diagnostics(v,scale):
    q=v['q'];r=v['residual']/scale;j=v['jacobian']/scale
    grad=np.sum(j*r[:,None],axis=0);cost=.5*float(np.sum(r*r))
    scaled=QSCALE*grad
    tol=1e-8
    active_low=q-LOW<=tol*QSCALE;active_high=HIGH-q<=tol*QSCALE
    projected=np.where((active_low&(scaled>0))|(active_high&(scaled<0)),0.,scaled)
    # Unit step in scaled geometry, projected into the original box.
    u=q/QSCALE
    mapping=u-np.clip(u-scaled,LOW/QSCALE,HIGH/QSCALE)
    normal=np.sum(v['design']*v['residual'][:,None],axis=0)
    normal_rel=float(np.linalg.norm(normal)/max(np.linalg.norm(v['design'])*np.linalg.norm(v['residual']),1e-30))
    finite=bool(np.isfinite(q).all() and np.isfinite(v['coefficients']).all() and np.isfinite(r).all() and np.isfinite(j).all())
    feasible=bool(finite and np.all(q>=LOW) and np.all(q<=HIGH) and v['rank']==4)
    score=float(np.max(np.abs(projected))/max(cost,1.))
    return dict(finite=finite,feasible=feasible,rank=v['rank'],condition_number=v['condition_number'],
        singular_values=v['singular_values'],normalized_cost=cost,geometry_gradient=grad,
        scaled_projected_gradient=projected,relative_scaled_projected_gradient=score,
        projected_unit_step_mapping=mapping,active_lower=active_low,active_upper=active_high,
        linear_normal_relative=normal_rel,numerical_complete=bool(feasible and score<=FIRST_ORDER_TOL and normal_rel<=1e-10))

def fit_source(z,x,y,domain):
    begin=time.perf_counter();good=domain&np.isfinite(z)
    xx=x[good];yy=y[good];d=z[good]
    if len(d)<100:raise ValueError('insufficient source-fit support')
    plane=np.column_stack([np.ones(len(d)),xx/90,yy/90])
    beta=np.linalg.lstsq(plane,d,rcond=None)[0]
    residual=d-np.sum(plane*beta,axis=1)
    scale=max(float(np.std(residual)),1e-12)
    starts=[]
    for width in WIDTH_STARTS:
        amp,cx,cy=coherent_start(residual,xx,yy,width)
        starts.append(dict(width=width,ordinary_amplitude=amp,ordinary_background=beta,
                           geometry=np.array([cx,cy,np.log(width),np.log(width),.2])))
    initialization_seconds=time.perf_counter()-begin
    attempts=[]
    for index,start in enumerate(starts):
        t=time.perf_counter();obj=Profile(xx,yy,d,scale)
        try:
            result=least_squares(obj.fun,start['geometry'],jac=obj.jac,bounds=(LOW,HIGH),**SETTINGS)
            v=obj.get(result.x);c=v['coefficients'];p=np.r_[c[0],result.x,c[1:]]
            diag=diagnostics(v,scale)
            attempts.append(dict(start_index=index,start=start,parameters=p,sse=float(np.sum(v['residual']**2)),
                success=bool(result.success),status=int(result.status),message=str(result.message),
                reported_nfev=int(result.nfev),reported_njev=int(result.njev),reported_optimality=float(result.optimality),
                residual_evaluations=obj.evaluations,linear_subsolves=obj.evaluations,fun_calls=obj.fun_calls,jac_calls=obj.jac_calls,
                numerical_derivative_evaluations=0,diagnostics=diag,seconds=time.perf_counter()-t))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as err:
            attempts.append(dict(start_index=index,start=start,error=str(err),success=False,status=-99,
                residual_evaluations=obj.evaluations,linear_subsolves=obj.evaluations,fun_calls=obj.fun_calls,jac_calls=obj.jac_calls,
                numerical_derivative_evaluations=0,diagnostics=dict(feasible=False),seconds=time.perf_counter()-t))
    feasible=[r for r in attempts if r['diagnostics']['feasible']]
    # Objective first; original deterministic start order breaks exact ties.
    chosen=min(feasible,key=lambda r:(r['sse'],r['start_index'])) if feasible else None
    available=bool(chosen is not None and chosen['success'])
    fit=None
    if chosen is not None:
        fit=fit_record(chosen['parameters'],chosen['sse'],len(d),chosen['status'],chosen['reported_nfev'],sum(v['success'] for v in attempts))
    return dict(available=available,selected_start=None if chosen is None else chosen['start_index'],
        selected_fit=fit,attempts=attempts,normalization_scale=scale,initialization_seconds=initialization_seconds,
        seconds=time.perf_counter()-begin,starts_attempted=len(attempts),initialization_linear_subsolves=1,
        residual_evaluations=sum(r['residual_evaluations'] for r in attempts),
        linear_subsolves=1+sum(r['linear_subsolves'] for r in attempts),numerical_derivative_evaluations=0,
        completion_policy='minimum feasible objective; incomplete selected result retained diagnostic but unavailable')
