"""Matched starlet with explicit zero-MAD normalization repair. No truth inputs."""
import time
import numpy as np
from scipy.ndimage import convolve1d, minimum_filter1d
from scipy.optimize import minimize

FILTERS=[]
for level in range(4):
    h=np.zeros(4*2**level+1);h[::2**level]=np.array([1,4,6,4,1])/16
    FILTERS.append(h)

def smooth(z, level):
    h=FILTERS[level]
    return convolve1d(convolve1d(z,h,axis=0,mode='constant',cval=0),h,axis=1,mode='constant',cval=0)

def analysis(z):
    c=np.asarray(z,float);bands=[]
    for level in range(4):
        n=smooth(c,level);bands.append(c-n);c=n
    return np.array(bands+[c])

def adjoint(v):
    # Reverse the entire smoothing cascade; sum(bands) is NOT this adjoint.
    g=v[4].copy()
    for level in range(3,-1,-1):g=v[level]+smooth(g-v[level],level)
    return g

def eligible(support):
    valid=np.asarray(support,bool);bands=[]
    for level in range(4):
        # Prior cascade footprints fill the gaps between dilated filter taps.
        # The first filter has no gaps. The complete cumulative footprint is
        # therefore a square, with successive radii 2, 6, 14, 30 pixels.
        n=len(FILTERS[level])
        valid=minimum_filter1d(minimum_filter1d(valid,n,axis=0,mode='constant',cval=0),n,axis=1,mode='constant',cval=0)
        bands.append(valid.copy())
    return np.array(bands+[valid.copy()])

def mad(v):return float(1.4826*np.median(np.abs(v-np.median(v))))

class Estimator:
    def __init__(self,x,y,shape,S,D,O,Q):
        self.shape=tuple(shape);self.S=np.asarray(S,bool);self.D=np.asarray(D,bool);self.O=np.asarray(O,bool)
        self.B=np.column_stack([np.ones(len(x)),x/90,y/90])
        self.valid=eligible(self.S.reshape(self.shape)).reshape(5,-1)
        self.q_split=float(np.median(Q[self.D]));self.stratum=(Q>self.q_split).astype(int)
        self.sigmas=None
        self.normalization_fallback=None

    def residual(self,z):
        if not np.isfinite(z[self.S]).all():raise ValueError('nonfinite supported input')
        beta=np.linalg.lstsq(self.B[self.O],z[self.O],rcond=None)[0]
        # Elementwise reduction avoids an unrelated BLAS floating-status issue.
        plane=np.sum(self.B*beta[None,:],axis=1)
        return np.where(self.S,z-plane,0.),beta

    def calibrate(self,z):
        Y,beta=self.residual(z);W=analysis(Y.reshape(self.shape)).reshape(5,-1)
        rows=[];sigma=np.zeros((5,2));failures=[]
        for j in range(5):
            for q in range(2):
                m=self.D&self.valid[j]&(self.stratum==q);n=int(m.sum());s=mad(W[j,m]) if n else None
                ok=n>=64 and s is not None and np.isfinite(s) and s>0
                rows.append(dict(band=j+1,stratum=q,eligible_pixels=n,scatter=s,available=bool(ok)))
                if not ok:failures.append(f'band {j+1}, stratum {q}: count={n}, scatter={s}')
                elif s is not None:sigma[j,q]=s
        self.sigmas=sigma if not failures else None
        self.normalization_fallback=mad(Y[self.O]) if not failures else None
        return dict(available=not failures,failures=failures,bands=rows,q_median=self.q_split,background=beta,
                    eligible_D_per_band=np.sum(self.valid&self.D,axis=1),outer_MAD=mad(Y[self.O]))

    def infer(self,z):
        start=time.perf_counter();u=np.zeros_like(z,dtype=float)
        self.last_trial=u.copy()
        rec=dict(available=False,admitted=False,reason=None)
        if self.sigmas is None:
            rec.update(reason='calibration_unavailable',seconds=time.perf_counter()-start)
            return u,rec,None
        Y,beta=self.residual(z);W=analysis(Y.reshape(self.shape)).reshape(5,-1)
        sigma=self.sigmas[:,self.stratum];omega=self.valid&self.D&(np.abs(W)>=5*sigma)
        rec.update(background=beta,selected_per_band=np.sum(omega,axis=1),outer_MAD=mad(Y[self.O]))
        solver_scale=self.normalization_fallback if rec['outer_MAD']==0 else rec['outer_MAD']
        rec.update(solver_scale=solver_scale,scale_source='fixed_calibration_null_outer_MAD' if rec['outer_MAD']==0 else 'current_outer_MAD')
        if not omega.any():rec.update(available=True,reason='empty_multiscale_support')
        elif solver_scale is None or not np.isfinite(solver_scale) or solver_scale<=0:
            rec.update(reason='normalization_unavailable')
        else:
            scale=solver_scale;target=Y/scale;weights=np.where(omega,(scale/sigma)**2,0).reshape((5,)+self.shape)
            target_bands=analysis(target.reshape(self.shape));domain=self.D
            def objective(v):
                canvas=np.zeros_like(Y);canvas[domain]=v
                diff=analysis(canvas.reshape(self.shape))-target_bands
                grad=adjoint(weights*diff).ravel()[domain]
                return .5*float(np.sum(weights*diff*diff)),grad
            zero=np.zeros(int(domain.sum()));_,g0=objective(zero)
            result=minimize(objective,zero,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(zero),
                            options=dict(maxiter=300,maxfun=3000,ftol=1e-12,gtol=1e-8))
            fun,grad=objective(result.x);pg=np.where((result.x<=0)&(grad>0),0,grad)
            relative=float(np.max(np.abs(pg))/max(float(np.max(np.abs(g0))),1e-12))
            finite=bool(np.isfinite(result.x).all() and np.isfinite(fun) and np.isfinite(relative))
            self.last_trial[domain]=result.x*scale
            ok=finite and bool(result.success) and relative<=1e-4
            rec.update(solver_success=bool(result.success),solver_message=str(result.message),objective=fun,
                       iterations=int(result.nit),function_evaluations=int(result.nfev),relative_projected_gradient=relative,
                       finite=finite,available=bool(ok),reason='accepted_solve' if ok else 'solver_gate_failure')
            if ok:
                u[domain]=result.x*scale;rec['admitted']=bool(np.any(u>0))
                if not rec['admitted']:rec.update(available=False,reason='exact_zero_solution')
        rec['seconds']=time.perf_counter()-start
        return u,rec,omega
