"""Localized positive RBF fit and separate spatial cross-prediction admission."""
from pathlib import Path
import sys,time
import numpy as np
from scipy import sparse,optimize,special

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'fruit_point_coherent_feedback_2026-09-11'
sys.path.insert(0,str(PREV))
import experiment_r02 as base

CONFIG=dict(spacing_arcsec=4.,fwhm_arcsec=5.,center_radius_arcsec=94.,lambda_value=.01,screened_floor=.01,
    cross_score_min=5.,fold_cosine_min=.8,maxiter=2000,maxfun=6000,kkt_relative_limit=1e-4)

class FitSystem:
    def __init__(self,A,B,penalty,lam):
        self.A=A.tocsr();self.n=A.shape[0];self.Q,self.T=np.linalg.qr(B,mode='reduced')
        if np.linalg.matrix_rank(self.T)!=3:raise ValueError('background rank unavailable')
        self.U=np.asarray(A.T@self.Q)/np.sqrt(self.n)
        self.K=(A.T@A/self.n+lam*penalty).tocsr()
        diag=self.K.diagonal()-np.sum(self.U*self.U,axis=1)
        if np.any(diag<=0):raise ValueError('nonpositive fit curvature')
        self.pre=np.sqrt(diag);self.lam=lam;self.warm=np.zeros(A.shape[1])
        upper=float(np.max(np.asarray(abs(self.K).sum(axis=1))))
        floor=lam*CONFIG['screened_floor']/A.shape[1]
        self.condition_bound=upper/floor if floor>0 else None
    def product(self,c):return self.K@c-base.mm(self.U,base.mm(self.U.T,c))
    def fit(self,y,warm=True):
        start=time.monotonic();scale=max(float(np.std(y)),1e-12);z=y/scale
        rhs=np.asarray(self.A.T@(z-base.mm(self.Q,base.mm(self.Q.T,z))))/self.n
        norm=max(float(np.max(abs(rhs))),1e-12)
        def objective(u):
            c=u/self.pre;g=self.product(c)-rhs
            return float(.5*np.dot(c,g-rhs)),g/self.pre
        u=self.warm*self.pre/scale if warm else np.zeros_like(rhs)
        res=optimize.minimize(objective,u,jac=True,method='L-BFGS-B',bounds=[(0.,None)]*len(u),
            options=dict(maxiter=CONFIG['maxiter'],maxfun=CONFIG['maxfun'],ftol=1e-13,gtol=1e-10,maxcor=15,maxls=40))
        c=res.x/self.pre;grad=self.product(c)-rhs
        pg=np.where(c>1e-8,grad,np.minimum(grad,0));kkt=float(np.max(abs(pg))/norm)
        if not res.success or not np.isfinite(c).all() or kkt>CONFIG['kkt_relative_limit']:
            raise ValueError(f'RBF solver unusable: {res.message}; relative KKT={kkt:.6g}; nit={res.nit}')
        c*=scale;beta=np.linalg.solve(self.T,base.mm(self.Q.T,y-self.A@c))
        if warm:self.warm=c.copy()
        return c,beta,dict(status=str(res.message),iterations=int(res.nit),function_evaluations=int(res.nfev),relative_kkt=kkt,
            condition_upper_bound=self.condition_bound,seconds=time.monotonic()-start)

class Basis:
    def __init__(self,data,a,lam=None):
        start=time.monotonic();self.a=a;self.x=data.x;self.y=data.ygrid;self.D=data.D[a];self.O=data.O[a];self.F=self.D|self.O
        self.ids=np.flatnonzero(self.F);self.di=np.flatnonzero(self.D);self.area=4.;self.spacing=CONFIG['spacing_arcsec'];self.lam=CONFIG['lambda_value'] if lam is None else lam
        sigma=CONFIG['fwhm_arcsec']/base.FWHM;rad=CONFIG['center_radius_arcsec'];steps=np.arange(-int(np.ceil(rad/self.spacing)),int(np.ceil(rad/self.spacing))+1)
        yy,xx=np.meshgrid(steps,steps,indexing='ij');keep=(xx*self.spacing)**2+(yy*self.spacing)**2<=rad*rad
        self.centers=np.column_stack([xx[keep],yy[keep]])*self.spacing;self.n=len(self.centers)
        lookup={(int(x),int(y)):i for i,(x,y) in enumerate(zip(self.x,self.y)) if self.D[i]}
        rows=[];cols=[];vals=[];extent=int(np.ceil(6*sigma/2))
        for j,(cx,cy) in enumerate(self.centers):
            gx=2*np.arange(int(np.floor(cx/2))-extent,int(np.floor(cx/2))+extent+2)
            gy=2*np.arange(int(np.floor(cy/2))-extent,int(np.floor(cy/2))+extent+2)
            wx=.5*(special.erf((gx+1-cx)/(np.sqrt(2)*sigma))-special.erf((gx-1-cx)/(np.sqrt(2)*sigma)))
            wy=.5*(special.erf((gy+1-cy)/(np.sqrt(2)*sigma))-special.erf((gy-1-cy)/(np.sqrt(2)*sigma)))
            norm=wx.sum()*wy.sum()
            for iy,y in enumerate(gy):
                for ix,x in enumerate(gx):
                    i=lookup.get((int(x),int(y)))
                    value=wx[ix]*wy[iy]/self.area/norm
                    if i is not None and value>0:rows.append(i);cols.append(j);vals.append(value)
        phi=sparse.csr_matrix((vals,(rows,cols)),shape=(len(self.x),self.n))
        # Centers with no admitted support cannot contribute or define a fit parameter.
        visible=np.asarray(phi.sum(axis=0)).ravel()>0
        self.centers=self.centers[visible];phi=phi[:,visible];self.n=phi.shape[1];self.phi=phi.tocsr()
        self.b=np.asarray(phi.sum(axis=0)).ravel()*self.area;self.H=(self.area*(phi.T@phi)).tocsr()
        coord={tuple(np.rint(p/self.spacing).astype(int)):j for j,p in enumerate(self.centers)};edges=[]
        for xy,j in coord.items():
            for dx,dy in [(1,0),(0,1)]:
                k=coord.get((xy[0]+dx,xy[1]+dy))
                if k is not None:edges.append((j,k))
        rr=np.repeat(np.arange(len(edges)),2);cc=np.array(edges).ravel();vv=np.tile([1.,-1.],len(edges))
        graph=sparse.csr_matrix((vv,(rr,cc)),shape=(len(edges),self.n))
        self.penalty=(graph.T@graph/max(len(edges),1)+CONFIG['screened_floor']*sparse.eye(self.n)/self.n).tocsr()
        self.A=(phi[self.ids]*self.spacing**2).tocsr();self.B=np.column_stack([np.ones(len(self.ids)),self.x[self.ids]/90,self.y[self.ids]/90])
        self.parity=((np.rint(self.x[self.ids]/2).astype(int)+np.rint(self.y[self.ids]/2).astype(int))%2)
        self.systems=[FitSystem(self.A,self.B,self.penalty,self.lam)]
        self.systems += [FitSystem(self.A[self.parity==f],self.B[self.parity==f],self.penalty,self.lam) for f in [0,1]]
        self.setup_seconds=time.monotonic()-start
        self.identity=dict(spacing_arcsec=self.spacing,fwhm_arcsec=CONFIG['fwhm_arcsec'],center_radius_arcsec=rad,basis_count=self.n,
            phi_nnz=int(phi.nnz),edge_count=len(edges),fit_pixels=len(self.ids),aperture_pixels=int(self.D.sum()),
            clipped_basis_integral_range=[float(self.b.min()),float(self.b.max())],lambda_value=self.lam,setup_seconds=self.setup_seconds,
            conditioning_bounds=[s.condition_bound for s in self.systems])
    def reset(self):
        for s in self.systems:s.warm[:]=0
    def fit_only(self,m,warm=False):
        c,beta,info=self.systems[0].fit(m[self.F],warm)
        a=c*self.spacing**2;return a,beta,info,np.asarray(self.phi@a)
    def concentration(self,a,admitted=True):
        F=float(np.dot(self.b,a));power=float(np.dot(a,self.H@a));model=self.phi@a
        directF=float(self.area*np.sum(model[self.D]));directPower=float(self.area*np.dot(model[self.D],model[self.D]))
        if not np.isclose(F,directF,rtol=1e-11,atol=1e-10) or not np.isclose(power,directPower,rtol=1e-11,atol=1e-10):raise ValueError('RBF overlap/integration discrepancy')
        raw=None if F<=0 or power<=0 else dict(integrated_brightness=F,effective_area_arcsec2=F*F/power,
            coefficient_effective_number=float(1/np.sum((a*self.b/F)**2)),direct_effective_area_arcsec2=directF**2/directPower)
        return dict(available=bool(admitted and raw is not None),reason='admitted_positive_model' if admitted and raw is not None else 'not_admitted_or_zero_source',
            raw_fit_diagnostic=raw,aperture='fixed D: admitted science support within 90 arcsec of map origin',
            brightness_units='legacy mJy/beam arcsec^2; not physical flux density',area_units='arcsec^2',background='excluded',basis_dependent_N=True)
    def infer(self,m,R):
        start=time.monotonic();a,beta,full,model=self.fit_only(m,warm=True);fits=[];folds=[];scores=[]
        for f in [0,1]:
            train=self.parity==f;c,bb,inf=self.systems[1+f].fit(m[self.ids[train]])
            held=~train;h=np.asarray(self.A[held]@c);Q,_=np.linalg.qr(self.B[held],mode='reduced');hp=h-base.mm(Q,base.mm(Q.T,h));den=np.linalg.norm(hp)
            score=float(np.dot(hp,m[self.ids[held]])/(R*den)) if den>1e-12 else None
            folds.append(np.asarray(self.phi@(c*self.spacing**2)));scores.append(score);fits.append(dict(solver=inf,background=bb,heldout_score=score))
        n0=np.linalg.norm(folds[0][self.D]);n1=np.linalg.norm(folds[1][self.D]);cos=float(np.dot(folds[0][self.D],folds[1][self.D])/(n0*n1)) if n0*n1>1e-20 else None
        admitted=all(s is not None and s>=CONFIG['cross_score_min'] for s in scores) and cos is not None and cos>=CONFIG['fold_cosine_min']
        reason='spatial_cross_prediction' if admitted else 'spatial_cross_prediction_rejection'
        return np.where(self.D,model,0) if admitted else np.zeros_like(model),a,dict(reason=reason,admitted=bool(admitted),scale=R,
            heldout_scores=scores,fold_cosine=cos,background=beta,full_solver=full,fold_fits=fits,concentration=self.concentration(a,admitted),
            inference_seconds=time.monotonic()-start)
