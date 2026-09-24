"""Small independent saved-state fixtures; no synthetic or observation Learn."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import yaml
from point_frozen_shoulder import close
from point_fruit_trajectory import bound


def masked_q(z,basis,mask,converged=True,tol=1e-10):
    # Independent SVD fixture oracle, not the observation executor.
    values=np.zeros_like(z);causes=np.ones(z.shape,dtype='u1')
    for t in range(len(z)):
        use=mask[t].astype(bool);b=basis[use]
        if not use.any():continue
        if not converged:causes[t,use]=2;continue
        u,s,vt=np.linalg.svd(b,full_matrices=False)
        if np.count_nonzero(s*s>max(s*s)*tol)!=basis.shape[1]:causes[t,use]=4;continue
        values[t,use]=z[t,use]-b@np.linalg.lstsq(b,z[t,use],rcond=None)[0];causes[t,use]=0
    return values,causes


class FrozenControls(unittest.TestCase):
    def test_algebra_zero_identity_noop_and_exact_protection(self):
        rng=np.random.default_rng(824);u=rng.normal(size=11);d=rng.normal(size=11);z=rng.normal(size=(11,2));b=np.linalg.qr(z)[0];q=np.eye(11)-b@b.T
        for op in (np.eye(11),q):
            close(op@u+(np.eye(11)-op)@u,u,'ideal protected response',1e-12)
            close(op@np.zeros(11),np.zeros(11),'zero injection',0,0)
            close((np.eye(11)-op)@np.zeros(11),np.zeros(11),'M0/identical candidates',0,0)
            close((op-op)@d,np.zeros(11),'identical state',0,0)
        close((np.eye(11)-np.eye(11))@d,np.zeros(11),'no-op protection',0,0)

    def test_masked_affine_scaling_and_rank_deficiency(self):
        b=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,-1.]])/3
        mask=np.ones((5,4),dtype='u1');mask[0,1:]=0;mask[1,3]=0;mask[4]=0
        z=np.arange(20.).reshape(5,4);mean=np.array([9.,-4.,3.,2.]);v=np.sin(z)
        q,c=masked_q(z,b,mask);a,_=masked_q(z-mean,b,mask);ap,_=masked_q(z+v-mean,b,mask);qv,_=masked_q(v,b,mask)
        close(ap-a,qv,'affine difference',1e-12);close(masked_q(2*v,b,mask)[0],2*qv,'scaling',1e-12)
        self.assertTrue((c[0,mask[0].astype(bool)]==4).all());self.assertTrue((c[4]==1).all());self.assertTrue((q[c!=0]==0).all())

    def test_native_saved_apply_and_decomposition(self):
        binary=os.environ.get('POINT_FROZEN_APPLY_BINARY')
        if not binary:self.skipTest('explicit built native fixture binary required')
        rng=np.random.default_rng(924);nt,nd,no,npix=17,5,7,9;columns=[0,2,3,5,6]
        mask=np.ones((nt,nd),dtype='u1');mask[0,1:]=0;mask[1,3:]=0;mask[4,0]=0;mask[7]=0
        y=20+rng.normal(size=(nt,nd));u=.2*rng.normal(size=y.shape);u[mask==0]=0
        vr=np.maximum(rng.normal(size=y.shape),0);vh=np.maximum(vr+.3*rng.normal(size=y.shape),0);vr[mask==0]=0;vh[mask==0]=0
        br=np.linalg.qr(rng.normal(size=(nd,2)))[0];bh=np.linalg.qr(br+.01*rng.normal(size=br.shape))[0]
        def mean(z):return np.array([np.sum(z[mask[:,d]>0,d],dtype=np.longdouble)/sum(mask[:,d]) for d in range(nd)],dtype=float)
        for converged in (True,False):
            xr=y-vr;xh=y+u-vh;mr,mh=mean(xr),mean(xh);r,cr=masked_q(xr-mr,br,mask,converged);h,ch=masked_q(xh-mh,bh,mask,converged)
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);inputs={}
                def save(n,a):
                    p=root/n;a.tofile(p);inputs[n]=bound(p)
                def expand(a,fill):
                    out=np.full((nt,no),fill,dtype=a.dtype);out[:,columns]=a;return out
                for n,a in dict(y=y,u=u,vr=vr,vh=vh,mr=mr,mh=mh,br=br,bh=bh,oldvr=.8*vr,oldvh=.7*vh,cleanr=expand(r,0),cleanh=expand(h,0)).items():save(n,np.asarray(a,dtype='<f8'))
                for n,a in dict(mask=mask,causer=expand(cr,17),causeh=expand(ch,17)).items():save(n,a)
                pix=np.full(y.shape,-1,dtype='<i4');pix[cr==0]=np.arange(np.count_nonzero(cr==0))%npix;pix[2]=-1;save('pixel',pix)
                request=dict(schema='point-frozen-apply-v1',nt=nt,nd=nd,no=no,pixels=npix,rank=2,columns=columns,rank_tolerance=1e-10,converged_R=converged,converged_H=converged,reverse=True,inputs=inputs)
                req=root/'request.yaml';req.write_text(yaml.safe_dump(request));result=subprocess.run([binary,str(req),str(root/'out')],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr);receipt=yaml.safe_load((root/'out/receipt.yaml').read_text());self.assertEqual(receipt['native_learn_calls'],0)
                maps=dict(zip(receipt['components'],np.fromfile(root/'out/numerators.f64',dtype='<f8').reshape(-1,npix)))
                def mapped(v):return np.bincount(pix[pix>=0],weights=v[pix>=0],minlength=npix)
                close(maps['D'],mapped(masked_q(u,br,mask,converged)[0]),'native masked Q fixture',1e-12)
                change=(vh-vr)-(.7*vh-.8*vr);centered=np.where(mask,change-mean(change),0)
                expected_history=change-masked_q(centered,br,mask,converged)[0]
                close(maps['HISTORY'],mapped(expected_history),'independent native history',1e-12)
                close(sum(maps[n] for n in ['D','F','CMU','CQ']),mapped((h+vh)-(r+vr)),'independent mapped closure',1e-12)

if __name__=='__main__':unittest.main()
