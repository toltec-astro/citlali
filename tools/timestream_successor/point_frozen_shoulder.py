"""Authenticated saved-state accounting; never fit, Learn, admit, or iterate."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import label, binary_dilation, binary_erosion
from inspect_forward_map import require, digest, read_yaml
from point_fruit_diagnostics import load
from point_empirical import coordinates, source_patch
from point_empirical_io import write_fits
from point_additive_diagnostics import save_json

CD=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-response-robustness-cd-20260923')
CD_HASH='d18b0de1585baa96bdf1c047b695b3ec6f971a546ad340e3457116478207d061'
ARRAYS=('a1100','a1400','a2000');SNAPSHOTS=(10,0,1,3,5)
PRIMARY=('D','F','CMU','CQ');CENTERED=('DC','FC','CQ');REVERSE=('REV_D','REV_F','REV_CMU','REV_CQ')
TOL=dict(replay_atol=1e-9,replay_rtol=1e-11,numerator_atol=1e-7,closure_atol=1e-8,closure_rtol=1e-11)

class Auth:
    def __init__(self):self.expected={};self.consumed={};self.seconds=0
    def bind(self,p,h):
        p=str(Path(p).resolve());require(p not in self.expected or self.expected[p]==h,'conflicting archived digest '+p);self.expected[p]=h
    def check(self,p):
        p=Path(p).resolve();s=str(p);require(s in self.expected,'unbound consumed file '+s)
        if s not in self.consumed:
            t=time.perf_counter();require(digest(p)==self.expected[s],'changed archived file '+s);self.seconds+=time.perf_counter()-t;self.consumed[s]=self.expected[s]
        return p
    def ref(self,p):p=self.check(p);return dict(path=str(p),sha256=self.expected[str(p)])
    def manifest(self,root,h):
        root=Path(root);p=root/'FILES_SHA256SUMS';self.bind(p,h);self.check(p)
        for line in p.read_text().splitlines():
            sha,name=line.split('  ',1);f=(root/name).resolve();require(f.is_relative_to(root),'manifest path escape');self.bind(f,sha)
    def json(self,p):return json.loads(self.check(p).read_text())
    def yaml(self,p):return read_yaml(self.check(p))
    def fits(self,p):return load(self.check(p))
    def verify_all(self):
        t=time.perf_counter()
        for p,h in self.consumed.items():require(digest(Path(p))==h,'completion input changed '+p)
        return time.perf_counter()-t


def ancestors():
    a=Auth();a.manifest(CD,CD_HASH)
    for n in ('COMPLETION.json','REVIEW.md','FINAL_REVIEW.json','REGENERATION.md'):a.check(CD/n)
    owner=a.json(CD/'freeze/prospective.json')
    for q in owner['ancestors']:a.manifest(q['root'],q['manifest']['sha256'])
    first=CD.parent/'successor-point-152389-fruit-one-cycle-20260923/cycle/provenance.json'
    provenance=a.json(first)
    for n in ('admission','additive'):a.manifest(owner['immutable_roots'][n],provenance[n+'_manifest'])
    return a,owner


def saved_context(auth,owner,array):
    p=np.asarray(owner['original_cycles'][array][0]['measurement']['fit']['parameters']);major=max(p[3:5])*np.sqrt(8*np.log(2))
    path=Path(owner['immutable_roots']['admission'])/array/'source-relative.fits';relative,h=auth.fits(path)
    with np.load(auth.check(Path(owner['immutable_roots']['additive'])/array/'model.npz')) as n:q=n['weights'].copy()
    x,y=coordinates(q.shape,h);radius=np.hypot(x-p[1],y-p[2]);injection,ih=auth.fits(owner['injections'][array]['C-shoulder']['path'])
    return dict(p=p,major=major,q=q,x=x,y=y,radius=radius,header=h,scale=relative['PIXEL_SCALE'],noise_valid=relative['NOISE_VALID'].astype(bool),h=injection['INJECTION'],hvalid=injection['VALID'].astype(bool))


def close(a,b,name,atol=1e-8,rtol=1e-11):
    a,b=np.asarray(a),np.asarray(b);require(a.shape==b.shape,name+' shape')
    e=abs(a-b);require(np.isfinite(e).all() and np.all(e<=atol+rtol*np.maximum(abs(a),abs(b))),name+' tolerance')
    return dict(max_abs=float(e.max(initial=0)),max_relative_floor_1=float((e/np.maximum(1,np.maximum(abs(a),abs(b)))).max(initial=0)))


def map_measures(v,known,domain,ctx,header):
    use=domain&np.isfinite(v)&np.isfinite(known);require(np.array_equal(use,domain),'component unavailable on comparison domain')
    denominator=np.sum(known[use]**2);g=float(np.sum(known[use]*v[use])/denominator)
    eta=float(np.linalg.norm(v[use]-known[use])/np.sqrt(denominator));orth=float(np.linalg.norm(v[use]-g*known[use])/np.sqrt(denominator))
    require(abs(eta**2-(g-1)**2-orth**2)<1e-10,'Pythagorean identity')
    patch,support,_,_,_=source_patch(v,domain,np.ones(v.shape),header)
    roi=ctx['q']!=0;require(np.all(support[roi]),'template incomplete')
    linear=dict(template=float(np.sum(ctx['q'][roi]*patch[roi])))
    for factor,key in [(1.,'aperture_1'),(1.5,'aperture_1p5'),(2.,'aperture_2')]:
        roi=ctx['radius']<=factor*ctx['major'];require(support[roi].all(),'aperture incomplete');linear[key]=float(np.sum(patch[roi])*4)
    edges=np.arange(0,42,2);profile=[float(np.mean(patch[m])) if np.any(m) else None for m in [support&(ctx['radius']>=lo)&(ctx['radius']<hi) for lo,hi in zip(edges[:-1],edges[1:])]]
    return dict(g=g,eta=eta,orthogonal_distortion=orth,linear=linear,profile=profile,pixels=int(use.sum()))


def candidate_diagnostics(auth,pair,ctx,out):
    if pair['cycle']==0:return dict(status='no applied candidates',F_must_be_zero=True)
    candidates=[];evidence={};valid=ctx['hvalid'];h=ctx['h'];table=[]
    for b in ('R','H'):
        c,ch=auth.fits(pair[b]['applied_candidate']['path']);m=c['ADMITTED'].astype(bool);v=c['VALID'].astype(bool)
        require(np.array_equal(v,valid),'candidate valid domain changed');require(np.all(c['CANDIDATE'][v&~m]==0) and np.all(c['CANDIDATE'][m]>0),'candidate positive/zero semantics')
        candidates.append(c)
        previous=pair[b]['previous'];total,th=auth.fits(previous['total']['path']);s,sv,_,_,_=source_patch(total['PRIMARY'],total['SCI_SUPPORT'].astype(bool),total['NORMALIZATION'],th)
        par=np.asarray(previous['measurement']['fit']['parameters']);scale=ctx['scale'];nv=ctx['noise_valid'];domain=np.hypot(ctx['x'],ctx['y'])<=40
        vexpected=domain&sv&nv&np.isfinite(s)&np.isfinite(scale)&(scale>0);require(np.array_equal(v,vexpected),'saved candidate validity differs')
        ratio=np.full(s.shape,np.nan);ratio[v]=(s[v]-par[6])/scale[v];close(ratio[v],c['RATIO'][v],'saved admission ratio',1e-8)
        # Read-only audit of historical connected membership, never construct a candidate.
        eligible=v&(ratio>=3);labels,_=label(eligible,structure=np.ones((3,3)));column=int(np.floor(par[1]/(ch['CDELT1']*3600)+ch['CRPIX1']-.5));row=int(np.floor(par[2]/(ch['CDELT2']*3600)+ch['CRPIX2']-.5));anchor=int(labels[row,column]);require(anchor>0,'historical anchor absent');require(np.array_equal(m,labels==anchor),'historical connected membership mismatch')
        fractions={}
        for n,power in [('brightness_area',1),('squared_brightness',2)]:
            den=np.sum(h[v]**power);inside=float(np.sum(h[v&m]**power)/den);fractions[n]=dict(inside=inside,outside=1-inside)
        evidence[b]=dict(candidate=pair[b]['applied_candidate'],preceding_total=previous['total'],fitted_constant=float(par[6]),anchor_row=row,anchor_column=column,anchor_component=anchor,component_size=int(m.sum()),geometric_overlap=fractions,ratio_reproduced=True,connected_membership_reproduced=True)
    r,hc=candidates;mr=r['ADMITTED'].astype(bool);mh=hc['ADMITTED'].astype(bool);common=mr&mh;added=mh&~mr;removed=mr&~mh
    parts={'RETAIN':np.where(valid&common,hc['CANDIDATE']-r['CANDIDATE'],0),'ADD':np.where(valid&added,hc['CANDIDATE'],0),'REMOVE':np.where(valid&removed,-r['CANDIDATE'],0)}
    close(sum(parts.values())[valid],(hc['CANDIDATE']-r['CANDIDATE'])[valid],'candidate grid partition',0,0)
    boundary=valid&(binary_dilation(mr|mh,structure=np.ones((3,3)))&~binary_erosion(mr&mh,structure=np.ones((3,3))))
    for iy,ix in np.argwhere(added|removed|boundary):
        table.append(dict(row=int(iy),column=int(ix),east=float(ctx['x'][iy,ix]),north=float(ctx['y'][iy,ix]),retained=bool(common[iy,ix]),added=bool(added[iy,ix]),removed=bool(removed[iy,ix]),ratio_R=float(r['RATIO'][iy,ix]),ratio_H=float(hc['RATIO'][iy,ix]),selected_R=bool(mr[iy,ix]),selected_H=bool(mh[iy,ix])))
    write_fits(out/'applied-masks.fits',ctx['header'],[('SHOULDER',ctx['h'],'mJy/nominal beam'),('VALID',valid.astype('u1'),'1'),('MASK_R',mr.astype('u1'),'1'),('MASK_H',mh.astype('u1'),'1'),('RATIO_R',r['RATIO'],'1'),('RATIO_H',hc['RATIO'],'1')]+[(n,np.where(valid,v,np.nan),'mJy/nominal beam') for n,v in parts.items()],companion='FROZEN-APPLIED-MASKS')
    return dict(status='verified arithmetic partition of historical candidates',branches=evidence,counts=dict(retained=int(common.sum()),added=int(added.sum()),removed=int(removed.sum())),pixel_evidence=table,interpretation='Overlap is geometric only; propagated F pieces include spatial mixing and prior history, not isolated threshold causation.')
