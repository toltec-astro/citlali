"""Bounded saved-product diagnostics; existing MAP/POINT estimators remain owners."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from point_transfer_experiment import REPO,D,ROOT,ROLES,BRANCHES,read
from point_additive_diagnostics import save_json
from point_fruit_trajectory import bound
from point_rtc_diagnostics import yread,arr
from inspect_forward_map import require,digest
from point_empirical import coordinates,stats

def population_module(run,dest):
    spec=importlib.util.spec_from_file_location('transfer_saved_population',REPO/'validation/rtc_two_stage_2026-09-28/two_stage_population.py')
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.B=run;p.D=dest;p.R=ROOT
    return p

def final_product(run):return run/'run/BASE-POINT/REF/a1100/cycle-05/native/products'
def load(path):
    with fits.open(path) as h:return {v.name:v.data.copy() for v in h},dict(h[0].header)

def select_summary():
    """B-only consequence regions frozen before any Q and independent of Q."""
    dest=ROOT/'selection';base=ROOT/'B-REF';pop=population_module(base,dest/'B-population');pop.main()
    records=[];selected=[];new=[];excerpts=[];packed={};full,_=load(final_product(base)/'total.fits');shape=full['PRIMARY'].shape
    footprint=np.zeros(shape,bool);quiet=np.zeros(shape,bool)
    for nw in range(1,5):
        n,r,pr,stages,pix,off,distance,lon,lat=pop.network(nw)
        rows=yread(dest/f'network{nw}.yaml')['population'];source=n/'source-probe';t=arr(source/'assigned-time.f64','<f8');x=arr(source/'original-x.f64','<f8',pix.shape);rr=arr(source/'original-r.f64','<f8',pix.shape)
        vx=arr(source/'input-valid.u8','u1',pix.shape).astype(bool);vr=arr(source/'input-r-valid.u8','u1',pix.shape).astype(bool);mapping=arr(source/'mapping-available.u8','u1',pix.shape).astype(bool)
        for q in rows:
            d=q['detector'];lo,hi=q['native_interval'];q.update(network=nw,channel=pr['channels'][d],occurrence=r['detectors'][d]['occurrence'],B_retained=int(np.isfinite(stages['T5'][lo:hi,d]).sum()),B_MAP_contributions=int((pix[lo:hi,d]>=0).sum()))
            records.append(q)
            if q['selected']:
                selected.append(q)
                if q['B_retained']:
                    new.append(q);wl,wh=q['event']['window'];pp=pix[wl:wh,d];footprint.ravel()[pp[pp>=0]]=True
        # At most one selected and one quiet control excerpt per network.
        chosen=[q for q in new if q['network']==nw][:1]
        candidates=[q for q in rows if not q['selected'] and q['B_retained'] and q['eligible']]
        for q in candidates:
            d=q['detector'];lo,hi=q['native_interval'];anchor=t[lo]
            found=None
            for k in range(8,int((t[hi-1]-anchor-2)/.125)+1):
                wl,wh=np.searchsorted(t,[anchor+k*.125-1,anchor+k*.125+2]);good=vx[wl:wh,d]&vr[wl:wh,d]&mapping[wl:wh,d]&(np.hypot(lon[wl:wh,d],lat[wl:wh,d])>60)&np.isfinite(stages['T5'][wl:wh,d])
                if wh>wl and good.all():found=[int(wl),int(wh)];break
            if found:
                q=dict(q,event=dict(window=found),control=True);chosen.append(q);pp=pix[slice(*found),d];quiet.ravel()[pp[pp>=0]]=True;break
        for q in chosen:
            j=q['detector'];lo,hi=q['event']['window'];key=f"n{nw}_"+('control' if q.get('control') else 'selected')
            excerpts.append(dict(key=key,network=nw,detector=j,channel=pr['channels'][j],first=lo,past_last=hi,physical_run=q['physical_run'],control=q.get('control',False)))
            for name,z in dict(time=t-t[0],x=x[:,j],r=rr[:,j],valid_x=vx[:,j],valid_r=vr[:,j],mapping=mapping[:,j],distance=distance[:,j],CAL=stages['CAL'][:,j],M0=stages['M0'][:,j],T5=stages['T5'][:,j],map_pixel=pix[:,j]).items():packed[key+'_'+name]=z[lo:hi]
    endpoint=np.load(base/'run/BASE-POINT/REF/a1100/endpoint/model.npz')
    require(len(excerpts)<=8,'bounded excerpt budget')
    np.savez_compressed(dest/'regions.npz',selected_footprint=footprint,quiet_footprint=quiet,background=endpoint['background_mask'].astype(bool))
    np.savez_compressed(dest/'B-excerpts.npz',**packed)
    save_json(dest/'EXCERPTS.json',excerpts)
    save_json(dest/'SELECTION.json',dict(policy=bound(D/'SELECTOR_POLICY.md'),records=records,selected_runs=len(selected),newly_retained_runs=len(new),selected=[{k:q[k] for k in ('network','detector','physical_run','native_interval','event','B_retained')} for q in selected],newly_retained=[{k:q[k] for k in ('network','detector','physical_run','native_interval')} for q in new],regions=bound(dest/'regions.npz'),excerpts=bound(dest/'EXCERPTS.json'),selection_uses='Original-domain compiled owner only; B native retained counts decide applicability, never thresholds or membership. Consequence footprints use first qualifying events. Quiet controls are first eligible unselected retained detector/window per network, not selected by Q or lowest Q variance.',no_Q_examined=True))
    return len(new)

def response_network(root,family,nw):
    base=root/f'{family}-REF/upstream/products/network{nw}';probe=base/'source-probe';pr=yread(probe/'receipt.yaml');n,d=pr['native_rows'],pr['detectors'];shape=n,d
    a=dict(shape=shape,channels=pr['channels'],probe_receipt=pr,mult=arr(probe/'cal-multiplier.f64','<f8',shape),valid=arr(probe/'input-valid.u8','u1',shape).astype(bool),mapping=arr(probe/'mapping-available.u8','u1',shape).astype(bool),filtered=np.full(shape,np.nan),state=np.zeros(shape,'u1'),center=np.zeros(shape,'u1'))
    for j,c in enumerate(a['channels']):
        q=base/'donor-continuity';a['filtered'][:,j]=arr(q/f'{c}-filtered.f64','<f8',(n,2))[:,0];a['state'][:,j]=arr(q/f'{c}-state.u8','u1');a['center'][:,j]=arr(q/f'{c}-map-center.u8','u1')
    return a

def residual_summary(a,center=None,scale=None):
    a=np.asarray(a);a=a[np.isfinite(a)];r=stats(a)
    if len(a):
        center=float(np.median(a)) if center is None else center
        scale=float(1.4826*np.median(abs(a-center))) if scale is None else scale
        r.update(minimum=float(a.min()),maximum=float(a.max()),q01_q05_q95_q99=np.percentile(a,[1,5,95,99]).tolist(),tail_center=center,tail_scale=scale,
                 tail6=int((abs(a-center)>6*scale).sum()) if scale>0 else None,tail12=int((abs(a-center)>12*scale).sum()) if scale>0 else None)
    return r

def canonical(m):
    p=np.asarray(m['guard']['accepted_parameters']);major=max(p[3],p[4]);minor=min(p[3],p[4]);angle=p[5]+(np.pi/2 if p[4]>p[3] else 0)
    return dict(peak=float(p[0]),cx=float(p[1]),cy=float(p[2]),FWHM_major=float(major*np.sqrt(8*np.log(2))),FWHM_minor=float(minor*np.sqrt(8*np.log(2))),major_angle=float(.5*np.arctan2(np.sin(2*angle),np.cos(2*angle))),angle_informative=bool(major/minor>1.05 and not m['fit']['constrained']),fit_constrained=m['fit']['constrained'],template=m['frozen_template_projection'],apertures=m['apertures'])

def compare():
    dest=ROOT/'report';dest.mkdir(exist_ok=True);base=ROOT/'B-REF';qrun=ROOT/'Q-REF'
    p=population_module(qrun,dest/'Q-population');p.main()
    b,h=load(final_product(base)/'total.fits');q,hq=load(final_product(qrun)/'total.fits')
    require(all(h[k]==hq[k] for k in ('CRPIX1','CRPIX2','CDELT1','CDELT2','CRVAL1','CRVAL2')),'B/Q map geometry differs')
    br,_=load(final_product(base)/'residual.fits');qr,_=load(final_product(qrun)/'residual.fits');br=br['PRIMARY'];qr=qr['PRIMARY'];bv=b['SCI_SUPPORT'].astype(bool);qv=q['SCI_SUPPORT'].astype(bool)
    regions=dict(np.load(ROOT/'selection/regions.npz'));xx,yy=coordinates(br.shape,h);regions['fit_disk']=np.hypot(xx,yy)<=40;regions['B_supported']=bv
    metrics={}
    for name,region in regions.items():
        domain=region&bv;common=domain&qv;lost=domain&~qv
        metrics[name]=dict(B_pixels=int(domain.sum()),common_pixels=int(common.sum()),Q_lost_pixels=int(lost.sum()),B=residual_summary(br[domain]),B_common=residual_summary(br[common]),Q_common=residual_summary(qr[common]),delta_common=residual_summary((qr-br)[common]),B_lost=residual_summary(br[lost]),lost_pixels=[dict(row=int(i),column=int(j),B_residual=float(br[i,j]),B_hits=float(b['HITS'][i,j]),Q_hits=float(q['HITS'][i,j])) for i,j in np.argwhere(lost)])
        frozen=residual_summary(br[domain])
        if frozen['n']:
            metrics[name]['B_common_fixed_tail']=residual_summary(br[common],frozen['tail_center'],frozen['tail_scale'])
            metrics[name]['Q_common_fixed_tail']=residual_summary(qr[common],frozen['tail_center'],frozen['tail_scale'])
    support=[];membership=[];packed={}
    expected=read(ROOT/'selection/SELECTION.json')['selected'];e_keys={(x['network'],x['detector'],x['physical_run']) for x in expected}
    for nw in range(1,5):
        original=base/f'upstream/products/network{nw}';bp=yread(original/'source-probe/receipt.yaml')
        for role in [r for r in ROLES if r.startswith('Q-')]:
            rt=ROOT/role/f'upstream/products/network{nw}';record=yread(rt/'source-probe/run-selection.yaml');pr=yread(rt/'source-probe/receipt.yaml');actual={(nw,x['detector'],x['physical_run']) for x in record['population'] if x['selected']}
            refkeys={x for x in e_keys if x[0]==nw}
            if role=='Q-REF':require(actual==refkeys,'saved selector / active unmodified selector disagreement')
            membership.append(dict(role=role,network=nw,selected=sorted(actual),gained_vs_unmodified=sorted(actual-refkeys),lost_vs_unmodified=sorted(refkeys-actual),manifest=bound(rt/'source-probe/run-selection.yaml')))
            require(pr['original_x_sha256']==bp['original_x_sha256'] and pr['original_r_sha256']==bp['original_r_sha256'] and pr['r_unchanged'] and pr['input_validity_unchanged'],'original acquisition changed')
            for item in record['population']:
                if item['selected']:
                    lo,hi=item['native_interval'];ch=pr['channels'][item['detector']]
                    state=arr(rt/f'donor-continuity/{ch}-state.u8','u1');center=arr(rt/f'donor-continuity/{ch}-map-center.u8','u1');require(not (state[lo:hi]&3).any() and not center[lo:hi].any(),'quarantined run still contributes')
        qbase=qrun/f'upstream/products/network{nw}';lost=gained=0
        for ch in bp['channels']:
            bs=arr(original/f'donor-continuity/{ch}-state.u8','u1');qs=arr(qbase/f'donor-continuity/{ch}-state.u8','u1');bv2=(bs&3)==3;qv2=(qs&3)==3;lost+=int((bv2&~qv2).sum());gained+=int((qv2&~bv2).sum())
        support.append(dict(network=nw,RTC_paired_lost=lost,RTC_paired_gained=gained))
    for ex in read(ROOT/'selection/EXCERPTS.json'):
        p=population_module(qrun,dest/'Q-population');_,_,_,stages,_,_,_,_,_=p.network(ex['network']);packed[ex['key']+'_Q_T5']=stages['T5'][ex['first']:ex['past_last'],ex['detector']]
    np.savez_compressed(dest/'Q-excerpts.npz',**packed)
    measurements={}
    for name in ROLES:
        r=read(ROOT/name/'run/BASE-POINT/REF/a1100/SUMMARY.json');measurements[name]=dict(M0=canonical(r['histories'][0]['measurement']),T5=canonical(r['histories'][5]['measurement']),endpoint_uncertainty=r['endpoint_uncertainty'])
    save_json(dest/'metrics.json',dict(residuals=metrics,native_support=support,membership=membership,POINT=measurements,support=dict(B_pixels=int(bv.sum()),Q_pixels=int(qv.sum()),lost=int((bv&~qv).sum()),gained=int((qv&~bv).sum()),B_hits=float(b['HITS'].sum()),Q_hits=float(q['HITS'].sum()),normalization_delta=stats((q['NORMALIZATION']-b['NORMALIZATION'])[bv])),qualifications=['Actual post-PTC residual.fits, never sky-restored total or total minus fitted Gaussian.','Q−B includes downstream relearning. Lost pixels are unavailable, not cleaned.','Endpoint scales are conditional single-map context, not paired-test significance.','Major-axis angles are meaningful only for resolved noncircular unconstrained fits.','No unattended adoption or parking decision.']))
    np.savez_compressed(dest/'maps.npz',B=b['PRIMARY'],Q=q['PRIMARY'],B_residual=br,Q_residual=qr,B_support=bv,Q_support=qv,difference=np.where(bv&qv,qr-br,np.nan),normalization_difference=q['NORMALIZATION']-b['NORMALIZATION'],**regions)
    import point_rtc_response as response
    response.load_network=response_network;oldload=response.load
    def patched(path):
        data,header=oldload(path)
        if Path(path).parent.name=='shapes' and 'INJECTION' not in data:data['INJECTION']=data['PRIMARY']
        return data,header
    response.load=patched
    for family in ('B','Q'):response.response(ROOT,family,dest/('response-'+family),reference_name=family+'-REF')
    render(dest)
    paths=[x for folder in (ROOT/'selection',dest) for x in sorted(folder.rglob('*')) if x.is_file()]
    save_json(dest/'MANIFEST.json',[dict(**bound(x),bytes=x.stat().st_size) for x in paths])

def render(dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=np.load(dest/'maps.npz');m=read(dest/'metrics.json');fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,name,support in zip(axes.flat,['B_residual','Q_residual','difference','B_support','Q_support','normalization_difference'],['B_support','Q_support',None,None,None,None]):
        a=z[name];a=np.where(z[support],a,np.nan) if support else a;im=ax.imshow(a,origin='lower');ax.set_title(name);fig.colorbar(im,ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(dest/'maps.png');plt.close(fig)
    ex=read(ROOT/'selection/EXCERPTS.json');b=np.load(ROOT/'selection/B-excerpts.npz');q=np.load(dest/'Q-excerpts.npz');fig,axes=plt.subplots(max(1,len(ex)),2,figsize=(11,max(3,2*len(ex))),squeeze=False)
    for row,e in enumerate(ex):
        key=e['key'];t=b[key+'_time'];ax=axes[row,0];ax.plot(t,b[key+'_x'],label='original x');ax.plot(t,b[key+'_r'],label='original r');ax.set_title(key);ax.legend(fontsize=7)
        ax=axes[row,1];ax.plot(t,b[key+'_T5'],label='B PTC residual');ax.plot(t,q[key+'_Q_T5'],label='Q PTC residual');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(dest/'excerpts.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for family in ('B','Q'):
        rows=[r for r in read(dest/f'response-{family}/source-responses.json') if r['cycle']==5]
        # Use the established response owner's field names without reinterpretation.
        for r in rows:axes[0].text(.02,.95-.13*(list(('B','Q')).index(family)*3+rows.index(r)),family+' '+r['branch']+' '+str(r['spatial']),fontsize=7,wrap=True)
    axes[0].axis('off');names=['peak','FWHM_major','FWHM_minor'];a=m['POINT']['B-REF']['T5'];c=m['POINT']['Q-REF']['T5'];axes[1].bar(names,[c[n]-a[n] for n in names]);axes[1].set_title('Q−B POINT parameters (native units)')
    fig.tight_layout();fig.savefig(dest/'response-point.png');plt.close(fig)
    rows=read(ROOT/'selection/SELECTION.json')['records'];fig,ax=plt.subplots(figsize=(8,3));ax.bar(range(1,5),[sum(q['selected'] and q['network']==nw for q in rows) for nw in range(1,5)]);ax.set(xlabel='Network',ylabel='Selected physical detector/runs',xticks=range(1,5));fig.tight_layout();fig.savefig(dest/'selection.png');plt.close(fig)


def no_applicable_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b=ROOT/'B-REF';z,h=load(final_product(b)/'total.fits');r,_=load(final_product(b)/'residual.fits')
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    for ax,name,a in zip(axes,['B T5 total','B actual PTC residual','B science support'],[z['PRIMARY'],r['PRIMARY'],z['SCI_SUPPORT']]):
        im=ax.imshow(np.where(z['SCI_SUPPORT'],a,np.nan),origin='lower');ax.set_title(name);fig.colorbar(im,ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(ROOT/'selection/B-no-applicable-case.png');plt.close(fig)
    paths=[p for p in sorted((ROOT/'selection').rglob('*')) if p.is_file()]
    save_json(ROOT/'selection/MANIFEST.json',[dict(**bound(p),bytes=p.stat().st_size) for p in paths])
