"""Fixed-generation T2 maps on the two explicitly approved legacy exports."""
import time
import netCDF4
import numpy as np
from scipy import sparse
from core import (ARRAYS,POLICIES,legacy,require,write_json,digest,weight_generation,
                  pixel_indices,accumulate,source_metrics,ring_metrics)

def generation(data,run):
    start=time.monotonic();nd=len(data['arr']);stats=[[],[],[]]
    for j in range(3):
        rows=data['chunk']//4==j
        for result,value in zip(stats,legacy.moments(data['b'][rows],data['valid'][rows])):result.append(value)
    n,mu,v=map(np.stack,stats);e=np.stack([data['eval_n'][4*j:4*j+4].sum(axis=0) for j in range(3)])
    result=dict(n=n,mu=mu,v=v,e=e,gamma=np.full(v.shape,np.nan),raw=np.full(v.shape,np.nan),reason=np.full(v.shape,1,np.uint8),available=np.zeros(3,bool),mean=np.full(3,np.nan))
    dispersion=[]
    for a,name in enumerate(ARRAYS):
        cols=data['arr']==a;gen=weight_generation(n[:,cols],mu[:,cols],v[:,cols],e[:,cols])
        for k in ('gamma','raw','reason'):result[k][:,cols]=gen[k]
        result['available'][a]=gen['available'];result['mean'][a]=gen['mean']
        sd=np.sort(np.sqrt(gen['v'][gen['H']]))
        dispersion.append(dict(array=name,required_groups=int((gen['e']>0).sum()),trained_groups=len(sd),fallback_groups=int(gen['L'].sum()),q10_sd=sd[int(np.floor(.1*(len(sd)-1)))] if len(sd) else np.nan,q90_sd=sd[int(np.floor(.9*(len(sd)-1)))] if len(sd) else np.nan))
    path=run.out/f"T2_{data['obsnum']}_generation.npz"
    np.savez_compressed(path,**result,column=np.arange(nd),uid=data['uid'],network=data['nw'],array=data['arr'],training_mask=data['valid'],training_chunk=data['chunk'],training_time=data['times'],evaluation_counts_by_chunk=data['eval_n'],central_evaluation_counts_by_chunk=data['central_n'],training_exclusions=data['exclusions'],evaluation_exclusions=data['eval_exclusions'],raw_signal_bad=data['raw_signal_bad'],edges=data['edges'])
    write_json(run.out/f"T2_{data['obsnum']}_GENERATION_LOCK.json",dict(generation_path=path.name,sha256=digest(path),evaluation_signal_moments_consumed=False,training_dispersion=dispersion,seconds=time.monotonic()-start,rule='N4U exact r0.6; this generation is fixed for all following full/window/response maps'))
    run.check(f"T2 {data['obsnum']} generation locked before evaluation mapping")
    return result

def load_evaluation(item,data,run):
    """Called only after the corresponding immutable generation has been saved."""
    rows={k:[] for k in ['x','y','b','column','storage_row','window']}
    with netCDF4.Dataset(item['path'],'r') as f:
        base={k:legacy.read_plain(f,k) for k in ['TelElAct','az_phys','alt_phys','pointing_offset_az','pointing_offset_alt','apt_x_t','apt_y_t']}
        for c,(lo,hi) in enumerate(zip(data['edges'][:-1],data['edges'][1:])):
            lo=lo+(hi-lo)//2
            b,finite,_=legacy.signal_values(f['signal'],slice(lo,hi))
            flags=legacy.read_plain_slice(f,'flags',slice(lo,hi));el=base['TelElAct'][lo:hi,None]
            x=base['az_phys'][lo:hi,None]+(np.cos(el)*base['apt_x_t']-np.sin(el)*base['apt_y_t']+base['pointing_offset_az'][lo:hi,None])*np.pi/648000
            y=base['alt_phys'][lo:hi,None]+(np.cos(el)*base['apt_y_t']+np.sin(el)*base['apt_x_t']+base['pointing_offset_alt'][lo:hi,None])*np.pi/648000
            use=(flags==0)&finite&np.isfinite(x)&np.isfinite(y)
            require(np.array_equal(use.sum(axis=0),data['eval_n'][c]),'Post-lock evaluation census changed')
            t,d=np.nonzero(use)
            for k,z in dict(x=x[use]*648000/np.pi,y=y[use]*648000/np.pi,b=b[use],column=d.astype(np.uint16),storage_row=(t+lo).astype(np.uint16),window=np.full(len(d),c//4,np.uint8)).items():rows[k].append(z)
            run.check(f"T2 {item['obsnum']} evaluation chunk {c}")
    return {k:np.concatenate(v) for k,v in rows.items()}

def grid(ev,R,limit):
    j,k=pixel_indices(ev['x'],ev['y'],2.)
    boundary=int(np.floor(3*R/2));j0=min(int(j.min()),-boundary);j1=max(int(j.max()),boundary);k0=min(int(k.min()),-boundary);k1=max(int(k.max()),boundary)
    shape=(k1-k0+1,j1-j0+1);require(np.prod(shape)<=limit,'Grid resource limit exceeded; no crop allowed')
    pixel=(k-k0)*shape[1]+j-j0;ky,jx=np.mgrid[k0:k1+1,j0:j1+1];x=2*jx.ravel();y=2*ky.ravel();r2=x*x+y*y
    regions=dict(C=r2<=R*R,O1=(r2>R*R)&(r2<=4*R*R),O2=(r2>4*R*R)&(r2<=9*R*R));regions['O']=regions['O1']|regions['O2'];regions['D']=regions['C']|regions['O']
    T=np.exp(-4*np.log(2)*r2/(R/3)**2)*regions['C']
    return dict(pixel=pixel.astype(np.int32),shape=shape,origin=(j0,k0),x=x,y=y,T=T,regions=regions,R=R)

def influence(pixel,column,gamma,fallback,npix,nd,regions):
    mat=sparse.coo_matrix((gamma,(column,pixel)),shape=(nd,npix)).tocsr()
    total=np.asarray(mat.sum(axis=0)).ravel();maxdet=mat.max(axis=0).toarray().ravel()
    fcount=np.bincount(pixel[fallback],minlength=npix);fq=np.bincount(pixel[fallback],weights=gamma[fallback],minlength=npix)
    dsum=np.asarray(mat.sum(axis=1)).ravel()
    agg={}
    for name,mask in dict(all=np.ones(npix,bool),**regions).items():
        dmass=np.asarray(mat[:,mask].sum(axis=1)).ravel();den=float(dmass.sum());hit=mask[pixel]
        count=int(hit.sum())
        agg[name]=dict(count=count,Q=den,max_detector_share=float(dmass.max()/den) if den>0 else np.nan,fallback_count=int(fcount[mask].sum()),fallback_Q=float(fq[mask].sum()),fallback_count_share=float(fcount[mask].sum()/count) if count else np.nan,fallback_Q_share=float(fq[mask].sum()/den) if den>0 else np.nan)
    share=np.divide(maxdet,total,out=np.full(npix,np.nan),where=total>0)
    return dict(max_detector_share=share,fallback_count=fcount,fallback_Q=fq,detector_Q=dsum),agg

def pair_metrics(pair,g):
    regions=g['regions'];native=[];common=[];support_rows={};npix=len(g['x'])
    intersection=pair[0]['S_sci']&pair[1]['S_sci']
    for a,m in enumerate(pair):
        complete={r:bool(np.all(m['S_sci'][mask])) for r,mask in regions.items()}
        nr={r:ring_metrics(m['map'],mask,g['shape']) if complete[r] else dict(pixels=int(mask.sum()),available=False,missing=int((mask&~m['S_sci']).sum())) for r,mask in regions.items() if r in ['O','O1','O2']}
        ns=source_metrics(m['map'],regions['C'],regions['O'],g['T'],g['x'],g['y']) if complete['C'] and complete['O'] else source_metrics(m['map'],np.zeros(npix,bool),np.zeros(npix,bool),g['T'],g['x'],g['y'])
        native.append(dict(complete=complete,rings=nr,source=ns))
        cr={r:ring_metrics(m['map'],regions[r]&intersection,g['shape']) for r in ['O','O1','O2']}
        cs=source_metrics(m['map'],regions['C']&intersection,regions['O']&intersection,g['T'],g['x'],g['y'])
        common.append(dict(rings=cr,source=cs,interpretation='intersection diagnostic only'))
    for r,mask in regions.items():
        support_rows[r]=dict(required=int(mask.sum()))
        for s in ['S_norm','S_sci']:
            support_rows[r][s]=dict(U=int((mask&pair[0][s]).sum()),N4U=int((mask&pair[1][s]).sum()),common=int((mask&pair[0][s]&pair[1][s]).sum()))
    alerts={}
    for r in ['O','O1','O2']:
        u,n=native[0]['rings'][r],native[1]['rings'][r]
        alerts[r+'_uniform_power_loss']=(bool(u['power']>1.1025*n['power']) if u['available'] and n['available'] else None)
        alerts[r+'_U_over_N_power']=(u['power']/n['power'] if u['available'] and n['available'] and n['power']>0 else None)
    u,n=native[0]['source'],native[1]['source'];A=u['amplitude'];B=n['amplitude']
    alerts['amplitude_change']=(bool(abs(B/A-1)>.02) if np.isfinite(A) and np.isfinite(B) and A>0 and B>0 else None)
    alerts['source_unresolved']=not(np.isfinite(A) and np.isfinite(B) and A>0 and B>0)
    for key in ['centroid_x','centroid_y','width_x','width_y']:
        a,b=u[key],n[key]
        alerts[key+'_change']=(bool(abs(a-b)>.02*g['R']/3) if key.startswith('centroid') else bool(abs(b/a-1)>.02)) if np.isfinite(a) and np.isfinite(b) and (not key.startswith('width') or a>0) else None
    d=support_rows['D'];alerts['U_support_loss']=((d['S_sci']['N4U']-d['S_sci']['U'])/d['required']>.01)
    alerts['complete_D']=all(n['complete']['D'] for n in native)
    return dict(native=dict(zip(POLICIES,native)),common=dict(zip(POLICIES,common)),support=support_rows,full_map_alerts=alerts)

def run_t2(run,settings):
    results=[]
    for item in settings['inputs']:
        obs=item['obsnum'];run.check(f'T2 {obs} training and census')
        data=legacy.load_observation(item,settings,run);gen=generation(data,run)
        ev_all=load_evaluation(item,data,run)
        for array,name in enumerate(ARRAYS):
            run.check(f'T2 {obs} {name} maps')
            choose=data['arr'][ev_all['column']]==array;ev={k:v[choose] for k,v in ev_all.items()};R=settings['guard_arcsec'][str(obs)];g=grid(ev,R,settings['grid_pixel_limit']);npix=len(g['x'])
            prefix=f'T2_{obs}_{name}';window=ev['window'];col=ev['column'];pixel=g['pixel'];fixed_gamma=gen['gamma'][window,col]
            fallback=np.isin(gen['reason'][window,col],[2,3,4]);nd=len(data['arr'])
            np.savez_compressed(run.out/(prefix+'_population.npz'),storage_row=ev['storage_row'],column=col,window=window,pixel=pixel,shape=g['shape'],origin=g['origin'],x=g['x'],y=g['y'],template=g['T'],**g['regions'])
            case=dict(obsnum=obs,array=name,occurrences=len(col),grid_shape=g['shape'],maps={},response=[],mapping_seconds=dict(U=0.,N4U=0.),fallback_groups=[])
            products={};pairs={}
            for scope in ['full','window0','window1','window2']:
                select=np.ones(len(col),bool) if scope=='full' else window==int(scope[-1]);pair=[]
                for arm,pname in enumerate(POLICIES):
                    start=time.monotonic();weights=np.ones(select.sum()) if arm==0 else fixed_gamma[select]
                    if arm==1 and not gen['available'][array]:
                        # An explicit unavailable product retains population identity; no partial normalization.
                        m=dict(map=np.full(npix,np.nan),norm_map=np.full(npix,np.nan),numerator=np.full(npix,np.nan),Q=np.full(npix,np.nan),Q2=np.full(npix,np.nan),count=np.bincount(pixel[select],minlength=npix),S_norm=np.zeros(npix,bool),S_sci=np.zeros(npix,bool),cause=np.full(npix,5,np.uint8),Qstar=np.nan,index=-1)
                        inf={};agg={};m['max_detector_share']=np.full(npix,np.nan);m['fallback_count']=np.bincount(pixel[select&fallback],minlength=npix);m['fallback_Q']=np.full(npix,np.nan)
                    else:
                        m=accumulate(pixel[select],ev['b'][select],weights,npix)
                        inf,agg=influence(pixel[select],col[select],weights,fallback[select],npix,nd,g['regions']);m.update(inf)
                    case['mapping_seconds'][pname]+=time.monotonic()-start
                    mapname=f'{scope}_{pname}';np.savez_compressed(run.out/(prefix+'_'+mapname+'.npz'),**m)
                    case['maps'][mapname]=dict(Qstar=m['Qstar'],influence=agg)
                    pair.append(m);products[mapname]=m
                pairs[scope]=pair_metrics(pair,g)
                delta=pair[0]['map']-pair[1]['map']
                np.savez_compressed(run.out/(prefix+'_'+scope+'_difference.npz'),U_minus_N4U=delta,common=pair[0]['S_sci']&pair[1]['S_sci'])
                run.check(f'T2 {obs} {name} {scope} saved')
            # Signed perturbations use the same finite occurrence population and stored generation.
            for arm,pname in enumerate(POLICIES):
                native=products['full_'+pname];weights=np.ones(len(col)) if arm==0 else fixed_gamma
                for sign in [1,-1]:
                    if arm==1 and not gen['available'][array]:continue
                    start=time.monotonic();response=accumulate(pixel,ev['b']+sign*g['T'][pixel],weights,npix);case['mapping_seconds'][pname]+=time.monotonic()-start
                    delta=response['map']-native['map'];error=delta-sign*g['T'];available=native['S_sci']
                    checks={}
                    for region,mask in dict(inside_C=g['regions']['C'],outside_C=~g['regions']['C']).items():
                        use=mask&available;checks[region]=dict(pixels=int(use.sum()),max_absolute_error=float(np.max(np.abs(error[use]))) if use.any() else None)
                    same=np.array_equal(response['S_sci'],native['S_sci']) and np.array_equal(response['S_norm'],native['S_norm'])
                    passed=same and all(c['max_absolute_error'] is None or c['max_absolute_error']<=1e-9 for c in checks.values())
                    response.update(delta=delta,response_error=error)
                    np.savez_compressed(run.out/(prefix+f'_response_{pname}_{sign:+d}.npz'),**response)
                    case['response'].append(dict(policy=pname,sign=sign,same_support=same,regions=checks,passed=passed))
                    require(passed,'Fixed-state response arithmetic gate failed')
            temporal={}
            for pname in POLICIES:
                ms=np.stack([products[f'window{j}_{pname}']['map'] for j in range(3)])
                ss=np.stack([products[f'window{j}_{pname}']['S_sci'] for j in range(3)]);all_good=np.all(ss,axis=0)
                vals=np.full(npix,np.nan);vals[all_good]=np.max(ms[:,all_good],axis=0)-np.min(ms[:,all_good],axis=0)
                fields=dict(range=vals,common_three=all_good)
                info={}
                for a,b in [(0,1),(0,2),(1,2)]:
                    use=ss[a]&ss[b]&g['regions']['D'];diff=np.where(use,ms[b]-ms[a],np.nan);fields[f'window{b}_minus_window{a}']=diff
                    info[f'{b}-{a}']=dict(required=int(g['regions']['D'].sum()),common=int(use.sum()),complete_D=bool(np.all(use[g['regions']['D']])),common_mean_difference=float(diff[use].mean()) if use.any() else np.nan,common_mean_squared_difference=float(np.mean(diff[use]**2)) if use.any() else np.nan)
                np.savez_compressed(run.out/(prefix+f'_temporal_{pname}.npz'),**fields);temporal[pname]=info
            for j,c in zip(*np.nonzero(np.isin(gen['reason'],[2,3,4])&(data['arr'][None,:]==array)&(gen['e']>0))):
                select=(col==c)&(window==j);p=pixel[select];qv=fixed_gamma[select]
                by={}
                for r,mask in g['regions'].items():
                    hit=mask[p];by[r]=dict(count=int(hit.sum()),Q=float(qv[hit].sum()),signal_numerator=float(np.sum(qv[hit]*ev['b'][select][hit])))
                case['fallback_groups'].append(dict(window=int(j),column=int(c),uid=data['uid'][c],reason=int(gen['reason'][j,c]),n=int(gen['n'][j,c]),evaluation_count=int(select.sum()),regions=by))
            case['comparisons']=pairs;case['temporal']=temporal
            write_json(run.out/(prefix+'_summary.json'),case);results.append(case)
            del ev,products,pairs
        del data,ev_all
    write_json(run.out/'T2_SUMMARY.json',results)
    return results
