"""Assess the fixed guarded matrix without native execution or adaptive inference."""
import argparse,time,json
from pathlib import Path
import numpy as np
from run_point_guarded_feedback import authenticated,ctx,read,ARRAYS,POLICIES,BRANCHES
from run_point_response_cd import AB,source
from point_guarded_feedback import infer,publish,ideal_comparison
from point_expected import candidate
from point_response_cd import injection_reference,response_metrics,vector_measurement,MEASUREMENT_NAMES,shape_projection
from report_point_response_cd import aligned,MEASUREMENT_UNITS,fmt
from point_fruit_diagnostics import load,patch_map
from point_fruit_trajectory import verified,bound,manifest,verify_manifest,selection_difference
from point_empirical import source_patch,stats,coordinates
from point_empirical_io import write_fits
from point_additive import NoiseRequest,learn
from point_corr_admission import Neighborhood
from point_post_fruit import applicability,operators
from point_additive_diagnostics import save_json
from inspect_forward_map import require,read_yaml

SNAPS=(0,1,3,5,10)

def increments(a,b):
    av,bv=vector_measurement(a),vector_measurement(b)
    return dict(zip(MEASUREMENT_NAMES,av-bv)) if av is not None and bv is not None else None

def metrics(delta,known,domain):
    if not domain.any() or not np.any(known[domain]):return None
    require(np.isfinite(delta[domain]).all() and np.isfinite(known[domain]).all(),'unavailable response values')
    r=response_metrics(delta,known,domain);r['rho']=float(np.linalg.norm((delta-r['g']*known)[domain])/np.linalg.norm(known[domain]))
    require(abs(r['eta']**2-(r['g']-1)**2-r['rho']**2)<1e-10,'response identity')
    return r

def profile(v,valid,c):
    f=c['fixed']
    return [float(np.mean(v[m])) if m.any() else None for m in [valid&(f.radius>=lo)&(f.radius<hi) for lo,hi in zip(f.edges[:-1],f.edges[1:])]]

def history(summary):
    rows=[]
    for state in summary['histories']:
        k=state['cycle_index'];m=state.get('measurement');row=dict(cycle=k,ready=state['ready'],stop_reason=state.get('stop_reason'),measurement=m,admission=state.get('admission'),PTC_segments=state.get('PTC_segments'),guards=state.get('guards'),turnover=state.get('candidate_turnover'))
        if m:row['measurements']=dict(zip(MEASUREMENT_NAMES,vector_measurement(m))) if vector_measurement(m) is not None else None
        if state.get('candidate'):
            v,_=load(verified(state['candidate']));sel=v['ADMITTED'].astype(bool)
            row['actual_candidate']=dict(selected_pixels=int(sel.sum()),brightness_area=float(np.sum(v['CANDIDATE'][sel])*4),path=state['candidate'])
        if rows and m and rows[-1]['measurement']:row['previous_cycle_drift']=increments(m,rows[-1]['measurement'])
        rows.append(row)
    return rows

def endpoint(out,p,c,a,pol,state,identity):
    """One existing-family fit, original frozen masks, diagnostic use only."""
    t=time.perf_counter();prod=verified(state['total']).parent;im,h=load(prod/'total.fits')
    oldroot=AB/'companion'/a;old=read(oldroot/'M0/model.json');domains=dict(np.load(oldroot/'frozen-domains.npz'))
    original_image,oh=load(Path(p['M0'][a]['REF']['total']['path']))
    _,_,union,place=aligned(im['PRIMARY'],h,original_image['PRIMARY'],oh)
    require(union['CRPIX1']==h['CRPIX1'] and union['CRPIX2']==h['CRPIX2'] and place(im['PRIMARY'],h).shape==im['PRIMARY'].shape,'endpoint grid differs from frozen mask grid')
    frozen_support=place(domains['common_support'],oh)==1;source_mask=place(domains['source_mask'],oh)==1
    supported=im['SCI_SUPPORT'].astype(bool)&frozen_support
    if (out/'learned.json').exists():
        meta=read(out/'learned.json');require(meta['identity']==identity,'cached endpoint identity');model=dict(np.load(out/'learned.npz'));model.update(summary=meta['summary'],regions=meta['regions'])
    else:
        marker=out/'FIT_ATTEMPT.json'
        require(not marker.exists(),'previous incomplete endpoint fit attempt; no retry')
        save_json(marker,dict(identity=identity,single_fit_attempt=True))
        model=learn(im['PRIMARY'],im['NORMALIZATION'],supported,h,c['parameters'],NoiseRequest(**old['request']),stamp_size=49,source_mask=source_mask)
        np.savez(out/'learned.npz',**{k:v for k,v in model.items() if isinstance(v,np.ndarray)})
        save_json(out/'learned.json',dict(summary=model['summary'],regions=model['regions'],identity=identity,single_existing_family_fit=True))
    ps,pv,_,_,_=source_patch(model['scale'],model['valid'],im['NORMALIZATION'],h)
    nb=Neighborhood(c['header'],ps.shape,c['fixed'].beam,ps,model['psd'],c['k0'],empirical=model['empirical'])
    sig,sv,_,_=patch_map(prod/'total.fits');base=candidate(sig,sv,c['fixed'],state['measurement']['guard'])
    fields=c['neighborhood'].scores(base);newfields=nb.scores(base);comp={}
    w=nb.weights[None,:]*(~(c['k0']|base['selected']).ravel()[nb.indices]);w[:,nb.center]=0
    footprint_valid=np.zeros(sig.shape,bool);weighted_valid=footprint_valid.copy()
    footprint_valid.ravel()[nb.anchor_indices]=nb.inside&pv.ravel()[nb.indices].all(axis=1)
    weighted_valid.ravel()[nb.anchor_indices]=nb.inside&(pv.ravel()[nb.indices]|(w==0)).all(axis=1)
    planes=[('BACKGROUND_SCALE',model['scale'],'mJy/nominal beam'),('AMPLITUDE_VALID',model['valid'].astype('u1'),'1'),('BACKGROUND_MASK',model['background_mask'].astype('u1'),'1'),('SOURCE_MASK',source_mask.astype('u1'),'1')]
    for policy in ('CORR','DIAG'):
        x,y=fields['policies'][policy],newfields['policies'][policy];use=x['available']&y['available'];rat=np.full(sig.shape,np.nan);rat[use]=y['variance'][use]/x['variance'][use]
        comp[policy]=dict(matched_pixels=int(use.sum()),ratio_quantiles=np.quantile(rat[use],[0,.1,.5,.9,1]) if use.any() else None,anchor_amplitude_interpolation_valid_pixels=int((use&pv).sum()),full_footprint_amplitude_valid_pixels=int((use&footprint_valid).sum()),nonzero_weight_amplitude_valid_pixels=int((use&weighted_valid).sum()),amplitude_validity_note='coverage interpolation flags only; background-to-source covariance transfer still unvalidated',frozen_numerical_failures=x['numerical_failures'],endpoint_numerical_failures=y['numerical_failures'])
        write_fits(out/f'{policy}-matched-variance.fits',c['header'],[(n,v,u) for n,v,u in [('FROZEN',x['variance'],'(mJy/nominal beam)^2'),('ENDPOINT',y['variance'],'(mJy/nominal beam)^2'),('RATIO',rat,'1'),('MATCHED',use.astype('u1'),'1')]],companion='POINT-GUARDED-ENDPOINT')
    receipt=read_yaml(prod/'receipt.yaml');op=np.memmap(prod/'operator-pixel.i32',dtype='<i4',mode='r');contributions={}
    for s in receipt['segments']:
        if not s['retained']:continue
        n=str(s['source_segment']['network']);v=op[s['MAP_first_input_row']:s['MAP_first_input_row']+s['retained']];v=v[v>=0]
        contributions.setdefault(n,np.zeros(im['PRIMARY'].shape,dtype=np.int64));contributions[n]+=np.bincount(v,minlength=im['PRIMARY'].size).reshape(im['PRIMARY'].shape)
    require(np.array_equal(sum(contributions.values()),im['HITS']),'endpoint contribution accounting')
    transfer=applicability(im['PRIMARY'],im['NORMALIZATION'],h,c['parameters'],model,contributions,operators(c['fixed']))
    rho=np.fft.ifft2(model['psd']).real;rho0=np.fft.ifft2(c['psd']).real;lag=np.arange(11)
    oldmodel=dict(np.load(oldroot/'M0/model.npz'));both=model['background_mask']&oldmodel['background_mask']&np.isfinite(oldmodel['scale'])
    ratio=model['scale'][both]/oldmodel['scale'][both]
    result=dict(status='available',identity=identity,array=a,policy=pol,summary=model['summary'],background=stats(im['PRIMARY'][model['background_mask']]),regions=model['regions'],applicability=transfer,
        matched_neighborhood_variance=comp,background_scale_ratio_to_C0_quantiles=np.quantile(ratio,[0,.1,.5,.9,1]),
        directional=dict(lag_arcsec=lag*2,endpoint_east=rho[0,lag],endpoint_north=rho[lag,0],frozen_east=rho0[0,lag],frozen_north=rho0[lag,0]),
        completed_125_stress_report=bound(Path(p['guard_root'])/'report/SUMMARY.json'),
        interpretation='Transferred background family, not source-location noise truth. Frozen sigma_A stays a conditional reference scale. No covariance fed back.',wall_seconds=time.perf_counter()-t)
    np.savez(out/'model.npz',**{k:v for k,v in model.items() if isinstance(v,np.ndarray)})
    write_fits(out/'background.fits',h,planes,companion='POINT-GUARDED-ENDPOINT')
    return result

def assess(root):
    start=time.perf_counter();revision=source();p=authenticated(root);out=root/'report';out.mkdir(exist_ok=True)
    summaries={};matrix=[];runtime=[];attempted=list((root/'run').glob('*/*/*/cycle-*/request.json'));receipts=list((root/'run').glob('*/*/*/cycle-*/native/products/receipt.yaml'))
    require(len(attempted)<=450 and len(receipts)<=len(attempted),'feedback bound')
    for a in ARRAYS:
      for pol in POLICIES:
       for b in BRANCHES:
        f=root/'run'/pol/b/a/'SUMMARY.json'
        if f.exists():
            verify_manifest(f.parent);s=read(f);summaries[a,pol,b]=s
            matrix.append(dict(array=a,policy=pol,branch=b,status='T10 complete' if s['ready'] and s['completed_cycle']==10 else 'stopped',cycle=s['completed_cycle'],reason=s['stop_reason']))
        else:
            blocked=[r for phase in ('profile','REF','compact','shoulder') for r in read(root/f'FEEDBACK-{phase}.json')['blocked'] if (r['array'],r['policy'],r['branch'])==(a,pol,b)]
            matrix.append(dict(array=a,policy=pol,branch=b,status='blocked',reason=blocked))
    for path in receipts:
        r=read_yaml(path);run=read(path.parent.parent/'RUN.json');runtime.append(dict(receipt=bound(path),wall_seconds=r['wall_seconds'],cpu_seconds=r['cpu_seconds'],rss_bytes=r['peak_rss_native_units']*(1 if r['peak_rss_unit']=='bytes' else 1024),learn_seconds=sum(x['learning_seconds'] for x in r['segments']),apply_seconds=sum(x['application_seconds'] for x in r['segments']),MAP_seconds=r['MAP_seconds'],output_bytes=run['output_bytes']))
    arrays={};ideal_count=0;endpoint_count=0
    for a in ARRAYS:
      c=ctx(p,a);ad=out/a;ad.mkdir(exist_ok=True);sh,shh=load(verified(p['original_owner']['injections'][a]['C-shoulder']));array=dict(policies={},historical={})
      for pol in POLICIES:
        d=ad/pol;d.mkdir(exist_ok=True);ref=summaries.get((a,pol,'REF'));results=dict(history={},branches={},even=[],cross_candidates={})
        if ref is None:array['policies'][pol]=results;continue
        refs={s['cycle_index']:s for s in ref['histories']}
        for b in BRANCHES:
            s=summaries.get((a,pol,b))
            if s:results['history'][b]=history(s)
        if ref['ready'] and ref['completed_cycle']==10:
            st=refs[10];sig,sv,_,_=patch_map(verified(st['total']));g=st['measurement']['guard'];base=candidate(sig,sv,c['fixed'],g);fields=c['neighborhood'].scores(base)
            for selectpol in POLICIES:
                _,summary=publish(d/'cross-candidates'/selectpol,c,sig,(g,base,fields),selectpol,3. if selectpol=='BASE-POINT' else p['thresholds'][a][selectpol]);results['cross_candidates'][selectpol]=summary
            ed=d/'endpoint';ed.mkdir(exist_ok=True);endpoint_count+=1
            endpoint_identity=dict(map=st['total'],masks=bound(AB/'companion'/a/'frozen-domains.npz'),request=bound(AB/'companion'/a/'M0/model.json'),frozen_context=bound(Path(p['correlation_root'])/'freeze'/a/'inputs.npz'))
            if (ed/'RESULTS.json').exists():
                results['endpoint']=read(ed/'RESULTS.json');require(results['endpoint']['identity']==endpoint_identity,'cached endpoint result identity')
            else:
                try:results['endpoint']=endpoint(ed,p,c,a,pol,st,endpoint_identity)
                except Exception as exc:
                    results['endpoint']=dict(status='computational_unavailable',cause=type(exc).__name__+': '+str(exc),map=st['total'],no_retry=True)
                results['endpoint']['identity']=endpoint_identity;save_json(ed/'RESULTS.json',results['endpoint'])
        for b in BRANCHES[1:]:
          branch=summaries.get((a,pol,b));rows=[];bd=d/b;bd.mkdir(exist_ok=True)
          if not branch:continue
          for s in branch['histories']:
            k=s['cycle_index'];rs=refs.get(k)
            if not s.get('total') or not rs or not rs.get('total'):continue
            im,ih=load(verified(s['total']));refim,rh=load(verified(rs['total']));v,r,h,place=aligned(im['PRIMARY'],ih,refim['PRIMARY'],rh)
            if b.startswith('C-'):
                old,oh=load(Path(p['historical_root'])/'report'/a/b/f'cycle-{k:02d}.fits');previous_header=h
                v,old_aligned,h,place=aligned(v,h,old['COMMON'],oh);r=place(r,previous_header)
            bv=place(im['SCI_SUPPORT'],ih)==1;rv=place(refim['SCI_SUPPORT'],rh)==1;common=bv&rv;delta=np.where(common,v-r,np.nan)
            ds,dv,_,_,_=source_patch(delta,common,np.ones(delta.shape),h)
            row=dict(cycle=k,measurement_increment=increments(s['measurement'],rs['measurement']) if s.get('measurement') else None,
                support=dict(common=int(common.sum()),lost=int((rv&~bv).sum()),gained=int((bv&~rv).sum()),normalization_changed=int((common&(place(im['NORMALIZATION'],ih)!=place(refim['NORMALIZATION'],rh))).sum())),
                difference_rms=float(np.sqrt(np.mean(delta[common]**2))),difference_max_abs=float(np.max(abs(delta[common]))),
                shoulder_projection=shape_projection(ds,dv,sh['INJECTION']),profile=profile(ds,dv,c))
            if s.get('candidate') and rs.get('candidate'):
                cb,_=load(verified(s['candidate']));cr,_=load(verified(rs['candidate']));valid=cb['VALID'].astype(bool)&cr['VALID'].astype(bool);cd=np.where(valid,cb['CANDIDATE']-cr['CANDIDATE'],np.nan)
                row['candidate_difference']=selection_difference(cb,cr);row['candidate_shoulder_projection']=shape_projection(cd,valid,sh['INJECTION']);row['candidate_difference_norm']=float(np.linalg.norm(cd[valid]))
            planes=[('DELTA',delta,'mJy/nominal beam'),('COMMON',common.astype('u1'),'1')]
            if b.startswith('C-'):
                raster,rash=load(verified(p['original_owner']['injections'][a][b]));known,kv,kh,audit=injection_reference(verified(rs['total']).parent,verified(s['total']).parent,raster['INJECTION'],rash)
                ki=place(known,kh);native=common&(place(kv,kh)==1)
                # Original physical domain is retained, including zero-injection pixels.
                oldmask=place(old['COMMON'],oh)==1;benchmark=native&oldmask
                row.update(projection_audit=audit,native_response=metrics(delta,ki,native),benchmark_response=metrics(delta,ki,benchmark),original_benchmark_pixels=int(oldmask.sum()),benchmark_supported_pixels=int(benchmark.sum()),benchmark_complete=bool(np.all(native[oldmask])),benchmark_domain='original archived physical COMMON intersect native paired support; missing original pixels explicitly counted')
                ks,ksv,_,_,_=source_patch(ki,native,np.ones(ki.shape),h);row['known_profile']=profile(ks,ksv,c);row['residual_profile']=profile(ds-ks,dv&ksv,c)
                row['signed_fixed_template']=float(np.sum(c['q'][c['q']!=0]*ds[c['q']!=0])) if np.all(dv[c['q']!=0]) else None
                row['signed_apertures']=[float(np.sum(ds[m])*4) if dv[m].all() else None for m in [c['radius']<=f*c['major'] for f in (1.,1.5,2.)]]
                planes.extend([('KNOWN',ki,'mJy/nominal beam'),('RESIDUAL',np.where(native,delta-ki,np.nan),'mJy/nominal beam'),('BENCHMARK',benchmark.astype('u1'),'1')])
                if k==10:
                    rpatch,rvalid,_,_=patch_map(verified(rs['total']));ideal_known,ideal_support,_,_,_=source_patch(known,kv,np.ones(known.shape),kh);cache=bd/'ideal-measurement.json'
                    identity=dict(reference=rs['total'],injection=p['original_owner']['injections'][a][b],source_method=p['source_code']['point_expected.py'])
                    if cache.exists():
                        cached=read(cache);require(cached['identity']==identity,'ideal cache identity');ideal=cached['measurement']
                    else:
                        cached=dict(identity=identity,measurement=None,failure='incomplete attempt; not repeated');save_json(cache,cached)
                        try:
                            ideal,prepared=infer(c,rpatch+ideal_known,rvalid&ideal_support);cached.update(measurement=ideal,failure=None)
                        except Exception as exc:ideal=None;cached['failure']=type(exc).__name__+': '+str(exc)
                        save_json(cache,cached)
                    ideal_count+=1
                    if ideal is not None:row['ideal_measurement']=ideal_comparison(c,rs['measurement'],s['measurement'],ideal)
                    else:row['ideal_control_unavailable']=cached['failure']
                    write_fits(bd/'ideal-T10.fits',c['header'],[('SIGNAL',rpatch+ideal_known,'mJy/nominal beam'),('VALID',(rvalid&ideal_support).astype('u1'),'1')],companion='POINT-GUARDED-UNAPPLIED-IDEAL')
            write_fits(bd/f'cycle-{k:02d}.fits',h,planes,companion='POINT-GUARDED-RESPONSE');save_json(bd/f'cycle-{k:02d}.json',row);rows.append(row)
          results['branches'][b]=rows
        plus={s['cycle_index']:s for s in summaries.get((a,pol,'C-core-plus'),{}).get('histories',[])};minus={s['cycle_index']:s for s in summaries.get((a,pol,'C-core-minus'),{}).get('histories',[])}
        for k in sorted(set(plus)&set(minus)&set(refs)):
            if not all(z.get('total') for z in (plus[k],minus[k],refs[k])):continue
            pp,pv,_,h=patch_map(verified(plus[k]['total']));mp,mv,_,_=patch_map(verified(minus[k]['total']));rp,rv,_,_=patch_map(verified(refs[k]['total']));valid=pv&mv&rv;even=np.where(valid,pp+mp-2*rp,np.nan)
            results['even'].append(dict(cycle=k,rms=float(np.sqrt(np.mean(even[valid]**2))),max_abs=float(np.max(abs(even[valid]))),profile=profile(even,valid,c)))
            write_fits(d/f'even-{k:02d}.fits',h,[('EVEN',even,'mJy/nominal beam'),('VALID',valid.astype('u1'),'1')],companion='POINT-GUARDED-EVEN')
        array['policies'][pol]=results;save_json(d/'RESULTS.json',results)
      # Historical BASE comparisons never substitute for new matched controls.
      oldresults=read(Path(p['historical_root'])/'report'/a/'results.json');array['historical']['response_and_seed']=oldresults
      hist=[]
      for b in BRANCHES:
        old=read(verified(p['originals'][a][b]));new=summaries.get((a,'BASE-POINT',b))
        if not new:continue
        oldst={s['cycle_index']:s for s in old['histories']}
        for s in new['histories']:
            k=s['cycle_index'];o=oldst.get(k)
            if not o or not s.get('measurement'):continue
            row=dict(branch=b,cycle=k,measurement_delta=increments(s['measurement'],o['measurement']),new_guard=s['measurement']['guard'],historical_fit=o['measurement']['fit'])
            if s.get('candidate') and o.get('candidate'):
                n,_=load(verified(s['candidate']));v,_=load(verified(o['candidate']));row['candidate_delta']=selection_difference(n,v)
            if s.get('total') and o.get('total'):
                n,nv,_,_=patch_map(verified(s['total']));v,vv,_,_=patch_map(verified(o['total']));use=nv&vv;row['source_stamp_map_max_abs']=float(np.max(abs(n-v)[use]));row['source_stamp_map_rms']=float(np.sqrt(np.mean((n-v)[use]**2)))
            hist.append(row)
      array['historical']['BASE_POINT_minus_BASE']=hist
      arrays[a]=array;save_json(ad/'RESULTS.json',array);draw(ad,a,array,c);print(json.dumps(dict(array=a,report='complete',ideal_controls=ideal_count,endpoint_analyses=endpoint_count)),flush=True)
    require(ideal_count<=27 and endpoint_count<=9,'diagnostic bound')
    summary=dict(source_revision=revision,prospective=bound(root/'freeze/prospective.json'),matrix=matrix,arrays=arrays,native_runtime=runtime,
        attempted_feedback_invocations=len(attempted),completed_native_receipts=len(receipts),logical_trajectories=45,new_M0_calculations=0,runtime_scope='Phase wall times include concurrently executed trajectories and final authentication, not summed native wall times. Native CPU is process CPU; sampled aggregate RSS includes coordinator and children every 0.5s. Inherited per-RUN no concurrent diagnostic wording applies to each invocation; other branches can perform diagnostics concurrently.',ideal_controls=ideal_count,endpoint_analyses=endpoint_count,
        phases={phase:read(root/f'FEEDBACK-{phase}.json') for phase in ('profile','REF','compact','shoulder')},report_wall_seconds=time.perf_counter()-start,
        units=dict(zip(MEASUREMENT_NAMES,MEASUREMENT_UNITS)))
    save_json(out/'SUMMARY.json',summary);tables(out,summary);authenticated(root);manifest(out)

def tables(out,s):
    lines=['# Guarded feedback numerical comparison','',
      'Brightness is mJy/nominal beam; offsets and widths arcsec; apertures brightness × arcsec². All full-map fits use POINT-EXPECTED-v1 and one shared fitted constant. sigma_A is conditional on frozen full C0, not a new endpoint uncertainty. Signed maps are never source-fitted.','',
      '| Array | Policy | Branch | Status | Stage | Reason |','|---|---|---|---|---:|---|']
    for r in s['matrix']:lines.append(f"| {r['array']} | {r['policy']} | {r['branch']} | {r['status']} | {r.get('cycle','—')} | {r['reason'] or ''} |")
    for a,data in s['arrays'].items():
      lines.extend(['',f'## {a}','',f'![Historical comparison]({out/a}/historical.png)',f'![Histories]({out/a}/histories.png)',f'![Response]({out/a}/response.png)',f'![Seed and data]({out/a}/seed.png)',f'![Signed maps]({out/a}/maps.png)',f'![Residuals]({out/a}/residuals.png)',f'![Profiles]({out/a}/profiles.png)',f'![Even response]({out/a}/even.png)',f'![Endpoint]({out/a}/endpoint.png)','',
        '| Policy | Injection | Stage | g | eta | rho | Original pixels / usable |','|---|---|---:|---:|---:|---:|---|'])
      for pol,r in data['policies'].items():
        for b,rows in r['branches'].items():
          for q in rows:
            if b.startswith('C-') and q['cycle'] in SNAPS:
              m=q['benchmark_response'] or {};lines.append(f"| {pol} | {b} | {q['cycle']} | {fmt(m.get('g'))} | {fmt(m.get('eta'))} | {fmt(m.get('rho'))} | {q['original_benchmark_pixels']} / {q['benchmark_supported_pixels']} |")
      lines.extend(['','| Policy | Branch | Stage | Peak | FWHM major/minor | Background | Template | Pixels / extra |','|---|---|---:|---:|---|---:|---:|---|'])
      for pol,r in data['policies'].items():
        for b,rows in r['history'].items():
          for q in rows:
            if q['cycle'] not in SNAPS:continue
            m=q.get('measurements') or {};ad=q.get('admission') or {};lines.append(f"| {pol} | {b} | {q['cycle']} | {fmt(m.get('peak'))} | {fmt(m.get('FWHM_major'))} / {fmt(m.get('FWHM_minor'))} | {fmt(m.get('background'))} | {fmt(m.get('fixed_template'))} | {q.get('actual_candidate',{}).get('selected_pixels')} / {ad.get('added_pixels')} |")
      detail=['# T10 ideal-addition controls and seed diagnostics','', 'Actual and REF fits are reused; only ideal full-map addition is fitted once. Ratios are unavailable for ideal increments ≤1e−8 in absolute native units. Raw aperture and background contributions are saved separately in cycle JSON.','']
      for pol,r in data['policies'].items():
        for b,rows in r['branches'].items():
          for q in rows:
            if q['cycle']!=10:continue
            detail.extend([f'## {pol} {b}',''])
            if 'ideal_measurement' in q:
                ic=q['ideal_measurement'];detail.extend([f"Ideal guard: {ic['ideal_map_addition']['guard']['status']}; constrained: {ic['ideal_map_addition']['fit']['constrained']}.",'','| Measurement | Actual increment | Ideal increment | Difference | Ratio |','|---|---:|---:|---:|---:|'])
                for z in ic['comparisons'] or []:detail.append('| '+z['name']+' | '+' | '.join(fmt(z[k]) for k in ('actual_increment','ideal_increment','absolute_discrepancy','ratio'))+' |')
            elif 'ideal_control_unavailable' in q:detail.append('Ideal control unavailable: '+q['ideal_control_unavailable'])
            else:detail.append(f"Total shoulder projection {fmt(q['shoulder_projection']['value'])}; candidate shoulder projection {fmt(q.get('candidate_shoulder_projection',{}).get('value'))}; full-map difference RMS {fmt(q['difference_rms'])}.")
            detail.append('')
      (out/a/'MEASUREMENTS.md').write_text('\n'.join(detail)+'\n')
      guard_lines=['# Guard and terminal drift','', 'Frozen full-C0 sigma_A is the conditional guard reference scale only. Active bounds and all diagnostics are retained in RESULTS.json. Seed m0 counts below come from the actual applied seed; ordinary admission remains separate.','', '| Policy | Branch | Stage | Accepted | sigma_A | A/sigma_A | Centroid radius | Active bounds | Pixels | Extra normal-admission |','|---|---|---:|---|---:|---:|---:|---|---:|---:|']
      drift_lines=['# T10 minus T9 and raw aperture accounting','', '| Policy | Branch | Peak drift | Background drift | Template drift | Aperture 2 drift | T10 raw aperture 2 | T10 background contribution |','|---|---|---:|---:|---:|---:|---:|---:|']
      for pol,r in data['policies'].items():
        for b,rows in r['history'].items():
          for q in rows:
            if q['cycle'] in SNAPS and q.get('measurement'):
                g=q['measurement']['guard'];gd=g.get('diagnostic') or {}
                guard_lines.append(f"| {pol} | {b} | {q['cycle']} | {g['accepted']} | {fmt(g.get('sigma_A'))} | {fmt(g.get('SNR_A'))} | {fmt(gd.get('centroid_radius_arcsec'))} | {gd.get('active_mask')} | {q.get('actual_candidate',{}).get('selected_pixels')} | {q.get('admission',{}).get('added_pixels')} |")
            if q['cycle']==10 and q.get('measurement'):
                dr=q.get('previous_cycle_drift') or {};m=q['measurement']
                drift_lines.append(f"| {pol} | {b} | "+' | '.join(fmt(dr.get(k)) for k in ('peak','background','fixed_template','aperture_2'))+f" | {fmt(m['raw_apertures'][2]['value_mJy_per_beam_arcsec2'])} | {fmt(m['aperture_background_terms'][2])} |")
      (out/a/'GUARDS.md').write_text('\n'.join(guard_lines)+'\n');(out/a/'DRIFT.md').write_text('\n'.join(drift_lines)+'\n')

    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    noise=['# Endpoint covariance applicability','', 'One existing-family fit per completed REF endpoint; original masks and coverage domains. No simulation, new source uncertainty, score change or feedback. Matched variance uses identical neighborhood weights and exclusions; ratios are model comparisons, not source-location truth.','', '| Array | Policy | Background std | Robust scale | Median background scale / C0 | CORR variance median / C0 | DIAG variance median / C0 | Source coverage-valid pixels | Source pixels |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    same=['# Same-map T10 candidate cross-check','', 'All three policy selections share the recorded guard fit on each endpoint. These candidates remain unapplied.','', '| Array | Producing policy | Selecting policy | Baseline pixels | Extra pixels | Proposed pixels | Extra brightness area |','|---|---|---|---:|---:|---:|---:|']
    for a,data in s['arrays'].items():
      for pol,r in data['policies'].items():
        for sel,q in r['cross_candidates'].items():same.append(f"| {a} | {pol} | {sel} | {q['baseline_pixels']} | {q['added_pixels']} | {q['proposed_pixels']} | {fmt(q['extra_brightness_area'])} |")
        e=r.get('endpoint',{})
        if e.get('status')!='available':noise.append(f"| {a} | {pol} | unavailable: {e.get('cause')} | | | | | | |");continue
        comp=e['matched_neighborhood_variance'];app=e['applicability'];rat=e['background_scale_ratio_to_C0_quantiles'];med=lambda policy:comp[policy]['ratio_quantiles'][2] if comp[policy]['ratio_quantiles'] is not None else None
        noise.append(f"| {a} | {pol} | {fmt(e['background']['std'])} | {fmt(e['background']['robust_scale'])} | {fmt(rat[2])} | {fmt(med('CORR'))} | {fmt(med('DIAG'))} | {app['source_depth_valid_pixels']} | {app['source_fit_pixels']} |")
    (out/'ENDPOINTS.md').write_text('\n'.join(noise)+'\n');(out/'CROSS_CANDIDATES.md').write_text('\n'.join(same)+'\n')


def draw(out,a,data,c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'BASE-POINT':'#333333','CORR':'#1f77b4','DIAG':'#e67e22'}
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for b in BRANCHES:
        rows=[q for q in data['historical']['BASE_POINT_minus_BASE'] if q['branch']==b]
        for z,key in zip(ax[0],('peak','background')):z.plot([q['cycle'] for q in rows],[(q['measurement_delta'] or {}).get(key,np.nan) for q in rows],'.-',label=b);z.set(xlabel='cycle',ylabel=key+' BASE-POINT − historical BASE [mJy/beam]');z.grid(alpha=.2)
    for b in BRANCHES[1:4]:
        rows=data['policies']['BASE-POINT']['branches'].get(b,[]);old=data['historical']['response_and_seed']['branches'].get(b,[])
        oldby={q['cycle']:q for q in old}
        ax[1,0].plot([q['cycle'] for q in rows],[(q['benchmark_response'] or {}).get('g',np.nan)-oldby[q['cycle']]['map_response']['g'] for q in rows],'.-',label=b)
    rows=data['policies']['BASE-POINT']['branches'].get('D-seed-shoulder',[]);old=data['historical']['response_and_seed']['branches'].get('D-seed-shoulder',[])
    ax[1,1].plot([q['cycle'] for q in rows],[q['shoulder_projection']['value'] for q in rows],'.-',label='BASE-POINT');ax[1,1].plot([q['cycle'] for q in old],[q['total_shoulder_projection'] for q in old],'--',label='historical BASE')
    ax[1,0].set(xlabel='cycle',ylabel='response g change');ax[1,1].set(xlabel='cycle',ylabel='seed-only shoulder projection')
    for z in ax.flat:z.legend(fontsize=7);z.grid(alpha=.2)
    fig.suptitle(a+' guard/fit change separated from supplemental admission');fig.savefig(out/'historical.png',dpi=140);plt.close(fig)

    fig,ax=plt.subplots(3,3,figsize=(14,11),constrained_layout=True)
    keys=['peak','east','north','FWHM_major','FWHM_minor','background','fixed_template','aperture_2','pixels']
    for pol,r in data['policies'].items():
        for b,rows in r['history'].items():
            if b!='REF':continue
            for axes,key in zip(ax.flat,keys):axes.plot([q['cycle'] for q in rows],[q['actual_candidate']['selected_pixels'] if key=='pixels' else (q.get('measurements') or {}).get(key,np.nan) for q in rows],'.-',color=colors[pol],label=pol);axes.set(title=key,xlabel='cycle');axes.grid(alpha=.2)
    ax[0,0].legend();fig.suptitle(a+' REF: shared guard, evolving maps');fig.savefig(out/'histories.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for j,b in enumerate(BRANCHES[1:4]):
      for pol,r in data['policies'].items():
        rows=r['branches'].get(b,[])
        for i,key in enumerate(('g','eta','rho')):ax[i,j].plot([q['cycle'] for q in rows],[(q['benchmark_response'] or {}).get(key,np.nan) for q in rows],'.-',label=pol,color=colors[pol]);ax[i,j].set(title=b if i==0 else '',ylabel=key,xlabel='cycle');ax[i,j].grid(alpha=.2)
    ax[0,0].legend();fig.suptitle(a+' finite response on original physical benchmark');fig.savefig(out/'response.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for j,(pol,r) in enumerate(data['policies'].items()):
      for b,style in [('C-shoulder','-'),('D-seed-shoulder','--')]:
        rows=r['branches'].get(b,[]);ks=[q['cycle'] for q in rows]
        ax[0,j].plot(ks,[q['shoulder_projection']['value'] for q in rows],style,marker='.',label=b)
        if b.startswith('D-'):ax[0,j].plot(ks,[q.get('candidate_shoulder_projection',{}).get('value',np.nan) for q in rows],':',label='seed candidate')
        ax[1,j].plot(ks,[(q.get('measurement_increment') or {}).get('background',np.nan) for q in rows],style,marker='.',label=b)
      ax[0,j].set(title=pol,ylabel='projection onto archived shoulder');ax[1,j].set(ylabel='fitted constant − own REF [mJy/beam]',xlabel='cycle');ax[0,j].legend(fontsize=8)
    for axes in ax.flat:axes.grid(alpha=.2)
    # Common y scales make policy differences directly visible.
    for row in ax:
        lo=min(z.get_ylim()[0] for z in row);hi=max(z.get_ylim()[1] for z in row)
        for z in row:z.set_ylim(lo,hi)
    fig.suptitle(a+' data shoulder versus first-candidate-only seed');fig.savefig(out/'seed.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(4,3,figsize=(12,14),constrained_layout=True)
    for i,b in enumerate(BRANCHES[1:]):
      ims=[]
      for pol in POLICIES:
        f=out/pol/b/'cycle-10.fits'
        if f.exists():im,h=load(f);v,valid,_,_,_=source_patch(im['DELTA'],im['COMMON'].astype(bool),np.ones(im['DELTA'].shape),h);ims.append(v)
        else:ims.append(None)
      vmax=max([float(np.nanmax(abs(v))) for v in ims if v is not None]+[1e-8])
      for j,(pol,v) in enumerate(zip(POLICIES,ims)):
        axes=ax[i,j];axes.set(title=pol+' '+b,xlabel='east [arcsec]',ylabel='north [arcsec]')
        if v is None:axes.text(.5,.5,'T10 unavailable',ha='center',transform=axes.transAxes);continue
        im=axes.imshow(v,origin='lower',extent=[41,-41,-41,41],cmap='RdBu_r',vmin=-vmax,vmax=vmax);fig.colorbar(im,ax=axes,shrink=.7)
    fig.suptitle(a+' T10 − own REF; matched scales by row [mJy/nominal beam]');fig.savefig(out/'maps.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(3,4,figsize=(14,11),constrained_layout=True)
    for i,b in enumerate(BRANCHES[1:4]):
        images=[]
        for pol in POLICIES:
            f=out/pol/b/'cycle-10.fits'
            if f.exists():
                im,h=load(f);v,sv,_,_,_=source_patch(im['RESIDUAL'],im['COMMON'].astype(bool),np.ones(im['DELTA'].shape),h);images.append(v)
                known,_,_,_,_=source_patch(im['KNOWN'],im['COMMON'].astype(bool),np.ones(im['DELTA'].shape),h)
            else:images.append(None)
        images=[known if any(v is not None for v in images) else None,*images]
        vmax=max([float(np.nanmax(abs(v))) for v in images if v is not None]+[1e-8])
        for j,(label,v) in enumerate(zip(('known injection',*POLICIES),images)):
            z=ax[i,j];z.set(title=b+' '+label,xlabel='east [arcsec]',ylabel='north [arcsec]')
            if v is None:z.text(.5,.5,'unavailable',transform=z.transAxes,ha='center');continue
            im=z.imshow(v,origin='lower',extent=[41,-41,-41,41],cmap='RdBu_r',vmin=-vmax,vmax=vmax);fig.colorbar(im,ax=z,shrink=.7)
    fig.suptitle(a+' known injection and T10 response residual; matched scales per row [mJy/beam]');fig.savefig(out/'residuals.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    radii=(c['fixed'].edges[:-1]+c['fixed'].edges[1:])/2
    for j,b in enumerate(BRANCHES[1:4]):
        for pol,r in data['policies'].items():
            last=next((q for q in r['branches'].get(b,[]) if q['cycle']==10),None)
            if last is None:continue
            ax[0,j].plot(radii,last['profile'],label=pol,color=colors[pol]);ax[1,j].plot(radii,last['residual_profile'],label=pol,color=colors[pol])
            if pol=='BASE-POINT':ax[0,j].plot(radii,last['known_profile'],'k:',label='known')
        ax[0,j].set(title=b,ylabel='signed response [mJy/beam]');ax[1,j].set(xlabel='radius from original centroid [arcsec]',ylabel='response − known [mJy/beam]')
    for z in ax.flat:z.axhline(0,color='gray',lw=.7);z.grid(alpha=.2);z.legend(fontsize=8)
    fig.suptitle(a+' source, wings and pedestal: full signed profiles');fig.savefig(out/'profiles.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,4,figsize=(16,4.5),constrained_layout=True);images=[]
    for pol in POLICIES:
        f=out/pol/'even-10.fits';images.append(load(f)[0]['EVEN'] if f.exists() else None)
    vmax=max([float(np.nanmax(abs(v))) for v in images if v is not None]+[1e-8])
    for j,(pol,v) in enumerate(zip(POLICIES,images)):
        ax[j].set(title=pol,xlabel='east [arcsec]',ylabel='north [arcsec]')
        if v is not None:
            im=ax[j].imshow(v,origin='lower',extent=[41,-41,-41,41],cmap='RdBu_r',vmin=-vmax,vmax=vmax);fig.colorbar(im,ax=ax[j],shrink=.7)
        rows=data['policies'][pol]['even'];ax[3].plot([q['cycle'] for q in rows],[q['rms'] for q in rows],'.-',label=pol,color=colors[pol])
    ax[3].set(xlabel='cycle',ylabel='even RMS [mJy/beam]');ax[3].legend(fontsize=8);ax[3].grid(alpha=.2)
    fig.suptitle(a+' sign asymmetry: plus + minus − 2 REF, common 41×41 support');fig.savefig(out/'even.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for pol,r in data['policies'].items():
        e=r.get('endpoint',{})
        if e.get('status')!='available':continue
        bins=e['summary']['bins'];ax[0].plot([(q['coverage_min']+q['coverage_max_exclusive'])/2 for q in bins],[q['std'] for q in bins],'.-',label=pol,color=colors[pol])
        for j,direction in enumerate(('east','north'),1):
            q=e['directional'];ax[j].plot(q['lag_arcsec'],q['endpoint_'+direction],'.-',label=pol,color=colors[pol]);ax[j].set(title=direction,xlabel='lag [arcsec]',ylabel='normalized correlation')
    if any(r.get('endpoint',{}).get('status')=='available' for r in data['policies'].values()):
        e=next(r['endpoint'] for r in data['policies'].values() if r.get('endpoint',{}).get('status')=='available')
        for j,direction in enumerate(('east','north'),1):ax[j].plot(e['directional']['lag_arcsec'],e['directional']['frozen_'+direction],'k:',label='frozen C0')
    ax[0].set(xlabel='coverage',ylabel='background scatter [mJy/nominal beam]')
    for z in ax:z.legend(fontsize=8);z.grid(alpha=.2)
    fig.suptitle(a+' diagnostic endpoint covariance; never fed back');fig.savefig(out/'endpoint.png',dpi=140);plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);assess(ap.parse_args().output.resolve())
