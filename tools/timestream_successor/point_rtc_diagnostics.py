"""Saved native-array population audit; never runs RTC/PTC or changes policy."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import yaml
from point_additive_diagnostics import save_json
from inspect_forward_map import require

def read(p):return json.loads(Path(p).read_text())
def yread(p):return yaml.load(Path(p).read_text(),Loader=yaml.CSafeLoader)
def arr(p,dtype,shape=None):
    a=np.fromfile(p,dtype);return a.reshape(shape) if shape is not None else a

def spectrum(x,dt):
    h=np.hanning(len(x));f=np.fft.rfft((x-np.median(x,axis=0))*h[:,None] if x.ndim==2 else (x-np.median(x))*h,axis=0)
    p=dt*abs(f)**2/np.sum(h*h);p[1:-1]*=2
    return p,f

def distribution(x):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None,maximum=float(a.max()) if len(a) else None)

def bands(p,freq):
    df=freq[1]-freq[0];line=(freq>=9.75)&(freq<=12.25);side=((freq>=7)&(freq<9.5))|((freq>12.5)&(freq<=15))
    power=p[line].sum(axis=0)*df;continuum=np.median(p[side],axis=0)
    return dict(line_power=power,line_excess=np.maximum(0,power-continuum*line.sum()*df),line_prominence=p[line].max(axis=0)/np.maximum(continuum,1e-300),low_power=p[(freq>=.5)&(freq<=3)].sum(axis=0)*df,high_power=p[(freq>=20)&(freq<=28)].sum(axis=0)*df,above_output_nyquist_power=p[(freq>=35)&(freq<=60)].sum(axis=0)*df if freq[-1]>35 else np.full(p.shape[1:] or (),np.nan))

def coherence(z):
    # Pairwise magnitude-squared coherence at the same fixed 11-Hz-nearest bin,
    # averaged across genuinely common complete windows; all detector pairs.
    valid=np.isfinite(z);v=valid.astype(float);a=np.where(valid,z,0);power=abs(a)**2
    count=np.einsum('wi,wj->ij',v,v,optimize=False);cross=np.einsum('wi,wj->ij',a.conj(),a,optimize=False);left=np.einsum('wi,wj->ij',power,v,optimize=False);right=left.T
    use=np.triu((count>=3)&(left>0)&(right>0),1)
    c=abs(cross[use])**2/(left[use]*right[use])
    return dict(**distribution(c),minimum_common_windows=3,metric='pairwise magnitude-squared coherence at nearest11Hz; same window pairs; descriptive, no electronics attribution',common_windows=distribution(count[use]))

def load_network(root,family,nw):
    ref=root/f'{family}-REF-rep1';b=ref/f'upstream/products/network{nw}';probe=root/f'{family}-RTC-diagnostic/native/products/network{nw}/source-probe'
    r=yread(b/'receipt.yaml');pr=yread(probe/'receipt.yaml');n,d=pr['native_rows'],pr['detectors'];a=dict(base=b,probe=probe,receipt=r,probe_receipt=pr,shape=(n,d))
    for key,file in [('raw','original-x.f64'),('raw_r','original-r.f64'),('lon','longitude-deg.f64'),('lat','latitude-deg.f64'),('mult','cal-multiplier.f64')]:a[key]=arr(probe/file,'<f8',(n,d))
    a['valid']=arr(probe/'input-valid.u8','u1',(n,d)).astype(bool);a['mapping']=arr(probe/'mapping-available.u8','u1',(n,d)).astype(bool)
    a['times']=arr(probe/'assigned-time.f64','<f8');a['dt']=float(r['cadence_interval_seconds']);a['channels']=pr['channels']
    for key in ('filtered','filtered_r','state','center','speed','causes'):
        a[key]=np.empty((n,d),dtype=float if 'filtered' in key else np.uint8)
    for j,c in enumerate(a['channels']):
        q=b/'donor-continuity';pair=arr(q/f'{c}-filtered.f64','<f8',(n,2));a['filtered'][:,j]=pair[:,0];a['filtered_r'][:,j]=pair[:,1]
        for key,file in [('state','state'),('center','map-center'),('speed','speed-evidence'),('causes','causes')]:a[key][:,j]=arr(q/f'{c}-{file}.u8','u1')
    return a

def population(root,family,dest,*,match_family=None,background_family=None):
    started=time.monotonic();dest.mkdir(parents=True,exist_ok=True);ref=root/f'{family}-REF-rep1'
    summary=read(root/f'{background_family or family}-REF-rep1/run/BASE-POINT/REF/a2000/SUMMARY.json');params=np.asarray(summary['histories'][0]['measurement']['guard']['accepted_parameters']);radius=4*max(params[3:5])*np.sqrt(8*np.log(2))
    output=[];allbands=[];psds={};rowids=[];freqs={};coherences=[];eventrows=[]
    for nw in (11,12):
        a=load_network(root,family,nw);n,d=a['shape'];dt=a['dt'];nfft=2*int(round(4/(2*dt)));freq=np.fft.rfftfreq(nfft,dt);freqs['native']=freq;freqs['output']=np.fft.rfftfreq(nfft//2,2*dt)
        bg=np.hypot(a['lon']*3600-params[1],a['lat']*3600-params[2])>radius
        filtered_good=(a['state']&1)!=0;paired=bg&a['valid']&a['mapping']&filtered_good
        other=load_network(root,match_family,nw) if match_family else None
        if other is not None:
            require(a['channels']==other['channels'] and np.array_equal(a['times'],other['times']),'cross-arm native identity/time differs')
            paired &= other['valid']&other['mapping']&((other['state']&1)!=0)
        raw=a['raw']*a['mult'];filtered=a['filtered']*a['mult'];windows=[];totals=np.zeros((2,len(freq),d));counts=np.zeros(d,int);z=[[],[]]
        for lo in range(0,n-nfft+1,nfft//2):
            hi=lo+nfft;good=paired[lo:hi].all(axis=0)
            if np.max(np.diff(a['times'][lo:hi]))>dt*1.001:continue
            windows.append((lo,hi,int(good.sum())))
            for stage,x in enumerate((raw,filtered)):
                zz=np.full(d,np.nan+1j*np.nan);ids=np.flatnonzero(good)
                if len(ids):
                    p,f=spectrum(x[lo:hi,ids],dt);totals[stage][:,ids]+=p;zz[ids]=f[np.argmin(abs(freq-11))]
                    for k,v in bands(p,freq).items():
                        allbands.extend(dict(network=nw,domain='native-matched',stage=('raw','RTC')[stage],segment=-1,start=lo,detector=int(j),metric=k,value=float(t) if np.isfinite(t) else None) for j,t in zip(ids,v))
                z[stage].append(zz)
            counts+=good
        for i,stage in enumerate(('raw','RTC')):
            average=np.divide(totals[i],counts[None,:],out=np.full_like(totals[i],np.nan),where=counts[None,:]>0);psds[f'n{nw}_{stage}']=average
            coherences.append(dict(network=nw,stage=stage,**coherence(np.asarray(z[i]))))
        events=yread(a['base']/'events.yaml');decisions=yread(a['base']/'event-decisions.yaml');nativefirst=a['receipt']['native_runs'][0][0] if isinstance(a['receipt']['native_runs'][0],list) else 0
        for e in events:
            row=e['seed_earlier_row']-nativefirst;j=e['detector']
            eventrows.append(dict(network=nw,event=e['event'],detector=j,native_row=e['seed_earlier_row'],near_source=bool(not bg[row,j]) if 0<=row<n else None,jump_admitted=e['jump_admitted'],hard_spike_accepted=e['hard_spike_accepted']))
        support=dict(network=nw,detectors=d,native_rows=n,native_cells=n*d,input_valid=int(a['valid'].sum()),mapping_available=int(a['mapping'].sum()),RTC_x_available=int(filtered_good.sum()),map_center_admitted=int(a['center'].sum()),scheduled_slots_per_detector=(n+1)//2,scheduled_x_available=int(filtered_good[::2].sum()),scheduled_map_centers=int(a['center'][::2].sum()),replacement_centers=int(((a['state']&4)!=0).sum()),replacement_influence=int(((a['state']&16)!=0).sum()),unrepaired_influence=int(((a['state']&32)!=0).sum()),speed_restricted=int((a['speed']!=0).sum()),transient_excluded_pair_cells=a['receipt']['transient_excluded_pair_cells'],event_count=len(events),candidate_count=a['receipt']['candidate_count'],admitted_jump_groups=a['receipt']['admitted_jump_groups'],event_classes=dict(Counter(e['class'] for e in decisions)),complete_spectral_windows_per_detector=counts.tolist(),native_windows=windows,source_exclusion_radius_arcsec=float(radius))
        # Independent descriptive adjacent-difference outliers, measured-data only;
        # the same raw difference scale for raw and filtered, not flag validation.
        adjacent=paired[1:]&paired[:-1];delta=np.diff(raw,axis=0);outliers=[]
        edge=np.zeros((n,d),bool);bad=~filtered_good
        for j in range(d):
            # A 0.25s retained neighborhood outside unavailable regions.
            from scipy.ndimage import maximum_filter1d
            edge[:,j]=maximum_filter1d(bad[:,j],size=2*int(round(.25/dt))+1)&filtered_good[:,j]
            use=adjacent[:,j];v=delta[use,j];scale=1.4826*np.median(abs(v-np.median(v))) if len(v) else np.nan
            row=dict(detector=j,adjacent_pairs=int(use.sum()),raw_difference_MAD=float(scale))
            for name,x in [('raw',raw),('RTC',filtered)]:
                dx=np.diff(x[:,j]);row[name+'_gt8MAD']=int(np.sum(abs(dx[use]-np.median(v))>8*scale)) if scale>0 else None
                row[name+'_near_boundary_abs_difference']=distribution(abs(dx[use&edge[1:,j]]));row[name+'_away_abs_difference']=distribution(abs(dx[use&~edge[1:,j]]))
            outliers.append(row)
        support['adjacent_difference_diagnostics']=outliers
        domain=a['valid']&a['mapping'];inside=domain&~bg;outside=domain&bg
        support['source_crossing_support']={name:dict(scheduled_original_cells=int(mask[::2].sum()),RTC_available=int((mask&filtered_good)[::2].sum()),map_centers=int((mask&a['center'].astype(bool))[::2].sum())) for name,mask in [('inside_source_exclusion',inside),('outside_source_exclusion',outside)]}
        support['RTC_cause_cells']={str(int(k)):int(v) for k,v in zip(*np.unique(a['causes'],return_counts=True))}
        output_z={key:[] for key in ('CAL','PTC_M0','T5_residual','T5_total')}
        # Full population PTC stages, exact same source-free physical4s windows.
        ptc=a['base']/'donor-continuity/ptc';pr=yread(ptc/'receipt.yaml');last=ref/'run/BASE-POINT/REF/a2000/cycle-05/native/products';lr=yread(last/'receipt.yaml');segrows=[]
        for s in pr['segments']:
            sid=s['scan'];nt=s['scheduled_times'];nd=s['detectors'];no=s['output_detectors'];ids=np.asarray(s['detector_indices']);cols=np.asarray(s['fit_columns_in_output']);stem=f'segment-{sid}'
            slots=arr(ptc/f'{stem}-slots.i64','<i8');rows=slots*2
            cal=arr(ptc/f'{stem}-input.f64','<f8',(nt,nd));elig=arr(ptc/f'{stem}-eligible.u8','u1',(nt,nd)).astype(bool);clean=arr(ptc/f'{stem}-cleaned.f64','<f8',(nt,no))[:,cols]
            directory=last/f'network-{nw}-segment-{sid}';operand=arr(directory/'operand.f64','<f8',(nt,nd));lastclean=arr(directory/'cleaned.f64','<f8',(nt,no))[:,cols]
            calc=filtered[rows[:,None],ids];require(np.allclose(cal[elig],calc[elig],rtol=1e-12,atol=1e-9),'native RTC/CAL schedule mismatch')
            common=elig&bg[rows[:,None],ids]&np.isfinite(clean)&np.isfinite(lastclean)
            if other is not None:
                # Match genuine all-stage support by native detector identity,
                # including detectors absent from the other arm's fit.
                op=other['base']/'donor-continuity/ptc';orp=yread(op/'receipt.yaml')
                os=next(q for q in orp['segments'] if q['scan']==sid)
                oids=np.asarray(os['detector_indices']);oc=np.asarray(os['fit_columns_in_output']);ont=os['scheduled_times'];ond=os['detectors'];ono=os['output_detectors']
                require(np.array_equal(slots,arr(op/f'{stem}-slots.i64','<i8')),'cross-arm physical slots differ')
                oe=arr(op/f'{stem}-eligible.u8','u1',(ont,ond)).astype(bool)
                om=arr(op/f'{stem}-cleaned.f64','<f8',(ont,ono))[:,oc]
                ol=root/f'{match_family}-REF-rep1/run/BASE-POINT/REF/a2000/cycle-05/native/products'/f'network-{nw}-segment-{sid}'
                ot=arr(ol/'cleaned.f64','<f8',(ont,ono))[:,oc]
                full=np.zeros((nt,d),bool);full[:,oids]=oe&np.isfinite(om)&np.isfinite(ot)
                common &= full[:,ids]
            segcount=np.zeros(d,int);sums={key:np.zeros((len(freqs['output']),d)) for key in ('raw_native','RTC_native','CAL','PTC_M0','T5_input','T5_residual','T5_total')}
            for lo in range(0,nt-nfft//2+1,nfft//4):
                hi=lo+nfft//2;first=rows[lo];past=first+nfft
                if past>n or np.any(np.diff(rows[lo:hi])!=2) or np.max(np.diff(a['times'][first:past]))>dt*1.001:continue
                good=common[lo:hi].all(axis=0)&paired[first:past,ids].all(axis=0);jj=np.flatnonzero(good);globalids=ids[jj]
                if not len(jj):continue
                stages={'raw_native':raw[first:past,globalids],'RTC_native':filtered[first:past,globalids], 'CAL':cal[lo:hi,jj], 'PTC_M0':clean[lo:hi,jj], 'T5_input':(cal-operand)[lo:hi,jj], 'T5_residual':lastclean[lo:hi,jj], 'T5_total':(lastclean+operand)[lo:hi,jj]}
                for stage,x in stages.items():
                    pp,ft=spectrum(x,dt if stage.endswith('native') else 2*dt);pp=pp[:len(freqs['output'])];sums[stage][:,globalids]+=pp
                    if stage in output_z:
                        zz=np.full(d,np.nan+1j*np.nan);zz[globalids]=ft[np.argmin(abs(freqs['output']-11))];output_z[stage].append(zz)
                    for k,v in bands(pp,freqs['output']).items():allbands.extend(dict(network=nw,domain='all-stage-matched',stage=stage,segment=sid,start=int(first),detector=int(j),metric=k,value=float(t) if np.isfinite(t) else None) for j,t in zip(globalids,v))
                segcount[globalids]+=1
            for stage,values in sums.items():psds[f'n{nw}_s{sid}_{stage}']=np.divide(values,segcount[None,:],out=np.full_like(values,np.nan),where=segcount[None,:]>0)
            segrows.append(dict(segment=sid,detectors_in_fit=nd,output_detectors=no,scheduled_times=nt,eligible_entries=int(elig.sum()),retained_entries=s['retained'],windows_by_full_population=segcount.tolist(),calibration_roundtrip_max_abs=float(np.max(abs(cal[elig]-calc[elig])))))
        support['segments']=segrows
        for stage,zz in output_z.items():coherences.append(dict(network=nw,stage=stage,**coherence(np.asarray(zz))))
        coeff=np.asarray(a['receipt']['donor_continuity']['realized_detectors'][0]['lowpass_FIR']);ff=np.linspace(0,.5/dt,4097);response=abs(np.sum(np.exp(-2j*np.pi*ff[:,None]*dt*np.arange(len(coeff)))*coeff,axis=1))
        psds[f'n{nw}_filter_frequency']=ff;psds[f'n{nw}_filter_amplitude']=response
        support['filter']=dict(taps=len(coeff),factor=2,output_nyquist_hz=.25/dt,amplitude_at_hz={str(f):float(abs(np.sum(np.exp(-2j*np.pi*f*dt*np.arange(len(coeff)))*coeff))) for f in (0,1,3,5,10,11,15,20,25,30,40,50)})
        output.append(support)
    np.savez_compressed(dest/'population-spectra.npz',**psds,**{k+'_frequency':v for k,v in freqs.items()})
    import gzip
    (dest/'window-bands.json.gz').write_bytes(gzip.compress((json.dumps(allbands,separators=(',',':'))+'\n').encode(),mtime=0))
    save_json(dest/'population-support.json',output);save_json(dest/'population-coherence.json',coherences);save_json(dest/'population-events.json',eventrows)
    grouped=[]
    for nw in (11,12):
        for domain in ('native-matched','all-stage-matched'):
            subset=[r for r in allbands if r['network']==nw and r['domain']==domain]
            for stage in sorted({r['stage'] for r in subset}):
                for metric in sorted({r['metric'] for r in subset}):
                    rows=[r for r in subset if r['stage']==stage and r['metric']==metric and r['value'] is not None];detectors=sorted({r['detector'] for r in rows});means=[np.mean([r['value'] for r in rows if r['detector']==d]) for d in detectors]
                    grouped.append(dict(network=nw,domain=domain,stage=stage,metric=metric,window_count=len(rows),detector_mean=distribution(means),window_population=distribution([r['value'] for r in rows]),detector_prominent_window_fraction=distribution([np.mean([r['value']>5 for r in rows if r['detector']==d]) for d in detectors]) if metric=='line_prominence' else None,prominent_gt5_fraction=float(np.mean([r['value']>5 for r in rows])) if metric=='line_prominence' and rows else None))
    save_json(dest/'population-band-summary.json',grouped)
    save_json(dest/'diagnostic-recipe.json',dict(seconds=time.monotonic()-started,source_exclusion=f'{background_family or family} REF M0 centroid, radius4*majorFWHM, actual native detector pointing',matched_other_arm=match_family,full_support_counts='own-arm full domain before intersection; spectral windows only require both arms when matched_other_arm is set',spectra='4s rounded to even native N;50% overlap; median centering; Hann; one-sided PSD dt*absFFT²/sumh²; complete contiguous matched windows only; no zero fill',native_units='native x times accepted sample CAL multiplier; mJy²/nominalbeam²/Hz',cadence='native full band; all-stage matched uses identical physical length and common bins up to output Nyquist; absent higher output frequencies not measured suppression',population='all detectors included in support denominator; mapping-unavailable/source-overlap/incomplete-window detectors explicitly lack spectral estimate',line_recipe='9.75–12.25Hz band, median continuum7–9.5/12.5–15Hz; nonnegative excess above continuum; maxbin/continuum>5 descriptive prominence only, not cleaning threshold',coherence='all pairs with >=3 genuine common windows; bin nearest11Hz; not attribution',event_recipe='adjacent absolute differences >8 raw matched-support MAD; retained edge within0.25s of unavailable sample; descriptive, not an independent event completeness claim'))
    print(json.dumps(dict(networks=[dict(network=r['network'],detectors=r['detectors'],native_rows=r['native_rows'],spectral_detectors=sum(n>0 for n in r['complete_spectral_windows_per_detector'])) for r in output],seconds=time.monotonic()-started)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--family',default='A');p.add_argument('--output',type=Path,required=True);a=p.parse_args();population(a.root,a.family,a.output)
