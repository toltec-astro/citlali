"""Authenticate and compare completed saved maps; no coordinate repair or reduction."""
from campaign import *
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def axes(h,shape):
    return tuple((np.arange(n)+1-h['CRPIX'+str(i)])*h['CDELT'+str(i)]*(3600 if h.get('CUNIT'+str(i),'deg')=='deg' else 1) for i,n in ((1,shape[1]),(2,shape[0])))
def window(z,v,x,y,half=None):
    ix=np.ones(len(x),bool) if half is None else (x>=-half-1e-8)&(x<half-1e-8)
    iy=np.ones(len(y),bool) if half is None else (y>=-half-1e-8)&(y<half-1e-8)
    return z[np.ix_(iy,ix)],v[np.ix_(iy,ix)],x[ix],y[iy]
def metrics(z,v,x,y,half=None):
    z,v,x,y=window(z,v,x,y,half);good=v&np.isfinite(z);a=z[good];require(len(a)>0,'empty comparison region')
    iy,ix=np.unravel_index(np.nanargmax(np.where(good,z,np.nan)),z.shape)
    xx,yy=np.meshgrid(x,y);ring=good&(np.hypot(xx,yy)>=20)&(np.hypot(xx,yy)<50)
    return dict(shape=list(z.shape),supported=int(good.sum()),missing=int((~good).sum()),peak=float(a.max()),peak_xy_arcsec=[float(x[ix]),float(y[iy])],minimum=float(a.min()),q01=float(np.quantile(a,.01)),median=float(np.median(a)),negative_fraction=float(np.mean(a<0)),annulus20to50_median=float(np.median(z[ring])) if ring.any() and (half is None or half>=50) else None,annulus_is_noise=False)
def load_map(ref,kind):
    with fits.open(verified(ref),checksum=True) as f:
        if kind=='R':
            h=f['signal_I'].header.copy();z=np.squeeze(f['signal_I'].data).copy();weight=np.squeeze(f['weight_I'].data);v=np.isfinite(z)&np.isfinite(weight)&(weight>0)
        else:
            h=f[0].header.copy();z=f[0].data.copy();v=f['LOCAL_VALID' if kind=='JINC' else 'SCI_SUPPORT'].data.astype(bool)
            require(np.array_equal(np.isfinite(z),v),'published signal missingness differs')
    return z,v,*axes(h,z.shape)
def panel(ax,m,title,half,divisor=None):
    z,v,x,y=m;z=np.where(v,z,np.nan)
    if divisor is not None:z=z/divisor
    im=ax.pcolormesh(x,y,z,shading='nearest',cmap='RdBu_r',vmin=-.3 if divisor else -100,vmax=1 if divisor else 560,rasterized=True)
    ax.set_xlim(-half,half);ax.set_ylim(-half,half);ax.set_aspect('equal');ax.set_title(title,fontsize=10)
    ax.set_xlabel('native longitude offset (arcsec)',fontsize=9);ax.set_ylabel('native latitude offset (arcsec)',fontsize=9)
    return im

def main():
    analysis_revision=revision();out=ROOT/'assessment';out.mkdir(exist_ok=False)
    status=read(ROOT/'STATUS.json');p=read(ROOT/'PREPARED.json');f=read(ROOT/'FREEZE.json');old=read(PARENT/'report/MEASUREMENTS.json')
    require(status['state']=='COMPLETE_PENDING_OWNER_REVIEW' and [r['observation'] for r in status['completed']]==list(OBS),'three completed roles required')
    attempts=[json.loads(x) for x in (ROOT/'attempts.jsonl').read_text().splitlines()]
    require([a['observation'] for a in attempts]==list(OBS) and len(attempts)==3,'unexpected attempts')
    require(f['source_revision']==p['source_revision'],'frozen source mismatch');verified(p['helper'])
    for b in f['bindings']+p['bindings']:verified(b)
    rows=[];maps=[]
    for obs,done,state,attempt in zip(OBS,status['completed'],p['states'],attempts):
        b=read(verified(done['bundle']));r=yread(verified(b['receipt']));q=read(verified(state['request']));dest=ROOT/str(obs)
        require(attempt['request']==state['request'] and b['parent']==state and r['request_sha256']==state['request']['sha256'],'execution request/parent mismatch')
        require(r['observation']==b['observation']==f'obs:{obs}_0_1' and r['frame']==b['frame']=='FK5','new acquisition/frame mismatch')
        require(r['samples']==q['samples'] and r['workers']==1 and r['spec']==q['spec'],'realized numerical settings differ')
        require((ROOT/f'{obs}.log').read_text()=='','unexpected native output')
        for name,sha in r['files'].items():require(digest(dest/name)==sha,'native output changed')
        verified(b['diagnostics']);verified(b['map'])
        with fits.open(verified(b['map']),checksum=True) as ff:
            shape=ff[0].data.shape;v=ff['LOCAL_VALID'].data.astype(bool)
            for name,file in [('NUMERATOR','N.f64'),('NORMALIZATION','C.f64'),('QUADRATIC','Q.f64'),('COEFFSQ_TIME','T.f64')]:
                require(np.array_equal(ff[name].data,np.fromfile(dest/file,dtype='<f8').reshape(shape)),'FITS numerical role fidelity')
            m=np.fromfile(dest/'m.f64',dtype='<f8').reshape(shape);causes=np.fromfile(dest/'causes.u8',dtype='u1').reshape(shape)
            require(np.array_equal(v,causes==0) and np.array_equal(ff[0].data[v],m[v]),'signal/local support fidelity')
            err=np.fromfile(dest/'error-4.f64',dtype='<f8').reshape(shape);tol=np.fromfile(dest/'tolerance-lower.f64',dtype='<f8').reshape(shape)
            require(np.all(err[v]<=tol[v]),'published signal lacks certificate')
            require(ff[0].header['SCAN']==1 and ff[0].header['OBSID']==state['identity'],'FITS scan identity differs')
        # Numerical support is not empirical coverage or trustworthy edge brightness.
        with fits.open(verified(b['map'])) as ff:
            signal=ff[0].data;hh=ff[0].header;c=ff['NORMALIZATION'].data;xx,yy=axes(hh,signal.shape)
            abs_c=np.fromfile(dest/'absolute-C.f64',dtype='<f8').reshape(signal.shape)
            members=np.fromfile(dest/'members.u64',dtype='<u8').reshape(signal.shape);extrema=[]
            for label,fun in [('maximum',np.nanargmax),('minimum',np.nanargmin)]:
                iy,ix=np.unravel_index(fun(signal),signal.shape)
                extrema.append(dict(kind=label,signal=float(signal[iy,ix]),xy_arcsec=[float(xx[ix]),float(yy[iy])],signed_C=float(c[iy,ix]),absolute_C=float(abs_c[iy,ix]),cancellation_ratio=float(abs(c[iy,ix])/abs_c[iy,ix]),members=int(members[iy,ix]),numerical_error_bound=float(err[iy,ix])))
        rr=next(x for x in old['observations'] if x['observation']==obs)
        refs=dict(R=rr['reference'],ordinary=rr['successor'],JINC=b['map']);loaded={k:load_map(ref,k) for k,ref in refs.items()}
        measures={k:{str(span):metrics(*m,None if span=='full' else span/2) for span in ('full',100,32)} for k,m in loaded.items()}
        values,counts=np.unique(causes,return_counts=True)
        row=dict(observation=obs,focus_relative_middle_mm=(obs-152386)*2,parents=refs,metrics=measures,certificate=dict(supported=int(v.sum()),unavailable=int((~v).sum()),cause_counts={str(k):int(n) for k,n in zip(values,counts)},max_accepted_signal_error=float(err[v].max()),max_error_to_tolerance=float((err[v]/tol[v]).max()),cause_bits={'1':'range/FP','2':'denominator','4':'ancillary','8':'signal','16':'no members'},frozen_scale=state['peak_scale']),runtime=dict(role_seconds=done['seconds'],native_wall_seconds=r['wall_seconds'],native_cpu_seconds=r['cpu_seconds'],cumulative_child_peak_RSS_bytes=done['child_peak_rss_bytes_so_far']),scan_identity='scan1 correct in receipt and FITS; predecessor scan2 strings preserved',same_sample_replay=state['ordinary_replay'],full_map_extrema=extrema)
        rows.append(row);maps.append(loaded)
        save_json(out/(str(obs)+'-measurements.json'),row)
    for name,half,norm in [('central-100arcsec',50,False),('normalized-100arcsec',50,True),('normalized-32arcsec',16,True),('full-context',210,False)]:
        fig,axs=plt.subplots(3,3,figsize=(13,12),constrained_layout=True)
        for j,(obs,loaded,row) in enumerate(zip(OBS,maps,rows)):
            for i,(key,label) in enumerate([('R','Historical: AltAz, 1″ JINC'),('ordinary','Same samples: FK5, 2″ ordinary'),('JINC','New: FK5, 1″ JINC')]):
                divisor=row['metrics'][key]['32']['peak'] if norm else None
                panel(axs[i,j],loaded[key],f'{obs} | {label}',half,divisor)
        fig.suptitle('a2000 OOF: native frames; no geometric alignment or fitted rescaling\n'+('Each panel divided by its own central32″ peak; raw amplitudes retained' if norm else 'Common raw brightness scale; unsupported pixels blank'))
        cb=plt.cm.ScalarMappable(norm=Normalize(-.3,1) if norm else Normalize(-100,560),cmap='RdBu_r')
        fig.colorbar(cb,ax=axs,label='central32″ peak fraction' if norm else 'mJy / nominal beam',shrink=.7)
        fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    require(revision()==analysis_revision,'analysis source changed')
    result=dict(executed_source=p['source_revision'],analysis_source=analysis_revision,analysis_script=bound(Path(__file__)),status=bound(ROOT/'STATUS.json'),freeze=bound(ROOT/'FREEZE.json'),prepared_bindings_verified=len(p['bindings']),observations=rows,figures=[bound(x) for x in sorted(out.glob('*.png'))],attempts=attempts,storage=capacity(),wall_seconds=status['seconds'],assessment_scope='saved maps only; fixed native-centered half-open square windows; R AltAz and S FK5 are not aligned. No coordinate changes, new cleaning, fits, noise inference or optical solver. Normalized panels divide by own central32 peak only, never alter parent data.')
    save_json(out/'MEASUREMENTS.json',result)
    for row in rows:print(row['observation'],{k:round(v['32']['peak'],6) for k,v in row['metrics'].items()},'JINC100',row['metrics']['JINC']['100'],flush=True)
if __name__=='__main__':main()
