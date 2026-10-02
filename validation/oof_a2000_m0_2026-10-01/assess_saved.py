"""Saved-product assessment only. No reduction, model, optical fit, or frame adapter."""
import os
os.environ.update(MPLBACKEND='Agg',MPLCONFIGDIR='/private/tmp/oof-assess-mpl',XDG_CACHE_HOME='/private/tmp/oof-assess-cache',PYTHONDONTWRITEBYTECODE='1')
import sys,json,re
from pathlib import Path
import numpy as np,yaml,netCDF4
from astropy.io import fits
from astropy.nddata import Cutout2D
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_fruit_trajectory import bound,verified
from point_additive_diagnostics import save_json
from point_storage import tree_bytes
from inspect_forward_map import require
from oof_saved_interface import embed_altaz
sys.path.insert(0,str(Path(__file__).parent));from run_campaign import ROOT,PRIOR,ROLES,CAP,read,verify_role
OBS=(152385,152386,152387)
def axes(h,shape):
    f=[3600 if h.get('CUNIT'+str(i),'deg')=='deg' else 1 for i in (1,2)]
    return tuple((np.arange(n)+1-h['CRPIX'+str(i)])*h['CDELT'+str(i)]*f[i-1] for i,n in ((1,shape[1]),(2,shape[0])))
def metrics(z,v,x,y,exposure=None):
    good=v&np.isfinite(z);a=z[good];xx,yy=np.meshgrid(x,y);step=abs((x[1]-x[0])*(y[1]-y[0]));j,i=np.unravel_index(np.nanargmax(np.where(good,z,np.nan)),z.shape)
    labels,n=label(~good);edge=set(np.unique(np.r_[labels[0,:],labels[-1,:],labels[:,0],labels[:,-1]]));internal=sum(np.sum(labels==k) for k in range(1,n+1) if k not in edge)
    r=np.hypot(xx,yy);wing=good&(r>=20)&(r<50);wv=z[wing]
    result=dict(shape=list(z.shape),supported=int(good.sum()),unsupported=int(z.size-good.sum()),internal_missing_pixels=int(internal),pixel_area_arcsec2=float(step),sample_x_range_arcsec=[float(min(x)),float(max(x))],sample_y_range_arcsec=[float(min(y)),float(max(y))],peak=float(a.max()),minimum=float(a.min()),median=float(np.median(a)),q01=float(np.quantile(a,.01)),q99=float(np.quantile(a,.99)),negative_fraction=float(np.mean(a<0)),negative_to_positive_integral=float(-a[a<0].sum()/a[a>0].sum()),signed_brightness_integral=float(a.sum()*step),peak_native_offset_arcsec=[float(x[i]),float(y[j])],brightness_unit='mJy/nominal beam; no absolute flux qualification',integral_unit='mJy/nominal beam arcsec^2',annulus20to50=dict(pixels=len(wv),median=float(np.median(wv)) if len(wv) else None,negative_fraction=float(np.mean(wv<0)) if len(wv) else None,interpretation='field structure, not established source-free background or noise'))
    if exposure is not None:result['exposure_statistic']=dict(minimum=float(exposure[good].min()),median=float(np.median(exposure[good])),maximum=float(exposure[good].max()))
    return result
def crop(z,v,x,y,n,exp):
    c=Cutout2D(z,(z.shape[1]/2,z.shape[0]/2),(n,n));s=c.slices_original;return c.data,v[s],x[s[1]],y[s[0]],exp[s],[[a.start,a.stop] for a in s]
def plot_map(ax,z,v,x,y,title,limit=None,norm=False):
    data=np.where(v,z,np.nan);data=data/np.nanmax(data) if norm else data
    ax.pcolormesh(x,y,data,cmap='RdBu_r',vmin=-.3 if norm else -100,vmax=1 if norm else 560,shading='nearest',rasterized=True)
    if limit:ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit)
    ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.set_xlabel('native longitude offset (arcsec)');ax.set_ylabel('native latitude offset (arcsec)')
def main(destination=None):
    status=read(ROOT/'STATUS.json');require(status['state']=='COMPLETE_PENDING_OWNER_REVIEW','only completed frozen batch can be assessed')
    freeze=read(ROOT/'FREEZE.json');rev=freeze['source_revision'];out=Path(destination) if destination else ROOT/'report';require(out.resolve().is_relative_to(ROOT.resolve()),'report must remain in campaign accounting scope');out.mkdir(exist_ok=False)
    consumers=read(ROOT/'preflight'/rev/'consumers/CONSUMERS.json');reference=read(ROOT/'preflight'/rev/'REFERENCE.json')
    attempts=[json.loads(s) for s in (ROOT/'attempts.jsonl').read_text().splitlines()];require(len(attempts)==6 and all(sum(a['kind']==k and a['identity']['run']==r for a in attempts)==1 for r in ROLES for k in ('parent','trajectory')),'attempt accounting differs')
    records=[];maps=[];rows=[]
    for obs,rr,consumer in zip(OBS,reference['maps'],consumers['maps']):
        p=ROOT/f'S-{obs}';verify_role(p,rev);launch=read(p/'LAUNCH.json');sp=p/'M0/products/a2000/map.fits';rp=verified(rr['map'])
        with fits.open(sp,checksum=True) as h:
            sh=h[0].header.copy();s=h[0].data.copy();sv=h['SCI_SUPPORT'].data.astype(bool);hits=h['HITS'].data.copy();normalization=h['NORMALIZATION'].data.copy()
        require(np.isfinite(s[sv]).all() and np.isnan(s[~sv]).all(),'native missingness');require(np.array_equal(normalization,hits),'uniform coefficient normalization differs')
        with fits.open(rp,checksum=True) as h:
            rh=h['signal_I'].header.copy();r=np.squeeze(h['signal_I'].data).copy();rw=np.squeeze(h['weight_I'].data);rv=np.isfinite(r)&np.isfinite(rw)&(rw>0);rc=np.squeeze(h['coverage_I'].data).copy()
        sx,sy=axes(sh,s.shape);rx,ry=axes(rh,r.shape)
        small=crop(s,sv,sx,sy,16,hits);wide=crop(s,sv,sx,sy,50,hits);rwide=crop(r,rv,rx,ry,100,rc)
        try:embed_altaz(s,sv,sh)
        except ValueError as e:frame_rejection=str(e)
        else:raise ValueError('FK5 incorrectly admitted to AltAz consumer')
        bundle=yaml.safe_load((p/'M0/products/a2000/bundle.yaml').read_text());plan=yaml.safe_load((p/'M0/products/a2000/plan.yaml').read_text());require(bundle['observation']==f'obs:{obs}_0_2'==plan['observation'],'reassess changed known identity defect')
        networks=[];ratios=[]
        rstats=rp.parent/f'toltec_commissioning_oof_{obs}_stats.nc'
        with netCDF4.Dataset(rstats) as nc:
            rn=np.asarray(nc['apt_nw'][:]);rch=np.asarray(nc['apt_kids_tone'][:]);rf=np.asarray(nc['apt_flxscale'][:]);rflag=np.asarray(nc['apt_flag'][:])
        for nw in (11,12):
            base=p/f'upstream/products/network{nw}/donor-continuity';cal=yaml.safe_load((base/'cal/receipt.yaml').read_text());ptc=yaml.safe_load((base/'ptc/receipt.yaml').read_text());cfg=read(p/f'inputs/network{nw}-input.json')
            require(cfg['scan']==1 and cfg['subobservation']==0,'source acquisition mismatch');require(f'inputs:{obs}_0_1:astrometry' in cal['AST_pointing_offset'],'AST acquisition mismatch')
            handoffs=list((p/'M0/products/a2000/handoff').glob(f'network-{nw}-segment-*-ast.yaml'))
            require(all(yaml.safe_load(f.read_text())['AST_telescope']==cal['AST_telescope'] for f in handoffs),'MAP/CAL telescope mismatch')
            require(ptc['rank']==3 and ptc['iteration_limit']==500 and float(ptc['relative_objective_tolerance'])==1e-5 and not ptc['failed_fits'],'PTC policy differs')
            cause={};factor_ratios=[]
            for d in cal['detectors']:
                idx=np.flatnonzero((rn==nw)&(rch==d['channel']));require(len(idx)==1,'reference calibration identity not unique')
                f=d.get('selected_flxscale');g=rf[idx[0]]
                if f is not None and np.isfinite(f) and f>0 and np.isfinite(g) and g>0:factor_ratios.append(float(f/g))
                for k,v in d['joint_cause_counts'].items():cause[str(k)]=cause.get(str(k),0)+v
            ratios+=factor_ratios
            networks.append(dict(network=nw,cal_available=cal['available'],cal_scheduled=cal['scheduled'],cal_detectors_with_samples=sum(d['available']>0 for d in cal['detectors']),cause_bits=cal['cause_bits'],joint_cause_counts=cause,flxscale_ratio_S_over_R=dict(count=len(factor_ratios),min=min(factor_ratios),median=float(np.median(factor_ratios)),max=max(factor_ratios)),factor_application=cal['application_counts_on_available_samples'],tau225=cal['WVR_constant_tau225'],WVR_limitation=cal['WVR_input_limitation'],ptc_available=ptc['available'],ptc_scheduled=ptc['scheduled_in_processing_segments'],iterations=[x['iterations'] for x in ptc['segments']],receipt_bindings=[bound(base/'cal/receipt.yaml'),bound(base/'ptc/receipt.yaml')]))
        avail=sum(n['cal_available'] for n in networks);scheduled=sum(n['cal_scheduled'] for n in networks);require(bundle['MAP_route_admitted']==avail and bundle['AST_invalid_coordinates']==0,'MAP admission/support discrepancy')
        a=metrics(small[0],small[1],small[2],small[3],small[4]);b=metrics(wide[0],wide[1],wide[2],wide[3],wide[4]);a['normalization_divisor']=a['peak']
        record=dict(observation=obs,focus_relative_middle_mm=(obs-152386)*2,successor=bound(sp),reference=bound(rp),source_revision=rev,execution=bound(p/'EXECUTION.json'),launch=bound(p/'LAUNCH.json'),resources=dict(seconds=launch['exit_seconds'],sampled_tree_RSS_bytes=launch['aggregate_tree_sampled_RSS'],cpu_seconds=launch['cpu_seconds'],retained_bytes=tree_bytes(p)),S_parent=metrics(s,sv,sx,sy,hits),R_parent=metrics(r,rv,rx,ry,rc),S_native_100arcsec=b,S_native_32arcsec=a,R_native_100arcsec=metrics(rwide[0],rwide[1],rwide[2],rwide[3],rwide[4]),R_operational=consumer,calibration=networks,CAL_available_fraction=avail/scheduled,MAP_admitted=avail,AST_invalid_coordinates=0,MAP_acquisition_identity=dict(expected=f'obs:{obs}_0_1',published=bundle['observation'],status='provenance conformance defect; original products preserved',scope='hardcoded strings map_saved_execution.h lines181,266; actual CAL/APT/AST loaded with OOF scan1'),consumer_admission=dict(status='S AltAz consumers unavailable',reason=frame_rejection,native_windows='unrecentered map-center diagnostics only, same32/100arcsec spans; not LM option2 source-centered input',crop_slices=dict(small=small[5],wide=wide[5]),normalization='native32 normalized separately for display only; raw parent and divisor retained; no R/S fitted gain'))
        records.append(record);maps.append((r,rv,rx,ry,s,sv,sx,sy));rows.append((small,wide,rwide))
        np.savez_compressed(out/f'{obs}-native-windows.npz',small=small[0],small_support=small[1],small_x=small[2],small_y=small[3],small_normalized=small[0]/a['peak'],wide=wide[0],wide_support=wide[1],wide_x=wide[2],wide_y=wide[3])
    for name,limit in [('full-context',210),('central-100arcsec',50)]:
        fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        for i,(obs,parts) in enumerate(zip(OBS,maps)):
            plot_map(axs[0,i],*parts[:4],f'R {obs}: AltAz, 1″ JINC',limit)
            plot_map(axs[1,i],*parts[4:],f'S {obs}: FK5 tangent, 2″ ordinary',limit)
        fig.suptitle('a2000 cold M0: native frames shown separately; no rotation or paired subtraction\nCommon brightness scale −100 to560 mJy/beam; empty support blank')
        sm=plt.cm.ScalarMappable(cmap='RdBu_r',norm=plt.Normalize(-100,560));fig.colorbar(sm,ax=axs,label='mJy/nominal beam',shrink=.75);fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for i,(obs,row,record) in enumerate(zip(OBS,rows,records)):
        c=record['R_operational'];npz=np.load(verified(c['windows']));plot_map(axs[0,i],npz['ml_normalized'],npz['ml_support'],np.array(c['ml']['x_centers_arcsec']),np.array(c['ml']['y_centers_arcsec']),f'R {obs}: actual ML32×32 AltAz',16,norm=False)
        # Replace color limits for normalized-only display, identical for all six.
        axs[0,i].collections[0].set_clim(-.3,1)
        small=row[0];plot_map(axs[1,i],*small[:4],f'S {obs}:16×16 FK5 diagnostic',16,norm=True)
    fig.suptitle('32″ windows, each divided by its own peak; parent amplitude retained separately\nS is not an admitted ML/AltAz input; no geometric alignment asserted')
    fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r',norm=plt.Normalize(-.3,1)),ax=axs,label='own-window peak fraction',shrink=.75);fig.savefig(out/'normalized-32arcsec.png',dpi=150);plt.close(fig)
    total=sum(tree_bytes(p) for p in PRIOR)+tree_bytes(ROOT);require(total<CAP,'shared disk ceiling exceeded')
    save_json(out/'MEASUREMENTS.json',dict(executed_source=rev,attempts=attempts,observations=records,storage=dict(shared_bytes=total,ceiling_bytes=CAP,role_bytes=sum(x['resources']['retained_bytes'] for x in records)),runtime_total_seconds=sum(x['resources']['seconds'] for x in records),worst_sampled_tree_RSS_bytes=max(x['resources']['sampled_tree_RSS_bytes'] for x in records),noise='not established: no certified source-free support selected; annular statistics are contamination/structure, not sensitivity',quality='not established for operational OOF consumers; native maps preserve focus structure but show contamination and provenance defect',figures=[bound(p) for p in out.glob('*.png')],extra_reductions=0,optical_fits=0,model_inferences=0))
    print(json.dumps([dict(obs=x['observation'],native32peak=x['S_native_32arcsec']['peak'],native100min=x['S_native_100arcsec']['minimum'],native100support=x['S_native_100arcsec']['supported'],CAL_fraction=x['CAL_available_fraction'],disk=x['resources']['retained_bytes']) for x in records],indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else None)
