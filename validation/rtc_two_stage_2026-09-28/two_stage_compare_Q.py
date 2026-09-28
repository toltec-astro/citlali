"""Saved B/Q maps, measurements, native decisions and support; no science reruns."""
from pathlib import Path
import sys,json
import numpy as np,yaml
from astropy.io import fits
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_empirical import coordinates,stats
MEASUREMENT_NAMES=['peak','cx','cy','FWHM_major','FWHM_minor','major_angle','constant','template','aperture1','aperture1.5','aperture2']
def canonical(m):
 p=np.array(m['guard']['accepted_parameters'],float);major=3 if p[3]>=p[4] else 4;minor=7-major;angle=p[5]+(np.pi/2 if major==4 else 0);angle=.5*np.arctan2(np.sin(2*angle),np.cos(2*angle));f=np.sqrt(8*np.log(2))
 return np.array([*p[:3],p[major]*f,p[minor]*f,angle,p[6],m['frozen_template_projection'],*[a['value_mJy_per_beam_arcsec2'] for a in m['apertures']]],float),major,minor
def sigmas(summary,m):
 s={v['name']:v['conditional_sigma'] for v in summary['endpoint_uncertainty']};_,major,minor=canonical(m);f=np.sqrt(8*np.log(2));s.update(FWHM_major=s['sx' if major==3 else 'sy']*f,FWHM_minor=s['sx' if minor==3 else 'sy']*f,major_angle=s['angle']);return s
from inspect_point_source import gaussian
from inspect_forward_map import require,digest
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-two-stage-20260928');D=R/'report';B=R/'B-REF-rep1';Q=R/'Q-REF'
def read(p):return json.loads(p.read_text())
def y(p):return yaml.load(p.read_text(),Loader=yaml.CSafeLoader)
def load(p):
 with fits.open(p) as h:return {a.name:a.data.copy() for a in h},dict(h[0].header)
def mapfile(run,k):return run/'M0/products/a1100/map.fits' if k==0 else run/f'run/BASE-POINT/REF/a1100/cycle-{k:02d}/native/products/total.fits'
bs=read(B/'run/BASE-POINT/REF/a1100/SUMMARY.json');qs=read(Q/'run/BASE-POINT/REF/a1100/SUMMARY.json');records=[];packed={}
for k in range(6):
 b,h=load(mapfile(B,k));q,qh=load(mapfile(Q,k));require(all(h[a]==qh[a] for a in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2')),'B/Q grid differs');bv=b['SCI_SUPPORT'].astype(bool);qv=q['SCI_SUPPORT'].astype(bool);common=bv&qv;delta=q['PRIMARY']-b['PRIMARY'];norm=q['NORMALIZATION']-b['NORMALIZATION'];x,z=coordinates(delta.shape,h);disk=np.hypot(x,z)<=40
 bm=bs['histories'][k]['measurement'];qm=qs['histories'][k]['measurement'];bvec,_,_=canonical(bm);qvec,_,_=canonical(qm);bnoise=sigmas(bs,bm);qnoise=sigmas(qs,qm)
 measures=[]
 for name,a,c in zip(MEASUREMENT_NAMES,bvec,qvec):
  delta_value=float(.5*np.arctan2(np.sin(2*(c-a)),np.cos(2*(c-a)))) if name=='major_angle' else float(c-a)
  measures.append(dict(name=name,B=float(a),Q=float(c),delta=delta_value,B_conditional_sigma=bnoise[name] if k==5 else None,Q_conditional_sigma=qnoise[name] if k==5 else None,delta_over_B_sigma=float(delta_value/bnoise[name]) if k==5 and bnoise[name] else None,qualification='canonical major/minor width and major-axis angle modulo pi; B angle fit constrained, its local conditional scale is not a reliable boundary-aware interval' if name in ('FWHM_major','FWHM_minor','major_angle') else None))

 records.append(dict(cycle=k,B_science_pixels=int(bv.sum()),Q_science_pixels=int(qv.sum()),lost_B_science_pixels=int((bv&~qv).sum()),gained_science_pixels=int((qv&~bv).sum()),B_domain_retained_fraction=float(common.sum()/bv.sum()),B_contributions=int(b['HITS'].sum()),Q_contributions=int(q['HITS'].sum()),lost_normalization=float(np.maximum(-norm,0).sum()),gained_normalization=float(np.maximum(norm,0).sum()),changed_normalization_pixels=int((norm!=0).sum()),relative_normalization_on_B_domain=stats((norm/np.where(b['NORMALIZATION']>0,b['NORMALIZATION'],np.nan))[bv]),delta_on_common_science=stats(delta[common]),delta_on_B_fit_disk=stats(delta[common&disk]),measurements=measures,qualifier='Q unavailable on lost B-supported pixels, never zero; same-domain deltas explicitly conditional. Conditional uncertainties are single-map scales, not exact paired significance; selected on B in-sample.'))
 if k==5:
  bres=b['PRIMARY']-gaussian(np.array(bm['guard']['accepted_parameters']),x,z);qres=q['PRIMARY']-gaussian(np.array(qm['guard']['accepted_parameters']),x,z);bg=np.load(B/'run/BASE-POINT/REF/a1100/endpoint/model.npz')['background_mask'].astype(bool)&bv
  win=np.load(D/'new-windows.npz');p=win['n2_new1_map_pixel'];event=np.zeros(delta.size,bool);event[p[p>=0]]=True;event=event.reshape(delta.shape)&bv
  rec=dict(B_background_pixels=int(bg.sum()),Q_available_on_B_background=int((bg&qv).sum()),B_background=stats(bres[bg]),B_on_common_background=stats(bres[bg&qv]),B_lost_background=stats(bres[bg&~qv]),Q_on_B_background=stats(qres[bg&qv]),B_fit_disk=stats(bres[bv&disk]),Q_on_B_fit_disk=stats(qres[common&disk]),selected_B_event_footprint_pixels=int(event.sum()),Q_available_on_B_event_footprint=int((event&qv).sum()),B_event_footprint=stats(bres[event]),B_on_common_event_footprint=stats(bres[event&qv]),B_lost_event_footprint=stats(bres[event&~qv]),lost_event_pixels=[dict(row=int(i),column=int(j),B_residual=float(bres[i,j]),Q_science_available=False,B_hits=float(b['HITS'][i,j]),Q_hits=float(q['HITS'][i,j])) for i,j in np.argwhere(event&~qv)],Q_event_footprint=stats(qres[event&qv]),event_footprint_delta=stats(delta[event&qv]),B_peak_fit_constrained=bm['fit']['constrained'],Q_peak_fit_constrained=qm['fit']['constrained'])
  packed=dict(B=b['PRIMARY'],Q=q['PRIMARY'],difference=np.where(common,delta,np.nan),B_support=bv,Q_support=qv,normalization_difference=norm,B_residual=bres,Q_residual=qres,B_event_footprint=event,B_background=bg,x_arcsec=x,y_arcsec=z)
# Actual native membership, original input protection, no gained support in REF.
support=[]
for nw in (1,2,3,4):
 rb=B/f'upstream/products/network{nw}';rq=Q/f'upstream/products/network{nw}';br=y(rb/'receipt.yaml');qr=y(rq/'receipt.yaml');bp=y(rb/'source-probe/receipt.yaml');qp=y(rq/'source-probe/receipt.yaml');lost=0;gained=0;centerlost=0;centergained=0;changed=[]
 require(qr['whole_run_quarantine']['detectors']==([88] if nw==2 else []),'wrong realized quarantine')
 for field in ('original_x_sha256','original_r_sha256'):require(bp[field]==qp[field],'Q raw parent changed')
 for ch in bp['channels']:
  a=np.fromfile(rb/f'donor-continuity/{ch}-state.u8','u1');b=np.fromfile(rq/f'donor-continuity/{ch}-state.u8','u1');ac=np.fromfile(rb/f'donor-continuity/{ch}-map-center.u8','u1').astype(bool);bc=np.fromfile(rq/f'donor-continuity/{ch}-map-center.u8','u1').astype(bool);av=(a&3)==3;bv2=(b&3)==3;l=int((av&~bv2).sum());g=int((bv2&~av).sum());lost+=l;gained+=g;centerlost+=int((ac&~bc).sum());centergained+=int((bc&~ac).sum())
  if l or g:changed.append(dict(channel=ch,lost=l,gained=g))
  if nw==2 and ch==88:require(not bv2.any() and not bc.any(),'selected run still contributes')
 support.append(dict(network=nw,RTC_paired_lost=lost,RTC_paired_gained=gained,MAP_centers_lost=centerlost,MAP_centers_gained=centergained,changed=changed,events_exact=(rb/'events.yaml').read_bytes()==(rq/'events.yaml').read_bytes(),decisions_exact=(rb/'event-decisions.yaml').read_bytes()==(rq/'event-decisions.yaml').read_bytes(),support_decisions_exact=(rb/'support-decisions.yaml').read_bytes()==(rq/'support-decisions.yaml').read_bytes(),manifest=qr['whole_run_quarantine']['manifest']))
 require(gained==0 and centergained==0,'REF quarantine unexpectedly recovered support')
 require(all(nw==2 and c['channel']==88 for c in changed),'unexpected REF RTC support change')
(D/'B-Q-comparison.json').write_text(json.dumps(dict(cycles=records,final_residuals=rec,native_support=support,uncertainty='Existing conditional uncertainty only, no new ensembles or paired-test significance; baseline-selected in-sample policy test.'),indent=2)+'\n');np.savez_compressed(D/'B-Q-maps.npz',**packed)
print(json.dumps(dict(final=records[-1],residuals=rec,native_support=support),indent=2))
