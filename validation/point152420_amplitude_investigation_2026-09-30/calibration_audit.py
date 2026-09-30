"""Read-only identity and arithmetic audit; no observation-profile start."""
from pathlib import Path
import hashlib, json
import numpy as np
import yaml
from astropy.table import Table
from netCDF4 import Dataset
D=Path(__file__).resolve().parent
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930/B-REF')
H=Path('/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/apts/hero/apt_152420_matched.ecsv')
S=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/v2/science/apts/v2/apt_152420_matched.apt-v2/sha256-8dd226355eafbbf9b803d1a8001a31a9e852b956040d64e0495d7db2da5974d4.apt.ecsv')
HST=H.parents[2]/'pointings/reduced/redu09/152420/raw/toltec_commissioning_pointing_152420_stats.nc'
def yread(p):return yaml.load(Path(p).read_text(),Loader=yaml.CSafeLoader)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def bound(p):return dict(path=str(p),sha256=digest(p),bytes=Path(p).stat().st_size)
def array(p,dtype,shape=None):
 a=np.fromfile(p,dtype=dtype)
 return a.reshape(shape) if shape else a
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def tables():
 h,s=Table.read(H),Table.read(S)
 key=lambda t,i:(int(t['nw'][i]),int(t['kids_tone'][i]))
 hd={key(h,i):i for i in range(len(h))};sd={key(s,i):i for i in range(len(s))}
 assert len(hd)==len(h) and len(sd)==len(s) and hd.keys()==sd.keys()
 for k in hd:
  for field in ('array','tone_freq','kids_f_out'):
   assert h[field][hd[k]]==s[field][sd[k]],(k,field)
 return h,s,hd,sd

def main():
 h,s,hd,sd=tables();rows=[];samples=[];native=[]
 with Dataset(HST) as ds:
  for field in ('nw','array','flxscale'):
   assert np.array_equal(ds['apt_'+field][:],np.asarray(h[field]),equal_nan=True),field
  weights=np.asarray(ds['weights'][:]);frac=np.asarray(ds['flagged_frac'][:]);actual=[]
  for nw in range(5):
   ix=np.asarray(h['nw'])==nw
   active=ix & np.any((weights>0)&np.isfinite(weights)&(frac<1),axis=0)
   actual.append(dict(network=nw,detectors=int(ix.sum()),positive_weight_some_unflagged=int(active.sum()),positive_weight_scan_detector_pairs=int(np.sum((weights[:,ix]>0)&np.isfinite(weights[:,ix])&(frac[:,ix]<1)))))
 for nw in range(5):
  keys=[k for k in hd if k[0]==nw];ii=np.array([hd[k] for k in keys]);jj=np.array([sd[k] for k in keys]);hf=np.asarray(h['flxscale'][ii],float);sf=np.asarray(s['flxscale'][jj],float)
  good=(np.asarray(h['flag'][ii])==0)&(np.asarray(s['flag'][jj])==0)&np.isfinite(hf)&(hf>0)&np.isfinite(sf)&(sf>0)&~np.ma.getmaskarray(s['flxscale'][jj])
  ratio=sf[good]/hf[good];rec=dict(network=nw,historical_flag0=int(np.sum(h['flag'][ii]==0)),successor_flag0=int(np.sum(s['flag'][jj]==0)),common=int(good.sum()),ratio_successor_over_historical_quantiles=np.quantile(ratio,[0,.1,.5,.9,1]).tolist())
  rows.append(rec)
  if nw==0:continue
  root=R/f'upstream/products/network{nw}';cal=root/'donor-continuity/cal';receipt=yread(cal/'receipt.yaml');pr=yread(root/'source-probe/receipt.yaml');shape=(pr['native_rows'],pr['detectors']);mult=array(root/'source-probe/cal-multiplier.f64','<f8',shape);channels=pr['channels'];errors=[];gains=[];ratios=[]
  # All available detector samples, independently evaluate actual CAL/RTC arithmetic.
  # Probe multipliers are a comparator, not an independent definition of truth.
  available_channels={d['channel'] for d in receipt['detectors'] if d['available']>0}
  ranked=sorted([i for i in np.flatnonzero(good) if keys[i][1] in available_channels],key=lambda i:sf[i]/hf[i]);representatives={keys[ranked[int(q*(len(ranked)-1))]][1] for q in (0,.25,.5,.75,1)}
  for det in receipt['detectors']:
   channel=det['channel'];j=channels.index(channel);idx=int(det['selected_row'].split('row=')[1]);assert (int(s['nw'][idx]),int(s['kids_tone'][idx]))==(nw,channel)
   if det['available']==0:continue
   assert det['selected_flxscale']==float(s['flxscale'][idx])
   slots=array(cal/f'channel-{channel}-slot.i64','<i8');cv=array(cal/f'channel-{channel}-value.f64','<f8');fv=array(root/f'donor-continuity/{channel}-filtered.f64','<f8',(shape[0],2))[slots,0]
   prediction=fv*mult[slots,j];err=float(np.max(abs(cv-prediction))/max(1.,np.max(abs(cv))));assert err<1e-12;errors.append(err)
   hi=hd[(nw,channel)];gooddet=h['flag'][hi]==0 and s['flag'][idx]==0 and np.isfinite(h['flxscale'][hi]) and h['flxscale'][hi]>0
   if not gooddet:continue
   usable=abs(fv)>1e-20
   gain=(cv[usable]/fv[usable])/float(s['flxscale'][idx]);assert np.all(np.isfinite(gain)) and np.all(gain>0)
   gains.append(float(np.median(gain)));ratios.append(float(np.median(mult[slots,j]/float(h['flxscale'][hi]))))
   if channel in representatives:
    k=int(np.flatnonzero(usable)[len(np.flatnonzero(usable))//2]);x=float(fv[k]);samples.append(dict(network=nw,channel=channel,slot=int(slots[k]),historical_row=hi,successor_row=idx,native_filtered_x=x,historical_factor=float(h['flxscale'][hi]),successor_factor=float(s['flxscale'][idx]),actual_successor_CAL=float(cv[k]),historical_independent_mJy=x*float(h['flxscale'][hi]),inferred_applied_atmosphere=float(cv[k]/x/float(s['flxscale'][idx])),probe_multiplier_error=float(cv[k]-prediction[k])))
  native.append(dict(network=nw,checked_available_detectors=len(errors),maximum_relative_CAL_vs_filtered_times_probe_error=max(errors),common_applied_atmosphere_detector_median_quantiles=np.quantile(gains,[0,.1,.5,.9,1]).tolist(),common_total_conversion_successor_over_historical_detector_median_quantiles=np.quantile(ratios,[0,.1,.5,.9,1]).tolist()))
 save(D/'results/calibration-audit.json',dict(inputs=[bound(p) for p in (H,S,HST)],identity='unique (nw,kids_tone), exact array/tone_freq/kids_f_out equality on all 5518 rows; selected CAL row and factor checked',historical_APT_snapshot_verified_fields=['nw','array','flxscale'],historical_population=actual,APT_factor_distribution=rows,native_CAL_arithmetic=native,representative_native_conversions=samples,scope='Same successor post-RTC native x samples; historical factor applied once, historical extinction disabled. Does not reproduce historical RTC or prove which APT absolute scale is correct. Atmosphere is inferred from actual CAL/filtered ratio, not independently reimplemented passband physics.'))
 print(json.dumps(dict(population=actual,factors=rows,arithmetic=native),indent=2))
if __name__=='__main__':main()
