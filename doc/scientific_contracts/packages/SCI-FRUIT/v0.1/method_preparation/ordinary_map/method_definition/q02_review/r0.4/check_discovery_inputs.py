from pathlib import Path
import json, hashlib, re, subprocess
import numpy as np
import netCDF4
W=Path('/private/tmp/citlali-sci-fruit-method-scope-20260908')
D=Path('/private/tmp/fruit-q02-preflight-20260909')
ROOT=Path('/Users/gwilson/work_toltec/local_data')
paths={152389:ROOT/'fruit-development/point-152389/refactor/reduced/redu00/152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc',123424:ROOT/'beammaps/pointings/reduced/redu01/123424/raw/toltec_commissioning_pointing_123424_ptc_timestream.nc'}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(d,k):
 v=d[k][:]
 assert not np.any(np.ma.getmaskarray(v)), f'masked input {k}'
 return np.asarray(v)
result={'purpose':'Discovery input-coordinate/flag/segment preflight only; no signal or stored weights read; no noise estimation or mapping comparison. Reserved 129081 not opened.','observations':[]}
for obs,path in paths.items():
 with netCDF4.Dataset(path) as f:
  scans=read(f,'scan_indices').astype(int);npts=len(f.dimensions['n_pts']);ndets=len(f.dimensions['n_dets'])
  lengths=np.r_[scans[0,1]+1,np.diff(scans[:,1])].astype(int)
  assert np.all(np.diff(scans[1:,0])==np.diff(scans[1:,1]))
  edges=np.r_[0,np.cumsum(lengths)];assert edges[-1]==npts and (lengths>0).all()
  recovered=np.array(list(zip(edges[:-1],edges[1:]-1)))
  raw=read(f,'raw_scan_indices').astype(int)
  corroboration={}
  if obs==152389:
   lens=[]
   for i in range(12):
    lp=path.parents[1]/'logs'/f'chunk_summary_{i}.log'
    value=int(re.search(r'^-Scan length: (\d+)$',lp.read_text(),re.M).group(1));lens.append(value)
   assert lens==lengths.tolist();assert read(f,'output_scan_index').tolist()==list(range(1,13))
   corroboration={'chunk_summary_lengths':lens,'output_scan_index':list(range(1,13))}
  else:
   raw_lengths=raw[:,1]-raw[:,0]+1
   assert np.all(raw_lengths%2==0) and (raw_lengths//2==lengths).all()
   corroboration={'raw_inner_lengths':raw_lengths.tolist(),'downsample_factor':2}
  base={k:read(f,k) for k in ['TelElAct','alt_phys','az_phys','pointing_offset_alt','pointing_offset_az','apt_x_t','apt_y_t','apt_uid','apt_array','apt_nw','apt_flag']}
  uid=base['apt_uid'];nw=base['apt_nw'];arr=base['apt_array'];apflag=base['apt_flag']
  flags=read(f,'flags');assert np.isin(flags,[0,1]).all()
  required_det=(flags==0).any(axis=0)
  keys=list(zip(nw[required_det].tolist(),uid[required_det].tolist()))
  uidstats={'slots':ndets,'uid_distinct_all':len(np.unique(uid)),'any_flag_zero_slots':int(required_det.sum()),'uid_distinct_among_any_flag_zero':len(np.unique(uid[required_det])),'network_uid_distinct_among_any_flag_zero':len(set(keys)),'nonfinite_uid_required':int(np.count_nonzero(~np.isfinite(uid[required_det]))),'nonfinite_network_required':int(np.count_nonzero(~np.isfinite(nw[required_det]))),'flag_zero_occurrences_where_final_apt_not_zero':int(np.count_nonzero((flags==0)&(apflag!=0)[None,:]))}
  # Metadata-derived guard proposed previously; feasibility only, not adopted.
  bmaj={a:float(read(f,'BMAJ_'+a).item()) for a in ['a1100','a1400','a2000']}
  bmin={a:float(read(f,'BMIN_'+a).item()) for a in bmaj}
  guard=3*max(list(bmaj.values())+list(bmin.values()))
  block_rows=[];maximum_difference=[0.,0.];bad_coord_good_flag=0
  for c,(start,stop) in enumerate(zip(edges[:-1],edges[1:])):
   s=slice(start,stop);e=base['TelElAct'][s,None];xo=base['apt_x_t'][None,:];yo=base['apt_y_t'][None,:]
   x=(np.cos(e)*xo-np.sin(e)*yo+base['pointing_offset_az'][s,None])*np.pi/648000 + base['az_phys'][s,None]
   y=(np.cos(e)*yo+np.sin(e)*xo+base['pointing_offset_alt'][s,None])*np.pi/648000 + base['alt_phys'][s,None]
   if obs==123424:
    for j,(key,pred) in enumerate([('det_lon',x),('det_lat',y)]):
     supplied=np.asarray(f[key][s,:]);assert np.array_equal(np.isfinite(supplied),np.isfinite(pred))
     finite=np.isfinite(pred);maximum_difference[j]=max(maximum_difference[j],float(np.max(np.abs(supplied[finite]-pred[finite]))))
   good=flags[s,:]==0;finite=np.isfinite(x)&np.isfinite(y)
   bad_coord_good_flag+=int(np.count_nonzero(good&~finite))
   split=int(lengths[c]//2)
   train=good[:split,:]&finite[:split,:]&((x[:split,:]**2+y[:split,:]**2)>(guard*np.pi/648000)**2)
   evaluation=good[split:,:]&finite[split:,:]
   ntrain=np.sum(train,axis=0);needed=evaluation.any(axis=0)
   central=(evaluation&((x[split:,:]**2+y[split:,:]**2)<=(guard*np.pi/648000)**2)).any(axis=0)
   per_array=[]
   for a in [0,1,2]:
    req=needed&(arr==a);fail=req&(ntrain<64)
    per_array.append(dict(array=a,required_groups=int(req.sum()),training_count_min=int(ntrain[req].min()) if req.any() else None,groups_below_64=int(fail.sum()),zero_training_groups=int(np.sum(req&(ntrain==0))),groups_below_64_with_central_evaluation=int(np.sum(fail&central))))
   block_rows.append(dict(segment=c,stored_start=int(start),stored_stop_exclusive=int(stop),samples=int(lengths[c]),first_half_count=split,arrays=per_array))
  assert max(maximum_difference)<1e-15
  t=read(f,'TelTime');delta=np.diff(t)
  result['observations'].append(dict(obsnum=obs,path=str(path),sha256=digest(path),bytes=path.stat().st_size,reported_scan_indices=scans.tolist(),recovered_scan_indices=recovered.tolist(),recovered_lengths=lengths.tolist(),corroboration=corroboration,rows_missed_by_reported_intervals=int(npts-np.sum(scans[:,1]-scans[:,0]+1)),time=dict(first=float(t[0]),last=float(t[-1]),strictly_increasing=bool(np.all(delta>0)),median_step_seconds=float(np.median(delta)),stored_SAMPRATE=float(read(f,'SAMPRATE').item()),note='TelTime unit label rad is malformed; SAMPRATE records acquisition rate, not automatically output cadence'),identity=uidstats,coordinate=dict(formula='reported-source altaz detector rotation plus stored telescope coordinates and pointing offsets',full_export_max_abs_difference_rad=maximum_difference if obs==123424 else None,flag_zero_nonfinite_coordinates=bad_coord_good_flag),flags=dict(values=[0,1],zero_count=int(np.sum(flags==0)),one_count=int(np.sum(flags==1))),nominal_beam_headers=dict(BMAJ=bmaj,BMIN=bmin),proposed_guard_arcsec=guard,training_geometry_feasibility=block_rows))
(D/'discovery_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
for r in result['observations']:
 print(json.dumps({k:r[k] for k in ['obsnum','recovered_lengths','rows_missed_by_reported_intervals','identity','coordinate','proposed_guard_arcsec']}))
 print(json.dumps({'obsnum':r['obsnum'],'training_groups_below_64':sum(a['groups_below_64'] for b in r['training_geometry_feasibility'] for a in b['arrays'])}))
