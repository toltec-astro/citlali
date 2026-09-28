"""Trace the old T5 excerpts through their authenticated actual native MAP operator."""
from retained_population import *
import gzip
old=W/'validation/rtc_step_consequences_2026-09-28';oldroot=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928/N14-REF-rep2')
sealed=json.loads(gzip.decompress((old/'NATIVE_REMOVALS.json.gz').read_bytes()))['files'];hashes={q['path'][len(str(oldroot))+1:]:q['sha256'] for q in sealed if q['path'].startswith(str(oldroot)+'/')}
actual='M0/products/a1100/operator-pixel.i32';original='run/BASE-POINT/REF/a1100/cycle-05/native/products/operator-pixel.i32'
require(digest(B/actual)==hashes[original],'old T5 operator differs; cannot trace old excerpt through new M0')
windows=read(old/'compact/report/N14-REF-rep2/windows.json');packed=np.load(old/'compact/report/N14-REF-rep2/windows.npz');maps={};records=[]
with fits.open(old/'compact/N14-REF-rep2/run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits.gz') as h:hits=h['HITS'].data.copy();support=h['SCI_SUPPORT'].data.astype(bool);header=dict(h[0].header)
from point_empirical import coordinates
x,yy=coordinates(hits.shape,header);disk=np.hypot(x,yy)<=40
for nw in (1,2,3,4):
 base,r,pr,stages,pix,off,distance,lon,lat=network(nw)
 for s in y(base/'donor-continuity/ptc/receipt.yaml')['segments']:
  sid=s['scan'];rel=f'upstream/products/network{nw}/donor-continuity/ptc/segment-{sid}-causes.u8';late=f'run/BASE-POINT/REF/a1100/cycle-05/native/products/network-{nw}-segment-{sid}/causes.u8';require(digest(B/rel)==hashes[late],'old T5 retained ordering differs')
 for q in [q for q in windows if q['network']==nw]:
  key=q['key'];lo,hi,j=q['first'],q['past_last'],q['detector'];v=packed[key+'_T5_residual'];pp=pix[lo:hi,j];good=np.isfinite(v)&(pp>=0);num=np.bincount(pp[good],weights=v[good],minlength=hits.size).reshape(hits.shape);count=np.bincount(pp[good],minlength=hits.size).reshape(hits.shape);proj=np.divide(num,hits,out=np.full(hits.shape,np.nan),where=hits>0);maxidx=np.argmax(np.where(good,abs(v),-1));maps[key]=proj
  records.append(dict(key=key,network=nw,detector=j,original_window=q,actual_MAP_contributions=int(good.sum()),source_disk_MAP_contributions=int(count[disk].sum()),science_supported_MAP_contributions=int(count[support].sum()),max_abs_projection_any_mapped_mJy_beam=float(np.nanmax(abs(proj))),peak_pixel=int(pp[maxidx]),peak_pixel_hits=float(hits.ravel()[pp[maxidx]]),peak_pixel_science_supported=bool(support.ravel()[pp[maxidx]]),T5_residual_min=float(np.nanmin(v)) if good.any() else None,T5_residual_max=float(np.nanmax(v)) if good.any() else None,peak_native_row=lo+int(maxidx),peak_distance_arcsec=float(distance[lo+maxidx,j]),max_abs_projection_science_mJy_beam=float(np.nanmax(abs(proj[support]))),max_abs_projection_40arcsec_mJy_beam=float(np.nanmax(abs(proj[disk&support]))),projection_description='sum of saved residual excerpt through actual T5 unit coefficients divided by full T5 normalization; no counterfactual removal/refit; zero outside excerpt is zero selected contribution, not missing sky support'))
save_json(D/'old-window-projections.json',dict(operator_sha256=hashes[original],support_verified_against_old_T5=True,records=records));np.savez_compressed(D/'old-window-projections.npz',**maps)
print('authenticated old T5 operator/support for32 old excerpts')
