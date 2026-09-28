from pathlib import Path
import sys,json,numpy as np
from astropy.io import fits
w=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(w/'tools/timestream_successor'))
from point_empirical import coordinates
r=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-two-stage-20260928');d=r/'report';q=json.loads((d/'new-windows.json').read_text());a=np.load(d/'new-windows.npz')
with fits.open(r/'B-REF-rep1/run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits') as h:hits=h['HITS'].data.copy();support=h['SCI_SUPPORT'].data.astype(bool);x,y=coordinates(hits.shape,h[0].header)
disk=(np.hypot(x,y)<=40)&support;recs=[];maps={}
for row in q:
 k=row['key'];z=a[k+'_T5'];p=a[k+'_map_pixel'];ok=(p>=0)&np.isfinite(z);num=np.bincount(p[ok],weights=z[ok],minlength=hits.size).reshape(hits.shape);proj=np.divide(num,hits,out=np.full(hits.shape,np.nan),where=hits>0);count=np.bincount(p[ok],minlength=hits.size).reshape(hits.shape);i=np.argmax(np.where(ok,abs(z),-1));recs.append(dict(key=k,detector=row['detector'],network=row['network'],MAP=int(ok.sum()),science=int(count[support].sum()),fit_disk=int(count[disk].sum()),max_abs_science=float(np.nanmax(abs(proj[support]))),max_abs_fit_disk=float(np.nanmax(abs(proj[disk]))),distance_range=[float(a[k+'_distance_arcsec'].min()),float(a[k+'_distance_arcsec'].max())],peak_row=int(a[k+'_row'][i]),peak_value=float(z[i]),peak_distance=float(a[k+'_distance_arcsec'][i]),peak_science=bool(support.ravel()[p[i]]) if p[i]>=0 else None,learn_trace=row['learn_trace']))
 maps[k]=proj
(d/'window-projections.json').write_text(json.dumps(dict(records=recs,description='Actual T5 residual excerpts / full B normalization, not isolated bias or counterfactual quarantine'),indent=2)+'\n');np.savez_compressed(d/'window-projections.npz',**maps)
for q in recs:print({k:v for k,v in q.items() if k!='learn_trace'})
