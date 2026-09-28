"""Direct saved final map/POINT assessment; no refit or noise relearning."""
from pathlib import Path
import sys,json
import numpy as np
from astropy.io import fits
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_empirical import coordinates,patch,stats
from inspect_point_source import gaussian as model
from point_additive_diagnostics import save_json
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-retained-residuals-20260928');B=W/'validation/rtc_step_consequences_2026-09-28/compact/N14-REF-rep2';D=R/'report';D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
s=read(B/'run/BASE-POINT/REF/a1100/SUMMARY.json');p=np.array(s['histories'][-1]['measurement']['guard']['accepted_parameters'])
with fits.open(B/'run/BASE-POINT/REF/a1100/cycle-05/native/products/total.fits.gz') as h:
 signal=h[0].data.copy();header=dict(h[0].header);support=h['SCI_SUPPORT'].data.astype(bool);hits=h['HITS'].data.copy()
x,y=coordinates(signal.shape,header);residual=signal-model(p,x,y);ep=np.load(B/'run/BASE-POINT/REF/a1100/endpoint/model.npz');bg=ep['background_mask'].astype(bool)&support
cy,cx=int(header['CRPIX2']-1),int(header['CRPIX1']-1);src=patch(signal,cy,cx,20);fitres=patch(residual,cy,cx,20);sv=patch(support,cy,cx,20);scale=ep['source_scale'];valid=ep['source_valid'];disk=patch(np.hypot(x,y)<=40,cy,cx,20);norm=np.where(valid&sv,fitres/scale,np.nan)
rec=dict(scope='unmodified final total map against its accepted saved elliptical Gaussian plus constant; no fresh fit; spatial residuals descriptive, single-map uncertainty not paired-difference significance',measurement=s['histories'][-1]['measurement'],endpoint_uncertainty=s['endpoint_uncertainty'],admission=s['histories'][-1]['admission'],background_raw=stats(signal[bg]),background_fit_residual=stats(residual[bg]),source_disk_fit_residual=stats(fitres[disk&sv]),source_disk_residual_over_local_noise=stats(norm[disk&sv&valid]),background_pixels=int(bg.sum()),science_pixels=int(support.sum()),source_noise_scale=stats(scale[disk&sv&valid]),noise_summary=read(B/'run/BASE-POINT/REF/a1100/endpoint/model.json'))
save_json(D/'direct-map.json',rec);np.savez_compressed(D/'direct-map.npz',signal=signal,residual=residual,support=support,hits=hits,background=bg,source_patch=src,source_fit_residual=fitres,source_scale=scale,source_normalized_residual=norm)
print(json.dumps({k:v for k,v in rec.items() if k not in ('noise_summary','measurement','endpoint_uncertainty','admission')},indent=2))
