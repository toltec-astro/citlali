from pathlib import Path
import json,numpy as np,matplotlib.pyplot as plt
from astropy.io import fits
W=Path('/Users/gwilson/.codex/worktrees/fruit-pipeline-freeze/citlali-refactor');V=W/'validation/fruit_pipeline_freeze_2026-09-26';D=V/'compact';R=W/'validation/point_152430_no_upper_coverage_veto_2026-09-26/compact'
s=json.loads((D/'run/BASE-POINT/REF/a2000/SUMMARY.json').read_text());h=s['histories'];peak=[r['measurement']['fit']['parameters'][0] for r in h];cx=[r['measurement']['fit']['parameters'][1] for r in h];cy=[r['measurement']['fit']['parameters'][2] for r in h]
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
axes[0].plot(range(11),peak,'o-',label='consolidated = retained');axes[0].set(xlabel='Stage (0 = M0; 10 = T10)',ylabel='Fitted peak [mJy / nominal beam]',title='Exact reference reproduction');axes[0].legend(fontsize=8)
rel='run/BASE-POINT/REF/a2000/cycle-10/native/products/total.fits.gz'
with fits.open(D/rel) as f,fits.open(R/rel) as r:
 a=f[0].data;b=r[0].data;head=f[0].header;iy,ix=round(head['CRPIX2']-1),round(head['CRPIX1']-1);z=a[iy-20:iy+21,ix-20:ix+21]
 im=axes[1].imshow(z,origin='lower',extent=(41,-41,-41,41),vmin=-30,vmax=750,cmap='inferno');axes[1].set(xlabel='Tangent east [arcsec]',ylabel='Tangent north [arcsec]',title='Reconstructed T10');fig.colorbar(im,ax=axes[1],label='mJy / nominal beam')
 diff=a-b;v=np.isfinite(diff);axes[2].plot(np.flatnonzero(v),diff[v],lw=.5);axes[2].set(xlabel='Supported pixel index',ylabel='Difference [mJy / nominal beam]',title='T10 minus retained reference',ylim=(-1e-9,1e-9));axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.suptitle('POINT 152430 · a2000 · ordinary MAP + BASE-POINT (one cold start)');fig.savefig(V/'regression.png',dpi=160);plt.close(fig)
rows=['| Stage | Peak (mJy/nominal beam) | East (arcsec) | North (arcsec) | Accepted pixels |','|---|---:|---:|---:|---:|']
for k,r in enumerate(h):rows.append(f"| {'M0' if k==0 else 'T'+str(k)} | {peak[k]:.9f} | {cx[k]:.9f} | {cy[k]:.9f} | {r['admission']['selected_pixels']} |")
u=s['endpoint_uncertainty'];sc={r['name']:r['conditional_sigma'] for r in u};assert all(r['conditional_sigma'] is None or np.isfinite(r['conditional_sigma']) for r in u)
text='''# Cold-start integration regression

**Pass: exact numerical reproduction on first attempt.** Source
`b47c13e1f5b745b94880218e347f77b4142686ff`, clean durable checkout;
`REGRESSION.json` binds the predeclared existing tolerances. M0 and T1–T10 MAP
and candidate planes all match exactly (maximum difference0); deterministic guard,
fit/template/aperture, initial noise/covariance fields and endpoint uncertainty
records also match exactly. No allowance was weakened or second trajectory used.

'''+ '\n'.join(rows)+f'''

T10 conditional peak sigma = {sc['peak']:.9f} mJy/nominal beam; template sigma =
{sc['template']:.9f} mJy/nominal beam; aperture2 sigma = {sc['aperture2']:.9f}
mJy/nominal beam arcsec². These are conditional model-based uncertainties, not a
full adaptive error law. The known shoulder-response limitation remains.

![Exact regression](regression.png)

The pipeline took117.213 s (111.893 s native stages); sampled native peak RSS
1.558 GB. Peak recorded stage storage0.814 GB stayed below3.323 GB ceiling.
One M0, one BASE trajectory, ten applications and one endpoint analysis were used,
including all attempts; the second-attempt allowance remains unused. The endpoint
emits the same retained NumPy matmul warning strings as the reference. Returned
conditional values are finite where identifiable and exactly equal to the prior
records; warnings are preserved in COLD_RUN.txt, not suppressed or treated as a new
scientific discrepancy. Native logs have zero error-level messages.

Isolation: cold-start.sb and ISOLATION.json record OS denial of old derived
operands and prior scratch code/build; original raw/tune/APT/telescope inputs,
declared environment and candidate remain readable. Two setup-only build option
corrections, an initial binary-path preflight typo and a denied-path regex probe
repair preceded science; they consumed no real-data attempts. No network transfer.

48 focused Python tests, 33 CTests, 124 upstream contract tests,207 baseline-tool
tests (137 subtests) and full configuration preflight passed. Relevant projection,
subtract/restore, fresh PTC, replacement, guarded failure, high-coverage and
read-only relocation tests are included. Native CTest was focused, not the entire
unbuilt legacy suite. This is local AppleClang/Homebrew evidence, not Unity/Spack.

Cleanup removed11,113 remaining native numeric files (533.939 MB); per-cycle
removals are separately recorded in compact/temporary-removals.jsonl. All saved
FITS and large YAML were losslessly compressed. The new compact bundle is26.184 MB
and the existing235 MB comparison was referenced in place. Original dependency
hashes remain unchanged. No native restart parent survives; reproduction explicitly
starts from originals. Reporting passes with all local_data access denied.
Source/configuration and compact results are durable Git history, not only scratch.
'''
(V/'REPORT.md').write_text(text)
