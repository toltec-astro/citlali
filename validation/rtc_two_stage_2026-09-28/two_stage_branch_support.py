from pathlib import Path
import json,numpy as np
from astropy.io import fits
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-two-stage-20260928');O=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928');rows=[]
def read(p):return json.loads(p.read_text())
def image(run,k):
 p=run/'M0/products/a1100/map.fits' if k==0 else run/f'run/BASE-POINT/REF/a1100/cycle-{k:02d}/native/products/total.fits'
 with fits.open(p) as h:return h['NORMALIZATION'].data.copy(),h['SCI_SUPPORT'].data.astype(bool)
def describe(run,k):
 q,s=image(run,k);positive=np.sort(q[q>0]);n=len(positive);reference=float(positive[((3*n)//4+n)//2]);cut=read(run/'inputs/map-request.json')['coverage_cut'];threshold=reference*cut;assert np.array_equal(s,(q>0)&(q>=threshold));return q,s,dict(reference=reference,coverage_cut=cut,science_threshold=threshold,science_pixels=int(s.sum()),normalization_sum=float(q.sum()))
for branch in ('REF','C-core-plus','C-core-minus','C-shoulder'):
 for k in (0,5):
  b=R/'B-REF-rep1' if branch=='REF' else O/('N14-'+branch);qrun=R/('Q-'+branch);bn,bs,bd=describe(b,k);qn,qs,qd=describe(qrun,k);rn,rs,rd=describe(R/'Q-REF',k);gain=qs&~rs;loss=rs&~qs
  rows.append(dict(branch=branch,cycle=k,B=bd,Q=qd,Q_REF=rd,B_domain_lost_pixels=int((bs&~qs).sum()),B_domain_gained_pixels=int((qs&~bs).sum()),B_domain_retained_fraction=float((bs&qs).sum()/bs.sum()),Q_REF_lost_pixels=int(loss.sum()),Q_REF_gained_pixels=int(gain.sum()),Q_REF_gained_with_unchanged_normalization=int((gain&(qn==rn)).sum()),Q_REF_gained_normalizations=np.unique(qn[gain]).tolist(),Q_REF_normalization_lost=float(np.maximum(rn-qn,0).sum()),Q_REF_normalization_gained=float(np.maximum(qn-rn,0).sum()),explanation='Existing sorted-positive-normalization reference and fixed coverage cut reproduced exactly. Science support can gain threshold-edge pixels without any new native exposure when the reference normalization falls. No exposure-recovery policy applied.'))
(R/'report/branch-support-domains.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
