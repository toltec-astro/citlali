import json, hashlib
from pathlib import Path
import numpy as np
from astropy.io import fits
r=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930/B-REF')
paths=[r/'M0/products/a1100/map.fits',r/'run/BASE-POINT/REF/a1100/cycle-05/native/products/residual.fits',r/'freeze/a1100/C0.npz']
a,b=[fits.open(p,memmap=False,checksum=True) for p in paths[:2]]
mask=np.load(paths[2])['source_mask']
v=a['SCI_SUPPORT'].data.astype(bool)&b['SCI_SUPPORT'].data.astype(bool)&~mask
assert a[0].data.shape==b[0].data.shape==mask.shape
for k in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2','CTYPE1','CTYPE2'):assert a[0].header[k]==b[0].header[k]
def stats(data):
 x=data[v];assert np.isfinite(x).all();m=float(np.median(x))
 return dict(n=len(x),mean=float(x.mean()),median=m,std=float(x.std(ddof=1)),robust_scale=float(1.482602218505602*np.median(abs(x-m))))
z=dict(domain='B M0 and T5 genuine residual on common science support outside frozen B/M0 four-major-FWHM source mask; descriptive, not independent-pixel significance',M0=stats(a[0].data),T5_residual=stats(b[0].data),bindings=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
import argparse
parser=argparse.ArgumentParser(description='Read-only saved-map background comparison; no fits or reduction')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
p=args.output;p.write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
