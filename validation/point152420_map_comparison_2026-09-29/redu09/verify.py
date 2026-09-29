"""Verify immutable comparison bindings and actual inherited footprint rejection."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import compare as c
from point_additive import linear_weights
r=json.loads((HERE/'results/metrics.json').read_text())
for b in r['inputs']+[r['script'],r['helper'],r['solver']]:assert c.bound(Path(b['path']))['sha256']==b['sha256']
d=c.load(Path(r['inputs'][0]['path']),True);h=dict(d['header']);assert abs(h['CDELT1'])==1 and abs(h['CDELT2'])==1
h.update(CRPIX1=41,CRPIX2=41,CDELT1=h['CDELT1']/3600,CDELT2=h['CDELT2']/3600)
try:linear_weights(h,(81,81),r['fits']['redu09_native']['parameters'])
except ValueError as e:assert str(e)=='fixed-template footprint exceeds source stamp';cause=str(e)
else:raise AssertionError('expected existing boundary rejection absent')
assert all(f['converged'] and not any(f['active_mask']) for f in r['fits'].values())
q=dict(source_map_hash_unchanged=True,all_bound_evidence_hashes_match=True,all_fits_converged_without_active_bounds=True,actual_unchanged_linear_weights_rejects_redu09=True,rejection=cause,metrics=c.bound(HERE/'results/metrics.json'),figure=c.bound(HERE/'results/contrast.png'),producer_checksum_limitation='No historical SHA256 binding in redu09 index and no internal FITS checksums. Exact present-day input SHA256 was recorded before fitting and checked afterward; historical content authenticity is not independently established.')
(HERE/'results/VERIFICATION.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
