"""Read-only bindings and result consistency checks; does not refit maps."""
from pathlib import Path
import hashlib,json,math
import yaml
from astropy.io import fits
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'results/metrics.json').read_text())
def binding(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for q in r['inputs']+[r['script'],r['solver']]:assert binding(Path(q['path']))['sha256']==q['sha256']
old,new,guard=[Path(q['path']) for q in r['inputs']]
index=old.parent/'index.yaml';bundle=new.parent/'bundle.yaml'
i=yaml.load(index.read_text(),Loader=yaml.CSafeLoader);b=yaml.load(bundle.read_text(),Loader=yaml.CSafeLoader)
assert next(q for q in i['products'] if q['name']==old.name)['sha256']==r['inputs'][0]['sha256']
assert b['observation']=='obs:152420_0_2' and b['array']=='a1100' and b['outcome']=='success'
assert b['files'][new.name]==r['inputs'][1]['sha256']
with fits.open(old) as f:
 assert f[0].header['OBSNUM0']=='152420'
 legacy_recipe={k:f[0].header[k] for k in ('CONFIG.CLEANED.NEIG','CONFIG.CLEANED.MODESEL','CONFIG.WEIGHT.TYPE','CONFIG.FRUITLOOPS','CONFIG.DOWNSAMPLED')}
assert r['current_saved_parameter_max_abs_difference']==0
assert r['inputs_unchanged'] and all(r['self_checks'].values())
assert all(f['converged'] and not any(f['active_mask']) for f in r['fits'].values())
a=r['fits']['old_native']['fwhm_major_minor_arcsec'];z=r['fits']['old_bin2']['fwhm_major_minor_arcsec'];n=r['fits']['new_native']['fwhm_major_minor_arcsec']
record=dict(checks_passed=True,metrics=binding(HERE/'results/metrics.json'),figure=binding(HERE/'results/comparison.png'),old_product_index=binding(index),new_map_bundle=binding(bundle),old_product_sha256_matches_index=True,new_product_sha256_matches_bundle=True,source_guard_reproduced_exactly=True,original_maps_unchanged=True,legacy_recipe=legacy_recipe,ratios=dict(new_over_old_major=n[0]/a[0],new_over_binned_old_major=n[0]/z[0],new_over_old_minor=n[1]/a[1],new_over_old_geometric_mean=math.sqrt(n[0]*n[1]/(a[0]*a[1]))),fits_checksum_qualification='Old FITS has no internal CHECKSUM/DATASUM (status2); its recorded SHA256 matches retained product index. New FITS internal checksums all pass (status1).',frame_qualification='Major/minor lengths and radial offsets are compared in arcsec; native orientation differs and no sky registration or pixelwise subtraction is claimed.')
(HERE/'results/VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
