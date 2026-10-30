"""Read saved POINT images/identities only; no reduction or source fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from netCDF4 import Dataset
import matplotlib.pyplot as plt

PROJECT = Path('/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027')
ARRAYS = ('a1100', 'a1400', 'a2000')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text())


def read_map(path):
    with fits.open(path, memmap=False) as hdus:
        h = hdus[0].header
        z = hdus[0].data.astype(float)
        valid = hdus['SCI_SUPPORT'].data.astype(bool) & np.isfinite(z)
        yi, xi = np.indices(z.shape)
        x = (xi+1-h['CRPIX1'])*h['CDELT1']+h['CRVAL1']
        y = (yi+1-h['CRPIX2'])*h['CDELT2']+h['CRVAL2']
        assert h['BUNIT'] == 'mJy/beam' and h['CUNIT1'] == h['CUNIT2'] == 'arcsec'
        assert h['CTYPE1'] == 'AZOFFSET' and h['CTYPE2'] == 'ELOFFSET'
        return z, valid, x, y, abs(h['CDELT1']), abs(h['CDELT2']), h['NOMBFWHM']


def circle_fraction(x, y, dx, dy, cx, cy, radius):
    # 8x8 quadrature of each pixel footprint; no map resampling or new fits.
    weight = np.zeros(x.shape)
    for ox in (np.arange(8)+.5)/8-.5:
        for oy in (np.arange(8)+.5)/8-.5:
            weight += (x+ox*dx-cx)**2+(y+oy*dy-cy)**2 <= radius**2
    return weight/64


def measurements(path, parameters, center):
    z, valid, x, y, dx, dy, nominal = read_map(path)
    distance = np.hypot(x-center[0], y-center[1])
    sky = valid & (distance >= 45) & (distance <= 75)
    assert sky.sum() > 0
    background = float(np.median(z[sky]))
    p = np.asarray(parameters)
    area = float(2*np.pi*p[0]*p[3]*p[4])
    row = {'map': str(path), 'sha256': digest(path), 'pixel_arcsec': [dx, dy],
           'nominal_beam_fwhm_arcsec': nominal, 'parameters': p.tolist(),
           'fwhm_arcsec': (np.sort(p[3:5])*np.sqrt(8*np.log(2))).tolist(),
           'gaussian_brightness_area': area,
           'gaussian_area_units': 'mJy/nominal-beam * arcsec^2; not calibrated total flux',
           'annulus_median': background, 'apertures': []}
    near = distance < 47
    zv, mv, xv, yv = z[near], valid[near], x[near], y[near]
    for radius in (8, 12, 20, 30, 45):
        weights = circle_fraction(xv, yv, dx, dy, *center, radius)
        area_valid = np.sum(weights[mv])*dx*dy
        row['apertures'].append({'radius_arcsec': radius,
            'valid_fraction_of_circle': float(area_valid/(np.pi*radius**2)),
            'annulus_subtracted_brightness_area': float(np.sum(weights[mv]*(zv[mv]-background))*dx*dy),
            'fit_constant_subtracted_brightness_area': float(np.sum(weights[mv]*(zv[mv]-p[6]))*dx*dy)})
    return row


def telescope_fields(path):
    with Dataset(path) as ds:
        names = ['Header.M1.ZernikeC', 'Header.M1.ModelEnabled', 'Header.M1.ZernikeEnabled',
                 'Header.M2.ZReq', 'Header.Weather.Temperature', 'Header.Radiometer.Tau',
                 'Header.TempSens.TempSens']
        result = {}
        for name in names:
            var = ds[name]
            a = np.asarray(var[:])
            result[name] = {'value': a.tolist(), 'units': getattr(var, 'units', '')}
        for name in ['Header.M1.ActPos', 'Header.M1.CmdPos']:
            a = np.asarray(ds[name][:])
            result[name] = {'elements': int(a.size), 'value_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                            'units': getattr(ds[name], 'units', '')}
        return result


def compare_telescope(a, b):
    differing = []
    with Dataset(a) as da, Dataset(b) as db:
        for k in sorted(set(da.variables) | set(db.variables)):
            if k not in da.variables or k not in db.variables:
                differing.append(k)
                continue
            va, vb = da[k][:], db[k][:]
            if va.shape != vb.shape or not np.ma.allequal(va, vb, fill_value=False):
                differing.append(k)
    return differing


def compare_inputs(old, new):
    oa = read_json(old/'.point-work/inputs/observation.json')
    nb = read_json(new/'.point-work/inputs/observation.json')
    result = []
    for left, right in zip(oa['networks'], nb['networks'], strict=True):
        assert left['network'] == right['network']
        assert digest(left['input']['path']) == left['input']['sha256']
        assert digest(right['input']['path']) == right['input']['sha256']
        a, b = read_json(left['input']['path']), read_json(right['input']['path'])
        keys = ['raw', 'tune', 'manifest', 'filter_plan', 'telescope']
        entry = {'network': left['network'], 'recorded_identity_comparison': {
            k: {'equal_sha256': a[k]['sha256'] == b[k]['sha256'],
                'old_sha256': a[k]['sha256'], 'new_sha256': b[k]['sha256']} for k in keys},
            'detector_requests_equal': a['detectors'] == b['detectors']}
        result.append(entry)
    # Byte-level provenance already identifies raw/tune; do not reread GB of raw data.
    tel_a, tel_b = a['telescope']['path'], b['telescope']['path']
    return {'networks': result, 'telescope_variable_differences': compare_telescope(tel_a, tel_b),
            'telescope': telescope_fields(tel_b),
            'raw_digest_scope': 'compared authenticated saved request digests; raw bytes not rehashed'}


def audit(output, figure):
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for oi, (obs, old_name, old_num) in enumerate([
        (152389, 'pointings-configurable-20261028', 'redu00'),
        (152418, 'pointings-disturbance-152418-20261028', 'redu01')]):
        old = PROJECT/old_name/'reduced'/old_num
        new = PROJECT/f'pointings-weighted-{obs}-20261029/reduced/redu00'
        runs = [read_json(r/'RUN.json') for r in [old, new]]
        for run in runs:
            for name in ('executable', 'received_config'):
                assert digest(run[name]['path']) == run[name]['sha256']
        entry = {'inputs': compare_inputs(old, new), 'arrays': {},
                 'run_identities': [{'path': str(r/'RUN.json'), 'sha256': digest(r/'RUN.json'),
                    'executable': run['executable'], 'config': run['received_config'],
                    'pixel_size_arcsec': run['plan']['pixel_size_arcsec'],
                    'ranks': run['plan']['ranks'], 'coverage_cut': run['plan']['coverage_cut'],
                    'cleaning_method': run['plan']['cleaning_method']} for r, run in zip([old,new],runs)]}
        cycles = read_json(new/'weighted-point.json')['cycles']
        for array in ARRAYS:
            fp = new/str(obs)/'raw'/f'point-fit-{array}.json'
            final_fit = read_json(fp)
            op = old/str(obs)/'raw'/f'point-fit-{array}.json'
            old_fit = read_json(op)
            assert digest(fp.parent/final_fit['map']) == final_fit['map_sha256']
            assert digest(op.parent/old_fit['map']) == old_fit['map_sha256']
            center = final_fit['guard']['accepted_parameters'][1:3]
            products = {
                'uniform': (op.parent/old_fit['map'], old_fit['guard']['accepted_parameters']),
                'weighted_M0': (new/f'.point-work/weighted-00/products/{array}/map.fits',
                                cycles[0]['source_selection'][array]['guard']['accepted_parameters']),
                'weighted_M10': (fp.parent/final_fit['map'], final_fit['guard']['accepted_parameters'])}
            vals = {label: measurements(path, par, center) for label, (path, par) in products.items()}
            vals['comparisons'] = {
                'peak_ratio_final_uniform': vals['weighted_M10']['parameters'][0]/vals['uniform']['parameters'][0],
                'gaussian_area_ratio_final_uniform': vals['weighted_M10']['gaussian_brightness_area']/vals['uniform']['gaussian_brightness_area'],
                'aperture20_ratio_final_uniform': vals['weighted_M10']['apertures'][2]['annulus_subtracted_brightness_area']/vals['uniform']['apertures'][2]['annulus_subtracted_brightness_area']}
            entry['arrays'][array] = vals
            if array != 'a1100':
                continue
            for label, (path, par) in products.items():
                z, valid, x, y, _, _, _ = read_map(path)
                rr = np.hypot(x-center[0], y-center[1])
                radii = np.arange(1., 30., 2.)
                profile = [float(np.mean(z[valid & (rr >= r-1) & (rr < r+1)]))-vals[label]['annulus_median']
                           for r in radii]
                axes[oi, 0].plot(radii, profile, 'o-', label=label.replace('_', ' '), markersize=3)
                ap = vals[label]['apertures']
                axes[oi, 1].plot([v['radius_arcsec'] for v in ap],
                    [v['annulus_subtracted_brightness_area'] for v in ap], 'o-', label=label.replace('_',' '))
            sampling = '1″ → 2″' if obs == 152389 else '2″ → 2″'
            axes[oi, 0].set(title=f'{obs} a1100: radial means ({sampling})', xlabel='Radius (arcsec)', ylabel='mJy / nominal beam')
            axes[oi, 1].set(title=f'{obs} a1100: aperture brightness-area', xlabel='Radius (arcsec)', ylabel='mJy / nominal beam × arcsec²')
            for ax in axes[oi]:
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
        results[str(obs)] = entry
    fig.suptitle('Saved POINT audit — uniform versus weighted maps; no new fits')
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    legacy = {}
    for obs in [152389, 152391, 152393, 152418]:
        path = PROJECT/f'reference/v2-science/pointings/reduced/redu00/{obs}/raw/ppt_commissioning_pointing_{obs}_citlali.ecsv'
        table = Table.read(path)
        legacy[str(obs)] = {'path': str(path), 'sha256': digest(path),
            'date': table.meta['date'], 'mjd': table.meta['mjd'], 'elevation_deg': table.meta['MEAN_EL'],
            'M2_ZReq_mm': table.meta['Header.M2.ZReq'],
            'M1_first_Zernike_only': table.meta['Header.M1.ZernikeC'],
            'arrays': {ARRAYS[int(row['array'])]: {k:float(row[k]) for k in
                       ['amp','amp_err','a_fwhm','b_fwhm','x_t','y_t']} for row in table}}
    out = {'scope': 'saved image arithmetic and metadata; no fits or reductions',
           'aperture_background_annulus_arcsec': [45,75], 'aperture_center': 'fixed final weighted centroid',
           'aperture_area_integration': '8x8 pixel-footprint quadrature; no support extrapolation',
           'observations': results, 'archived_PPT': legacy}
    output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    audit(args.output, args.figure)
