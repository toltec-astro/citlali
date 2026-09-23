"""Fixed-definition POINT trajectory measurements and explicit validity guards."""
from pathlib import Path
import numpy as np
from point_fruit_diagnostics import load, patch_map, replacement, learned_state, compare_zero
from point_empirical import coordinates, aperture_measurements, stats
from point_admission import fit_diagnostic, projection
from point_empirical_io import write_fits
from point_additive_diagnostics import save_json
from inspect_forward_map import require, read_yaml, digest


def bound(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def verified(ref):
    path = Path(ref['path'])
    require(digest(path) == ref['sha256'], 'changed bound artifact: ' + str(path))
    return path


def verify_manifest(root, expected=None):
    root = Path(root).resolve()
    manifest = root / 'FILES_SHA256SUMS'
    if expected:
        require(digest(manifest) == expected, 'foreign manifest')
    count = 0
    for line in manifest.read_text().splitlines():
        sha, name = line.split('  ', 1)
        p = (root / name).resolve()
        require(p.is_relative_to(root) and digest(p) == sha, 'manifest mismatch: ' + name)
        count += 1
    return count


def manifest(root):
    p = root / 'FILES_SHA256SUMS'
    p.write_text(''.join(f'{digest(f)}  {f.relative_to(root)}\n'
                         for f in sorted(root.rglob('*')) if f.is_file() and f != p))


class FixedMeasurements:
    """All spatial comparisons stay bound to M0, including source/background regions."""
    def __init__(self, maps, admission, additive):
        self.maps = Path(maps)
        self.original, self.support, self.coverage, self.header = patch_map(self.maps/'map.fits')
        self.beam = read_yaml(self.maps/'bundle.yaml')['nominal_beam_fwhm_arcsec']
        self.original_fit = fit_diagnostic(self.original, self.support, self.header, self.beam)
        require(self.original_fit['status'] == 'success', 'original fit unavailable')
        self.parameters = np.asarray(self.original_fit['parameters'])
        self.x, self.y = coordinates(self.original.shape, self.header)
        self.disk = np.hypot(self.x, self.y) <= 40
        self.radius = np.hypot(self.x-self.parameters[1], self.y-self.parameters[2])
        self.major = max(self.parameters[3:5])*np.sqrt(8*np.log(2))
        self.background = self.disk & (self.radius > 3*self.major)
        self.edges = np.arange(0, 42, 2)
        self.weights = np.load(Path(additive)/'model.npz')['weights']
        relative, _ = load(Path(admission)/'source-relative.fits')
        self.scale = relative['PIXEL_SCALE']
        self.noise_valid = relative['NOISE_VALID'].astype(bool)

    def measure(self, signal, support, candidate=False):
        fit = fit_diagnostic(signal, support, self.header, self.beam)
        good = fit['status'] == 'success'
        bg = fit['parameters'][6] if good and not candidate else 0.
        rel = signal-bg
        profile = [float(np.mean(signal[m])) if np.any(m) else None for m in
                   [support & (self.radius >= lo) & (self.radius < hi)
                    for lo, hi in zip(self.edges[:-1], self.edges[1:])]]
        return dict(fit=fit, frozen_template_projection=projection(rel, support, self.weights),
                    apertures=aperture_measurements(rel, support, self.x, self.y,
                                                   self.parameters[1:3], self.major, self.beam, 4.),
                    background=stats(signal[support & self.background]), profile=profile)

    def infer(self, signal, support):
        return replacement(signal, support, self.scale, self.noise_valid, self.header, self.beam)


def candidate_guard(candidate, header):
    from scipy.ndimage import binary_erosion, binary_dilation
    s = candidate['ADMITTED'].astype(bool)
    v = candidate['VALID'].astype(bool)
    b = candidate['CANDIDATE']
    x, y = coordinates(b.shape, header)
    domain = np.hypot(x, y) <= 40
    boundary = domain & ~binary_erosion(domain, structure=np.ones((3, 3)))
    require(s.any(), 'empty candidate')
    require(not np.any(s & (~v | ~domain | ~np.isfinite(b))), 'unsupported admitted feedback')
    require(not np.any(s & boundary), 'candidate boundary contact')
    require(not np.any(binary_dilation(s, structure=np.ones((3, 3))) & domain & ~v), 'candidate unsupported contact')
    require(np.all(np.isfinite(b[v])) and np.all(b[v & ~s] == 0) and np.all(b[s] > 0),
            'candidate brightness/intentional-zero semantics')
    require(not np.any(v & ~domain) and np.all(np.isnan(b[~v])), 'candidate unavailable semantics')


def processing_guard(original, products, receipt):
    old, oh = load(original)
    new, nh = load(products/'total.fits')
    for key in ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CDELT1', 'CDELT2', 'RADESYS', 'EQUINOX']:
        require(oh[key] == nh[key], 'WCS changed: '+key)
    for name in ['SCI_SUPPORT', 'NORM_SUPPORT', 'NORMALIZATION', 'HITS']:
        require(np.array_equal(old[name], new[name]), 'coverage/normalization changed: '+name)
    support = new['SCI_SUPPORT'].astype(bool)
    require(np.isfinite(new['PRIMARY'][support]).all(), 'nonfinite supported total')
    residual, _ = load(products/'residual.fits')
    restored, _ = load(products/'restored.fits')
    for part in [residual, restored]:
        for name in ['SCI_SUPPORT', 'NORM_SUPPORT', 'NORMALIZATION', 'HITS']:
            require(np.array_equal(new[name], part[name]), 'decomposition operator support differs')
    err = float(np.max(np.abs((new['PRIMARY']-residual['PRIMARY']-restored['PRIMARY'])[support])))
    require(np.isfinite(err) and err < 1e-8, 'broken subtraction/restoration accounting')
    for s in receipt['segments']:
        parent = s['source_segment']
        require(s['input_eligibility_changed'] == s['output_cause_changed'] == 0, 'eligibility changed')
        require(s['converged'] == parent['converged'], 'PTC convergence state changed')
        if not s['converged']:
            require(parent['network'] == 3 and parent['scan'] == 5 and
                    s['stopping_reason'] == parent['stopping_reason'], 'new PTC failure')
        require(s['rank'] == 10 and s['projection_support_counts'][0] == 0, 'rank or projected support changed')
        require(s['application_failed_times'] == parent.get('application_failed_times', 0), 'PTC application failure')
    return dict(pass_guard=True, decomposition_max_abs=err, coverage_normalization_eligibility_exact=True,
                inherited_failures=[dict(network=s['source_segment']['network'], scan=s['source_segment']['scan'],
                                         reason=s['stopping_reason']) for s in receipt['segments'] if not s['converged']])


def difference(new, old, domain, amplitude):
    d = (new-old)[domain]
    require(d.size > 0 and np.isfinite(d).all(), 'difference domain unavailable')
    rms = float(np.sqrt(np.mean(d*d)))
    return dict(pixels=int(d.size), mean=float(d.mean()), rms=rms, max_abs=float(abs(d).max()),
                rms_over_M0_fitted_amplitude=rms/amplitude)


def selection_difference(new, old):
    s, p = new['ADMITTED'].astype(bool), old['ADMITTED'].astype(bool)
    kept = s & p
    d = (new['CANDIDATE']-old['CANDIDATE'])[kept]
    return dict(added=int((s & ~p).sum()), removed=int((p & ~s).sum()), retained=int(kept.sum()),
                retained_brightness_mean_change=float(d.mean()) if d.size else None,
                retained_brightness_rms_change=float(np.sqrt(np.mean(d*d))) if d.size else None,
                retained_brightness_max_abs_change=float(abs(d).max()) if d.size else None)


def diagnose_cycle(array, index, fixed, products, out, previous, histories):
    out.mkdir(exist_ok=False)
    signal, support, _, header = patch_map(products/'total.fits')
    receipt = read_yaml(products/'receipt.yaml')
    fit, cand = fixed.infer(signal, support)
    measurement = fixed.measure(signal, support)
    report = dict(array=array, cycle_index=index, measurement=measurement,
                  receipt=bound(products/'receipt.yaml'), total=bound(products/'total.fits'),
                  previous_state=bound(previous), candidate_status=None, guards=None, differences={})
    if cand is not None:
        path = out/f'm{index}.fits'
        write_fits(path, header, [('CANDIDATE', cand['candidate'], 'mJy/nominal beam'),
                   ('ADMITTED', cand['selected'].astype('u1'), '1'), ('VALID', cand['valid'].astype('u1'), '1'),
                   ('RATIO', cand['z'], '1')], companion='FRUIT-UNAPPLIED')
        report.update(candidate=bound(path), candidate_status=cand['summary'],
                      candidate_measurement=fixed.measure(cand['candidate'], cand['valid'], True))
    save_json(out/'measurements.json', report)  # Preserve trigger measurements before any guard raises.
    try:
        report['guards'] = processing_guard(fixed.maps/'map.fits', products, receipt)
        require(fit['status'] == 'success' and not fit.get('constrained', False) and cand is not None,
                'unusable or constrained total fit')
        require(cand['summary']['status'] == 'retained', 'unusable replacement candidate')
        candidate, _ = load(out/f'm{index}.fits')
        candidate_guard(candidate, header)
        report['PTC_segments'] = learned_state(products, receipt)
        restore, _, _, _ = patch_map(products/'restored.fits')
        applied, _ = load(verified(__import__('json').loads(previous.read_text())['candidate']))
        valid = applied['VALID'].astype(bool) & support
        err = float(np.max(abs(restore[valid]-applied['CANDIDATE'][valid])))
        require(err < 1e-8, 'projected/remapped operand differs on actual support')
        report['guards']['remapped_operand_max_abs'] = err
        full, full_header = load(products/'total.fits')
        planes = []
        for lag in (1, 2):
            old = histories[index-lag]
            oldfull, _ = load(verified(old['total']))
            oldpatch, _, _, _ = patch_map(Path(old['total']['path']))
            oldcandidate, _ = load(verified(old['candidate']))
            af = measurement['fit']['parameters'][0]; ao = old['measurement']['fit']['parameters'][0]
            report['differences'][str(lag)] = dict(from_cycle=index-lag, fitted_amplitude=af-ao,
                fitted_amplitude_fraction=(af-ao)/ao,
                full_map=difference(full['PRIMARY'], oldfull['PRIMARY'], full['SCI_SUPPORT'].astype(bool), fixed.parameters[0]),
                fixed_disk=difference(signal, oldpatch, fixed.disk, fixed.parameters[0]),
                fixed_disk_background_subtracted=difference(signal-fit['parameters'][6], oldpatch-old['measurement']['fit']['parameters'][6], fixed.disk, fixed.parameters[0]),
                selection=selection_difference(candidate, oldcandidate))
            planes.append((f'DELTA_LAG{lag}', full['PRIMARY']-oldfull['PRIMARY'], 'mJy/nominal beam'))
        write_fits(out/'differences.fits', full_header, planes, companion='FRUIT-TRAJECTORY')
        report['ready'] = True
    except (ValueError, KeyError, OSError) as e:
        report.update(ready=False, stop_reason=str(e))
    save_json(out/'measurements.json', report)
    return report
