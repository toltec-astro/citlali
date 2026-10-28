"""Real FITS publication and POINT angular-domain regression checks."""
import json

import numpy as np
import pytest
import yaml
from astropy.io import fits

import configured_point_products as products
import point_reference
from inspect_forward_map import digest


def native_map(root, array='a2000', *, step=2., observation=152389, size=None):
    """Native six-plane source-offset fixture, with known physical centroid."""
    size = size or int(240 / step) + 1
    center = (size - 1) / 2
    yy, xx = np.indices((size, size))
    x, y = (xx - center) * -step, (yy - center) * step
    signal = (3. + 1000. * np.exp(-((x - 2.) ** 2 / 32. + (y + 3.) ** 2 / 35.28))
              + np.sin(.7 * x) * np.cos(.37 * y))
    support = np.hypot(x, y) < 110.
    normalization = 114. + np.arange(x.size).reshape(x.shape) % 343
    normalization = np.where(support, normalization, 0.)
    numerator = signal * normalization
    signal = np.where(support, signal, np.nan)
    beam = dict(a1100=5., a1400=7.5, a2000=10.)[array]
    header = fits.Header(dict(**products.FRAME, CRPIX1=center + 1,
                              CRPIX2=center + 1, CRVAL1=0., CRVAL2=0.,
                              CDELT1=-step, CDELT2=step, NOMBFWHM=beam,
                              ARRAY=array))
    values = (signal, numerator, normalization, normalization.astype('int64'),
              support.astype('uint8'), support.astype('uint8'))
    hdus = []
    for i, (name, data, unit) in enumerate(zip(products.PLANES, values, products.UNITS)):
        h = header.copy()
        h['BUNIT'] = unit
        hdus.append(fits.PrimaryHDU(data, h) if i == 0 else fits.ImageHDU(data, h, name=name))
    root.mkdir(parents=True)
    path = root / 'map.fits'
    fits.HDUList(hdus).writeto(path, checksum=True)
    bundle = dict(schema='citlali-forward-map-bundle-v1',
                  role='SCI-MAP:base_observation_map@1', outcome='success',
                  observation=f'obs:{observation}_0_2', array=array,
                  signal_unit='mJy/beam', measured_beam_or_full_response_inferred=False,
                  nominal_beam_fwhm_arcsec=beam,
                  product_identity=f'synthetic-native:{digest(path)}',
                  WCS=dict(header), files={'map.fits': digest(path)})
    (root / 'bundle.yaml').write_text(yaml.safe_dump(bundle))
    return path


def refresh_bundle(path):
    bundle_path = path.parent / 'bundle.yaml'
    bundle = yaml.safe_load(bundle_path.read_text())
    bundle['files']['map.fits'] = digest(path)
    bundle_path.write_text(yaml.safe_dump(bundle))


def test_all_array_publication_preserves_native_bytes_and_avoids_ppt(tmp_path, monkeypatch):
    inputs = tmp_path / 'native'
    sources = {array: native_map(inputs / array, array) for array in products.ARRAYS}
    monkeypatch.setattr(point_reference, 'measure_forward',
                        lambda *a, **kw: pytest.fail('fit requested while disabled'))
    output = tmp_path / 'redu00/152389/raw'
    records = products.publish_maps(inputs, output, 152389, products.ARRAYS,
                                    fit_diagnostics=False, verbose=False)
    assert len(records) == 3
    for row in records:
        matches = list(output.glob(f'toltec_*_{row["array"]}*.fits'))
        assert matches == [output / row['map']]
        assert matches[0].read_bytes() == sources[row['array']].read_bytes()
        assert row['fit_status'] == 'not_requested'
        assert row['calibration_eligible'] is False
        assert row['legacy_analysis_compatible'] is False
    assert not list(output.rglob('ppt_*.ecsv'))
    assert not list(output.glob('point-fit-*'))
    with pytest.raises(ValueError, match='fresh'):
        products.publish_maps(inputs, output, 152389, products.ARRAYS,
                              fit_diagnostics=False, verbose=False)


def test_missing_array_and_wrong_observation_fail_before_publication(tmp_path):
    inputs = tmp_path / 'native'
    native_map(inputs / 'a1100', 'a1100')
    output = tmp_path / 'raw'
    with pytest.raises(FileNotFoundError):
        products.publish_maps(inputs, output, 152389, products.ARRAYS,
                              fit_diagnostics=False, verbose=False)
    assert not output.exists()
    with pytest.raises(ValueError, match='observation differs'):
        products.publish_maps(inputs, output, 152390, ('a1100',),
                              fit_diagnostics=False, verbose=False)
    assert not output.exists()


@pytest.mark.parametrize('identity', ['obs:152389_1_2', 'obs:152389_0_1'])
def test_wrong_subobservation_or_scan_is_not_published(tmp_path, identity):
    inputs = tmp_path / 'native'
    path = native_map(inputs / 'a2000')
    bundle_path = path.parent / 'bundle.yaml'
    bundle = yaml.safe_load(bundle_path.read_text())
    bundle['observation'] = identity
    bundle_path.write_text(yaml.safe_dump(bundle))
    with pytest.raises(ValueError, match='subobservation/scan differs'):
        products.publish_maps(inputs, tmp_path / 'raw', 152389, ('a2000',),
                              fit_diagnostics=False, verbose=False)
    assert not (tmp_path / 'raw').exists()


@pytest.mark.parametrize('defect', ['units', 'frame', 'missing', 'checksum', 'digest'])
def test_malformed_native_product_is_not_published(tmp_path, defect):
    inputs = tmp_path / 'native'
    path = native_map(inputs / 'a2000')
    if defect == 'digest':
        path.write_bytes(path.read_bytes() + b'foreign')
    else:
        with fits.open(path, mode='update', checksum='remove' if defect == 'checksum' else False) as hdus:
            if defect == 'units':
                hdus[0].header['BUNIT'] = 'MJy/sr'
            elif defect == 'frame':
                hdus[1].header['CTYPE1'] = 'RA---TAN'
            elif defect == 'missing':
                hdus[0].data[0, 0] = 0.
        refresh_bundle(path)
    with pytest.raises(ValueError):
        products.publish_maps(inputs, tmp_path / 'raw', 152389, ('a2000',),
                              fit_diagnostics=False, verbose=False)
    assert not (tmp_path / 'raw').exists()


@pytest.mark.parametrize('step, expected_half', [(1., 40), (2., 20)])
def test_forward_fit_uses_same_40_arcsec_domain_and_covariance_crop(tmp_path, monkeypatch,
                                                                 step, expected_half):
    path = native_map(tmp_path / 'native/a2000', step=step)
    calls = []
    original = point_reference.learn

    def learn(*args, **kwargs):
        calls.append(kwargs['stamp_size'])
        return original(*args, **kwargs)

    monkeypatch.setattr(point_reference, 'learn', learn)
    monkeypatch.setattr(point_reference, 'Neighborhood',
                        lambda *a, **kw: pytest.fail('forward fit constructed feedback context'))
    result = point_reference.measure_forward(path)
    assert result['status'] == 'accepted'
    assert result['optimizer_calls'] == 1
    assert calls == [max(49, 2 * expected_half + 1)]
    y, x = np.indices((2 * expected_half + 1,) * 2)
    expected_pixels = np.sum(np.hypot((x - expected_half) * step,
                                      (y - expected_half) * step) <= 40.)
    assert result['fit_pixels'] == expected_pixels
    assert result['accepted_centroid_tangent_arcsec'] == pytest.approx([2., -3.], abs=.01)
    assert result['sigma_A'] > 0.
    assert result['pointing_correction'] is None
    assert 'expected_origin_sky' not in result
    assert not list(tmp_path.rglob('*.npz'))


def test_requested_fit_can_report_unavailable_without_calibration_claim(tmp_path):
    inputs = tmp_path / 'native'
    native_map(inputs / 'a2000', size=31)
    output = tmp_path / 'raw'
    row, = products.publish_maps(inputs, output, 152389, ('a2000',),
                                 fit_diagnostics=True, verbose=False)
    fit = json.loads((output / row['fit']).read_text())
    assert row['fit_status'] == fit['guard']['status'] == 'unavailable'
    assert 'truncated' in fit['guard']['cause']
    assert fit['guard']['accepted_parameters'] is None
    assert fit['guard']['sigma_A'] is None
    assert fit['pointing_correction'] is None and not fit['calibration_eligible']
    assert not list(output.glob('ppt_*'))
    assert not list(output.glob('*details'))


def test_fit_computation_bug_propagates(tmp_path, monkeypatch):
    inputs = tmp_path / 'native'
    native_map(inputs / 'a2000')

    def fail(*args, **kwargs):
        raise RuntimeError('unexpected solver defect')

    monkeypatch.setattr(point_reference, 'measure_forward', fail)
    with pytest.raises(RuntimeError, match='unexpected solver defect'):
        products.publish_maps(inputs, tmp_path / 'raw', 152389, ('a2000',),
                              fit_diagnostics=True, verbose=False)


def test_verbose_retains_existing_covariance_details_without_changing_fit(tmp_path):
    path = native_map(tmp_path / 'native/a2000')
    quiet = point_reference.measure_forward(path)
    details = tmp_path / 'details'
    verbose = point_reference.measure_forward(path, details=details)
    assert quiet == verbose
    assert {p.name for p in details.iterdir()} == {'C0.json', 'C0.npz'}
