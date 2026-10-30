"""Configured frame/source lifecycle checks using synthetic maps only."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

import configured_point_feedback as feedback
from inspect_forward_map import digest
from test_configured_point_products import native_map, refresh_bundle


def bundle_update(path, **values):
    target = path.parent / 'bundle.yaml'
    bundle = yaml.safe_load(target.read_text())
    bundle.update(values)
    target.write_text(yaml.safe_dump(bundle))


@pytest.mark.parametrize('array', ['a1100', 'a1400', 'a2000'])
@pytest.mark.parametrize('step', [1., 2.])
def test_native_frame_dynamic_domain_and_quiet_selector(tmp_path, monkeypatch, array, step):
    path = native_map(tmp_path / 'map', array, step=step)
    calls = []
    original = feedback.learn

    def learn(*args, **kwargs):
        calls.append((args[5].mask_fwhm, kwargs['amplitude_only'], kwargs['stamp_size']))
        return original(*args, **kwargs)

    monkeypatch.setattr(feedback, 'learn', learn)
    context, record = feedback.initialize(path, tmp_path / 'feedback')
    half = int(40 / step)
    assert record['status'] == 'ready' and context is not None
    assert record['guard']['accepted_centroid_tangent_arcsec'] == pytest.approx([2., -3.], abs=.01)
    assert calls == [(4., False, max(49, 2*half+1)),
                     (3., True, max(49, 2*half+1))]
    assert context.scale.shape == (2*half+1, 2*half+1)
    assert context.scale.flags.writeable is False
    assert context.noise_valid.flags.writeable is False
    with fits.open(record['model_path']) as hdus:
        assert [h.name for h in hdus] == ['CANDIDATE', 'ADMITTED', 'VALID']
        for hdu in hdus:
            assert all(hdu.header[key] == value for key, value in feedback.FRAME.items())
            assert hdu.header['ARRAY'] == array
            assert hdu.header['CDELT1'] == -step and hdu.header['CDELT2'] == step
            assert hdu.verify_checksum() == hdu.verify_datasum() == 1
        admitted = hdus['ADMITTED'].data.astype(bool)
        valid = hdus['VALID'].data.astype(bool)
        model = hdus[0].data
        y, x = np.indices(model.shape)
        disk = np.hypot((x-half)*step, (y-half)*step) <= 40.
        assert not np.any(valid & ~disk)
        assert np.isnan(model[~valid]).all()
        assert np.all(model[valid & ~admitted] == 0.)
        assert np.all(model[admitted] > 0.)
    assert record['model_sha256'] == digest(record['model_path'])
    assert {p.name for p in (tmp_path / 'feedback').iterdir()} == {'feedback.json', 'candidate.fits'}
    assert 'expected_origin_sky' not in record['guard']
    with pytest.raises(ValueError, match='fresh'):
        feedback.initialize(path, tmp_path / 'feedback')


def test_each_update_is_replacement_and_weighted_depth_can_change(tmp_path, monkeypatch):
    original = native_map(tmp_path / 'original')
    current = native_map(tmp_path / 'current')
    bundle_update(original, frozen_health_identity='health:fixed')
    bundle_update(current, frozen_health_identity='health:fixed')
    with fits.open(current, mode='update') as hdus:
        hdus[0].data *= 1.2
        hdus['NORMALIZATION'].data *= 2.3
        hdus['NUMERATOR'].data *= 1.2 * 2.3
        # A lost remote pixel must not falsely fail a frozen-support assertion.
        row, column = 60, 110
        hdus['SCI_SUPPORT'].data[row, column] = 0
        hdus[0].data[row, column] = np.nan
    refresh_bundle(current)
    context, first = feedback.initialize(original, tmp_path / 'M0')
    monkeypatch.setattr(feedback, 'learn', lambda *a, **kw: pytest.fail('selection relearned after M0'))
    second = feedback.update(current, context, tmp_path / 'M1')
    assert second['status'] == 'ready'
    assert second['initial_map_sha256'] == first['map_sha256']
    assert second['frozen_health_identity'] == 'health:fixed'
    assert second['guard']['accepted_parameters'][0] == pytest.approx(
        1.2 * first['guard']['accepted_parameters'][0], rel=1e-9)
    with fits.open(first['model_path']) as a, fits.open(second['model_path']) as b:
        common = a['ADMITTED'].data.astype(bool) & b['ADMITTED'].data.astype(bool)
        assert common.any()
        assert np.allclose(b[0].data[common], 1.2*a[0].data[common], rtol=1e-9, atol=1e-9)
    # Repeating the same total map replaces again and cannot accumulate models.
    third = feedback.update(current, context, tmp_path / 'M2')
    with fits.open(second['model_path']) as a, fits.open(third['model_path']) as b:
        assert np.array_equal(a[0].data, b[0].data, equal_nan=True)


@pytest.mark.parametrize('defect', ['observation', 'health', 'array', 'grid', 'hash'])
def test_foreign_or_changed_identity_is_rejected(tmp_path, defect):
    original = native_map(tmp_path / 'original')
    bundle_update(original, frozen_health_identity='health:one')
    context, record = feedback.initialize(original, tmp_path / 'M0')
    assert record['status'] == 'ready'
    other = native_map(tmp_path / 'other', 'a1400' if defect == 'array' else 'a2000',
                       step=1. if defect == 'grid' else 2.,
                       observation=152418 if defect == 'observation' else 152389)
    bundle_update(other, frozen_health_identity='health:two' if defect == 'health' else 'health:one')
    if defect == 'hash':
        other.write_bytes(other.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        feedback.update(other, context, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_missing_source_support_stops_without_source_or_fallback(tmp_path):
    truncated = native_map(tmp_path / 'small', size=31)
    context, record = feedback.initialize(truncated, tmp_path / 'stopped')
    assert context is None and record['status'] == 'stopped'
    assert record['guard']['status'] == 'unavailable'
    assert 'truncated' in record['cause']
    assert record['model_path'] is record['model_sha256'] is None
    assert {p.name for p in (tmp_path / 'stopped').iterdir()} == {'feedback.json'}
    path = native_map(tmp_path / 'whole')
    context, _ = feedback.initialize(path, tmp_path / 'M0')
    missing = native_map(tmp_path / 'missing')
    with fits.open(missing, mode='update') as hdus:
        hdus['SCI_SUPPORT'].data[60, 60] = 0
        hdus[0].data[60, 60] = np.nan
    refresh_bundle(missing)
    record = feedback.update(missing, context, tmp_path / 'M1')
    assert record['status'] == 'stopped' and record['guard']['status'] == 'unavailable'
    assert record['model_path'] is None


def test_candidate_guard_is_not_relaxed_and_solver_errors_propagate(tmp_path, monkeypatch):
    path = native_map(tmp_path / 'map')
    original = feedback.candidate

    def empty(*args):
        selected = original(*args)
        selected['selected'][:] = False
        selected['candidate'][selected['valid']] = 0.
        return selected

    monkeypatch.setattr(feedback, 'candidate', empty)
    context, record = feedback.initialize(path, tmp_path / 'empty')
    assert context is None and record['status'] == 'stopped'
    assert record['guard']['accepted']
    assert 'empty candidate' in record['cause']
    assert record['model_path'] is None

    def failure(*args, **kwargs):
        raise RuntimeError('unexpected scientific owner defect')

    monkeypatch.setattr(feedback, 'measure_expected', failure)
    with pytest.raises(RuntimeError, match='unexpected'):
        feedback.initialize(path, tmp_path / 'defect')
    assert not (tmp_path / 'defect').exists()


def test_verbose_only_changes_diagnostics(tmp_path):
    path = native_map(tmp_path / 'map')
    _, quiet = feedback.initialize(path, tmp_path / 'quiet')
    _, verbose = feedback.initialize(path, tmp_path / 'verbose', verbose=True)
    assert quiet['guard'] == verbose['guard']
    assert quiet['candidate_summary'] == verbose['candidate_summary']
    with fits.open(quiet['model_path']) as a, fits.open(verbose['model_path']) as b:
        assert [h.name for h in b] == ['CANDIDATE', 'ADMITTED', 'VALID', 'RATIO']
        for one, two in zip(a, b):
            assert np.array_equal(one.data, two.data, equal_nan=True)
    assert {p.name for p in (tmp_path / 'verbose').glob('*.npz')} == {'C0.npz', 'BASE.npz'}


def test_recovery_measurement_only_uses_frozen_context(tmp_path, monkeypatch):
    path = native_map(tmp_path / 'map', 'a1100', step=2.)
    context, initial = feedback.initialize(path, tmp_path / 'initial')
    assert initial['status'] == 'ready'
    def forbidden(*args, **kwargs):
        raise AssertionError('measurement-only must not learn or select feedback')
    monkeypatch.setattr(feedback, 'learn', forbidden)
    monkeypatch.setattr(feedback, 'candidate', forbidden)
    with fits.open(path) as h:
        zero = np.zeros(h[0].data.shape)
    ordinary = feedback.measure_only(path, context)
    added = feedback.measure_only(path, context, zero)
    assert ordinary['accepted_parameters'] == added['accepted_parameters']
    with pytest.raises(ValueError, match='addition invalid'):
        feedback.measure_only(path, context, np.zeros((2, 2)))
