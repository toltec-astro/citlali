"""Owner-run SCIENCE compact-route regression cases; no real observation access."""
from copy import deepcopy
import json

import numpy as np
import pytest
import yaml
from astropy.io import fits

import configured_point_cli as cli
import configured_point_feedback as feedback
import configured_science_products as products
from configured_map_frames import SCIENCE_FRAME, grid_keys
from configured_point_config import REPO, SCIENCE_SCHEMA, load_request
from inspect_forward_map import digest
from test_configured_point_config import config
from test_configured_point_weighted import weighted_map, orchestration


@pytest.fixture
def science_config(config):
    overlay = yaml.safe_load((REPO / 'data/configured_science_compact_v1.yaml').read_text())
    config['successor'] = overlay['successor']
    config['runtime'].update(overlay['runtime'])
    config['coadd'] = overlay['coadd']
    return config


def celestial_map(root, array='a2000', *, cycle=0, health='sha256:'+'1'*64,
                  center=(180., 30.)):
    bundle = weighted_map(root, array, cycle=cycle, health=health)
    target = root / 'map.fits'
    with fits.open(target, mode='update') as hdus:
        for hdu in hdus:
            hdu.header.update(SCIENCE_FRAME)
            hdu.header['CRVAL1'], hdu.header['CRVAL2'] = center
            hdu.header['CDELT1'] /= 3600.
            hdu.header['CDELT2'] /= 3600.
        bundle['WCS'] = {k: hdus[0].header[k] for k in grid_keys(hdus[0].header)}
    plan_path = root / 'plan.yaml'
    plan = yaml.safe_load(plan_path.read_text())
    plan.update(schema='citlali-weighted-science-map-plan-v1', WCS=bundle['WCS'])
    plan_path.write_text(yaml.safe_dump(plan))
    bundle['files'].update({'map.fits': digest(target), 'plan.yaml': digest(plan_path)})
    (root / 'bundle.yaml').write_text(yaml.safe_dump(bundle))
    return bundle


def test_requested_science_plan_has_one_mode_and_retains_inputs(science_config, tmp_path):
    original = deepcopy(science_config)
    science_config['runtime']['n_threads'] = 32
    request = load_request(science_config, tmp_path)
    assert request.reduction_type == 'science' and request.weighted
    assert request.workers == 32 and request.feedback_iterations == 5
    assert request.feedback_enabled and not request.fit_diagnostics
    assert request.effective_config['successor']['schema'] == SCIENCE_SCHEMA
    assert request.effective_config['mapmaking']['pixel_axes'] == 'radec'
    assert request.effective_config['coadd'] == {'enabled': False}
    assert request.effective_config['inputs'][0]['cal_items'][1] == original['inputs'][0]['cal_items'][1]
    original['runtime']['n_threads'] = 32
    assert science_config == original


@pytest.mark.parametrize('defect', ['mode', 'frame', 'coadd', 'coadd_type', 'obs', 'fit', 'workers', 'comparison'])
def test_unsupported_science_choices_fail_before_output(science_config, tmp_path, defect):
    if defect == 'mode':
        science_config['runtime']['reduction_type'] = 'pointing'
    elif defect == 'frame':
        science_config['successor']['map']['frame'] = 'delta-source-AltAz'
    elif defect == 'coadd':
        science_config['coadd']['enabled'] = True
    elif defect == 'coadd_type':
        science_config['coadd']['enabled'] = 'false'
    elif defect == 'obs':
        science_config['inputs'][0]['meta']['name'] = '152418_0_2'
    elif defect == 'fit':
        science_config['successor']['products']['fit_diagnostics'] = True
    elif defect == 'workers':
        science_config['runtime']['n_threads'] = 33
    else:
        science_config['successor']['weight_comparison'] = True
    with pytest.raises(ValueError):
        load_request(science_config, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_point_entry_rejects_science_before_executable_or_output(science_config, tmp_path, monkeypatch):
    source = tmp_path / 'request.yaml'
    source.write_text(yaml.safe_dump(science_config))
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **k: pytest.fail('launched native'))
    with pytest.raises(ValueError, match='requested reduction mode'):
        cli.execute(source, tmp_path / 'missing-native', launch_dir=tmp_path)
    assert not (tmp_path / 'reduced').exists()


@pytest.mark.parametrize('array', products.ARRAYS)
def test_celestial_feedback_retains_units_origin_and_replacement(tmp_path, monkeypatch, array):
    initial = tmp_path / 'M0' / array
    celestial_map(initial, array)
    context, first = feedback.initialize(initial / 'map.fits', tmp_path / 'feedback0')
    assert first['status'] == 'ready'
    assert first['guard']['accepted_centroid_tangent_arcsec'] == pytest.approx([2., -3.], abs=.01)
    assert first['schema'] == 'configured-science-compact-feedback-v1'
    assert first['expected_origin_sky'] == dict(ra_deg=180., dec_deg=30., frame='FK5', equinox=2000.)
    assert first['guard']['centroid_axes'] == ['tangent_ra', 'tangent_dec']
    assert first['coordinate_units'] == 'arcsec' and first['fits_axis_units'] == 'deg'
    with fits.open(first['model_path']) as hdus:
        for hdu in hdus:
            assert all(hdu.header[k] == v for k, v in SCIENCE_FRAME.items())
            assert hdu.header['CRVAL1'] == 180. and hdu.header['CRVAL2'] == 30.
            assert hdu.header['CDELT1'] == -2./3600.
            assert hdu.verify_checksum() == hdu.verify_datasum() == 1
        valid = hdus['VALID'].data.astype(bool)
        admitted = hdus['ADMITTED'].data.astype(bool)
        assert np.isnan(hdus[0].data[~valid]).all()
        assert (hdus[0].data[valid & ~admitted] == 0).all()
    current = tmp_path / 'M1' / array
    celestial_map(current, array, cycle=1)
    monkeypatch.setattr(feedback, 'learn', lambda *a, **kw: pytest.fail('relearned frozen selector'))
    second = feedback.update(current / 'map.fits', context, tmp_path / 'feedback1')
    assert second['status'] == 'ready' and second['replacement'] and second['gain'] == 1.
    assert second['initial_map_sha256'] == first['map_sha256']
    with fits.open(first['model_path']) as a, fits.open(second['model_path']) as b:
        assert np.array_equal(a[0].data, b[0].data, equal_nan=True)
    foreign = tmp_path / 'foreign' / array
    celestial_map(foreign, array, cycle=1, center=(181., 30.))
    with pytest.raises(ValueError, match='frozen grid'):
        feedback.update(foreign / 'map.fits', context, tmp_path / 'bad')


def test_all_three_celestial_products_and_no_point_consumer(tmp_path, monkeypatch):
    root = tmp_path / 'maps'
    for array in products.ARRAYS:
        celestial_map(root / array, array, cycle=5)
    import point_reference
    monkeypatch.setattr(point_reference, 'measure_forward', lambda *a, **kw: pytest.fail('POINT fit invoked'))
    output = tmp_path / 'raw'
    records = products.publish_maps(root, output, 152389, products.ARRAYS)
    assert len(records) == 3
    assert len(list(output.iterdir())) == 6
    for record in records:
        assert record['native_cycle_index'] == 5 and record['coadd'] is False
        assert record['frame'] == SCIENCE_FRAME['ASTFRAME']
        assert (output / record['map']).read_bytes() == (root / record['array'] / 'map.fits').read_bytes()
        assert not record['calibration_eligible']
    assert not list(output.glob('ppt*')) and not list(output.glob('point*'))


@pytest.mark.parametrize('defect', ['center', 'mode', 'checksum', 'copy'])
def test_science_publication_fails_on_foreign_or_required_output(tmp_path, monkeypatch, defect):
    root = tmp_path / 'maps'
    for array in products.ARRAYS:
        if defect == 'mode' and array == 'a1400':
            weighted_map(root / array, array)
        else:
            celestial_map(root / array, array, center=(181., 30.) if defect == 'center' and array == 'a1400' else (180., 30.))
    if defect == 'checksum':
        target = root / 'a1400' / 'map.fits'
        target.write_bytes(target.read_bytes() + b'changed')
    if defect == 'copy':
        def fail(*a, **kw):
            raise OSError('required output failed')
        monkeypatch.setattr(products.shutil, 'copyfile', fail)
    with pytest.raises((ValueError, OSError)):
        products.publish_maps(root, tmp_path / 'raw', 152389, products.ARRAYS)
    if defect != 'copy':
        assert not (tmp_path / 'raw').exists()


def test_science_iterations_reuse_original_memory_cal_and_health(orchestration):
    state = orchestration
    state.request.reduction_type = 'science'
    rows = []
    for nw in state.prepared['networks']:
        root = state.work / 'upstream/products' / f'network{nw}'
        for target in (root / 'receipt.yaml', root / 'processing-scans.yaml', root / 'donor-continuity/cal/receipt.yaml'):
            target.unlink()  # A disk-replay fallback would fail this test.
        rows.append(dict(root=str(root), input=cli.bind(state.work / f'inputs/network{nw}-input.json'),
                         transport='observation-memory', handoff_sha256='a'*64))
    state.record['stages'].append(dict(point_handoff=rows))
    original = state.native_call

    def native(*args, **kwargs):
        request_path, directory = args[1:3]
        value = json.loads(request_path.read_text())
        assert value['schema'] == 'citlali-configured-weighted-science-v1'
        result = original(*args, **kwargs)
        if value['stage'] == 'cycle':
            output = directory / 'products'
            cycle = yaml.safe_load((output / 'cycle.yaml').read_text())
            # Convert the synthetic producer's existing maps, retaining identities.
            for array in products.ARRAYS:
                root = output / array
                with fits.open(root / 'map.fits', mode='update') as hdus:
                    for hdu in hdus:
                        hdu.header.update(SCIENCE_FRAME)
                        hdu.header['CRVAL1'], hdu.header['CRVAL2'] = 180., 30.
                        hdu.header['CDELT1'] /= 3600.
                        hdu.header['CDELT2'] /= 3600.
                    wcs = {k: hdus[0].header[k] for k in grid_keys(hdus[0].header)}
                bundle = cycle['arrays'][array]
                bundle['WCS'] = wcs
                plan_path = root / 'plan.yaml'
                plan = yaml.safe_load(plan_path.read_text())
                plan.update(schema='citlali-weighted-science-map-plan-v1', WCS=wcs)
                plan_path.write_text(yaml.safe_dump(plan))
                bundle['files'].update({'map.fits': digest(root / 'map.fits'), 'plan.yaml': digest(plan_path)})
                (root / 'bundle.yaml').write_text(yaml.safe_dump(bundle))
            cycle['schema'] = 'citlali-weighted-science-cycle-v1'
            (output / 'cycle.yaml').write_text(yaml.safe_dump(cycle))
        return result

    import configured_point_weighted as weighted
    final = weighted.execute(state.prepared, state.work, state.reduction, None, state.request, state.record,
                             native_call=native, bind=cli.bind, write_json=cli.write_json, memory_handoff=True)
    assert final == state.work / 'weighted-02/products'
    assert [row['stage'] for row in state.calls] == ['health', 'cycle', 'cycle', 'cycle']
    cycles = state.calls[1:]
    assert cycles[0]['models'] == {}
    assert all(set(row['models']) == set(products.ARRAYS) for row in cycles[1:])
    assert all(row['networks'] == rows and row['health'] == state.record['frozen_health'] for row in cycles)
    assert not (state.work / 'health/products/health.yaml').exists()
    assert state.record['feedback']['stopping_reason'] == 'fixed_budget_completed'
    assert state.record['feedback']['convergence_claim'] is False
    assert (state.reduction / 'weighted-science.json').is_file()


@pytest.mark.parametrize('changed', [False, True])
def test_science_cli_final_binding_precedes_visible_publication(science_config, tmp_path, monkeypatch, changed):
    from test_configured_point_weighted import _weighted_cli_handoff_case
    _weighted_cli_handoff_case(tmp_path, science_config, monkeypatch, retain=False, changed=changed, science=True)
    root = tmp_path / 'reduced/redu00'
    record = json.loads((root / 'RUN.json').read_text())
    assert record['schema'] == 'citlali-configured-science-run-v1'
    assert (root / '.science-work').is_dir()
    assert not list(root.rglob('ppt*')) and not list(root.rglob('point-assessment*'))


def test_science_entry_and_overlay_are_packaged():
    import tomllib
    from configured_point_runtime import RUNTIME_MODULES
    from package_configured_point import DATA_FILES
    from pathlib import Path
    project = tomllib.loads((REPO / 'pyproject.toml').read_text())
    assert project['project']['scripts']['citlali-configured-science'] == 'configured_science_cli:main'
    assert {'configured_map_frames', 'configured_science_cli', 'configured_science_products'} <= set(RUNTIME_MODULES)
    assert Path('data/configured_science_compact_v1.yaml') in DATA_FILES
    assert cli.default_config('science')['successor']['schema'] == SCIENCE_SCHEMA
