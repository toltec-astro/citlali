"""Weighted publication/orchestration contracts with synthetic products only."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from astropy.io import fits

import configured_point_cli as cli
import configured_point_feedback as feedback
import configured_point_products as products
import configured_point_weighted as weighted
from inspect_forward_map import digest
from test_configured_point_products import native_map
from test_configured_point_v2_config import config, weighted_config


HEALTH = 'sha256:' + '1' * 64


def weighted_map(root, array='a2000', *, cycle=0, health=HEALTH):
    path = native_map(root, array)
    with fits.open(path, mode='update') as hdus:
        for hdu, unit in zip(hdus, products.EMPIRICAL_UNITS):
            hdu.header['BUNIT'] = unit
        hdus['NORMALIZATION'].data *= 2.3
        hdus['NUMERATOR'].data *= 2.3
    bundle_path = root / 'bundle.yaml'
    bundle = yaml.safe_load(bundle_path.read_text())
    plan = dict(schema='citlali-weighted-point-map-plan-v1', observation=bundle['observation'],
                array=array, cycle_index=cycle, frozen_health_identity=health, WCS=bundle['WCS'])
    (root / 'plan.yaml').write_text(yaml.safe_dump(plan))
    bundle.update(coefficient_family=products.EMPIRICAL_FAMILY,
                  normalization_interpretation=products.EMPIRICAL_INTERPRETATION,
                  cycle_index=cycle, frozen_health_identity=health,
                  contributing_detectors=3, contributing_samples=37)
    bundle['files'].update({'map.fits': digest(path), 'plan.yaml': digest(root / 'plan.yaml')})
    bundle_path.write_text(yaml.safe_dump(bundle))
    return bundle


def publish(root, output, arrays=products.ARRAYS):
    return products.publish_maps(root, output, 152389, arrays,
                                 fit_diagnostics=False, verbose=False)


def test_weighted_publisher_retains_actual_units_and_no_precision_claim(tmp_path):
    root = tmp_path / 'native'
    originals = {}
    for array in products.ARRAYS:
        weighted_map(root / array, array, cycle=3)
        originals[array] = (root / array / 'map.fits').read_bytes()
    rows = publish(root, tmp_path / 'raw')
    assert len(rows) == 3
    for row in rows:
        assert row['coefficient_family'] == products.EMPIRICAL_FAMILY
        assert row['coefficient_interpretation'] == products.EMPIRICAL_INTERPRETATION
        assert row['native_cycle_index'] == 3 and row['frozen_health_identity'] == HEALTH
        assert row['plane_units'] == dict(zip(products.PLANES, products.EMPIRICAL_UNITS))
        assert 'not map precision' in row['normalization']
        assert row['calibration_eligible'] is row['legacy_analysis_compatible'] is False
        path = tmp_path / 'raw' / row['map']
        assert path.read_bytes() == originals[row['array']]
        with fits.open(path) as hdus:
            assert [h.header['BUNIT'] for h in hdus] == list(products.EMPIRICAL_UNITS)
            assert not np.array_equal(hdus['HITS'].data, hdus['NORMALIZATION'].data)


@pytest.mark.parametrize('defect', ['family', 'units', 'interpretation', 'plan_hash',
                                  'plan_cycle', 'plan_health', 'cycle', 'health', 'missing_array', 'mixed'])
def test_malformed_weighted_family_or_associations_never_publish(tmp_path, defect):
    root = tmp_path / 'native'
    for array in products.ARRAYS:
        if defect == 'missing_array' and array == 'a1400':
            continue
        if defect == 'mixed' and array == 'a1400':
            native_map(root / array, array)
        else:
            weighted_map(root / array, array,
                         cycle=1 if defect == 'cycle' and array == 'a1400' else 0,
                         health='sha256:'+'2'*64 if defect == 'health' and array == 'a1400' else HEALTH)
    target = root / 'a1100'
    bundle_path = target / 'bundle.yaml'
    bundle = yaml.safe_load(bundle_path.read_text())
    if defect == 'family':
        bundle['coefficient_family'] = 'invented-inverse-covariance'
    elif defect == 'interpretation':
        bundle['normalization_interpretation'] = 'known-map-precision'
    elif defect == 'units':
        with fits.open(target / 'map.fits', mode='update') as hdus:
            hdus['NORMALIZATION'].header['BUNIT'] = '1'
        bundle['files']['map.fits'] = digest(target / 'map.fits')
    elif defect in ('plan_cycle', 'plan_health', 'plan_hash'):
        plan_path = target / 'plan.yaml'
        plan = yaml.safe_load(plan_path.read_text())
        plan['cycle_index'] = 12
        if defect == 'plan_health':
            plan['cycle_index'] = 0
            plan['frozen_health_identity'] = 'sha256:'+'2'*64
        plan_path.write_text(yaml.safe_dump(plan))
        if defect != 'plan_hash':
            bundle['files']['plan.yaml'] = digest(plan_path)
    bundle_path.write_text(yaml.safe_dump(bundle))
    with pytest.raises((ValueError, FileNotFoundError)):
        publish(root, tmp_path / 'raw')
    assert not (tmp_path / 'raw').exists()


def test_weighted_publication_cannot_silently_select_a_subset(tmp_path):
    weighted_map(tmp_path / 'native/a2000')
    with pytest.raises(ValueError, match='all three arrays'):
        publish(tmp_path / 'native', tmp_path / 'raw', ('a2000',))
    assert not (tmp_path / 'raw').exists()


def health_record():
    networks = []
    for array, network in enumerate((0, 7, 11)):
        rows = []
        for index, state in enumerate(weighted.STATES):
            rows.append(dict(detector_id=f'network:{network}/channel:{index}', state=state,
                             reason=state, relative_response=1. if state == 'retained' else .03,
                             interval=[.01, .05]))
        networks.append(dict(array=array, network=network, detectors=rows))
    return dict(schema='citlali-point-optical-health-v1', min_relative_response=.10,
                method='fixture-source-protected-response',
                uncertainty_limitations='fixture intervals; no false-rejection qualification',
                networks=networks)


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    reduction = tmp_path / 'reduction'
    work = reduction / '.point-work'
    inputs = work / 'inputs'
    inputs.mkdir(parents=True)
    effective = inputs / 'effective.yaml'
    effective.write_text('successor: {schema: citlali-point-v2}\n')
    for network in (0, 7, 11):
        (inputs / f'network{network}-input.json').write_text(json.dumps(dict(network=network)))
        root = work / 'upstream/products' / f'network{network}'
        (root / 'donor-continuity/cal').mkdir(parents=True)
        (root / 'donor-continuity/cal/receipt.yaml').write_text('completed: true\n')
        (root / 'receipt.yaml').write_text('original: RTC\n')
        (root / 'processing-scans.yaml').write_text('scans: [1, 2]\n')
    prepared = dict(networks=[0, 7, 11], effective_config=effective, observation=152389)
    request = SimpleNamespace(verbose=False, feedback_enabled=True, feedback_iterations=2, weight_comparison=False)
    record = dict(stages=[])
    state = SimpleNamespace(work=work, reduction=reduction, prepared=prepared, request=request,
                            record=record, calls=[], selections=[], defect=None, stop=None)

    def native_call(native, request_path, directory, **kwargs):
        value = json.loads(request_path.read_text())
        state.calls.append(deepcopy(value))
        output = directory / 'products'
        output.mkdir(parents=True)
        if value['stage'] == 'health':
            (output / 'health.yaml').write_text(yaml.safe_dump(health_record()))
        else:
            cycle = value['cycle_index']
            health = 'sha256:' + value['health']['sha256']
            arrays = {array: weighted_map(output / array, array, cycle=cycle, health=health)
                      for array in products.ARRAYS}
            result = dict(schema='citlali-weighted-point-cycle-v1', cycle_index=cycle,
                          frozen_health_identity=health, coefficient_family=products.EMPIRICAL_FAMILY,
                          normalization_interpretation=products.EMPIRICAL_INTERPRETATION, arrays=arrays)
            if state.defect == 'cycle':
                result['cycle_index'] += 1
            elif state.defect == 'health':
                result['frozen_health_identity'] = HEALTH
            elif state.defect == 'missing_array':
                del result['arrays']['a1400']
            elif state.defect == 'nested_cycle':
                result['arrays']['a1400']['cycle_index'] += 1
            elif state.defect == 'nested_health':
                result['arrays']['a1400']['frozen_health_identity'] = HEALTH
            elif state.defect in ('actual_cycle', 'actual_health', 'actual_observation'):
                bundle = result['arrays']['a1400']
                if state.defect == 'actual_cycle':
                    bundle['cycle_index'] += 1
                elif state.defect == 'actual_health':
                    bundle['frozen_health_identity'] = HEALTH
                else:
                    bundle['observation'] = 'obs:152418_0_2'
                (output / 'a1400/bundle.yaml').write_text(yaml.safe_dump(bundle))
            (output / 'cycle.yaml').write_text(yaml.safe_dump(result))
        return dict(stage=value['stage'], terminal=kwargs['terminal'])

    original_initialize, original_update = feedback.initialize, feedback.update

    def select(path, context, destination, verbose, initial):
        cycle = int(path.parents[2].name.split('-')[1])
        array = path.parent.name
        state.selections.append((cycle, array, initial))
        if state.stop == (cycle, array):
            result = dict(status='stopped', cause='fixture source support unavailable', model_path=None,
                          model_sha256=None, guard=dict(status='unavailable'))
            return (None, result) if initial else result
        result = original_initialize(path, destination, verbose) if initial else original_update(path, context, destination, verbose)
        if state.defect == 'mutate_health' and cycle == 0 and array == 'a2000':
            health = work / 'health/products/health.yaml'
            health.write_text(health.read_text() + '\n# changed after initial source selection\n')
        return result

    monkeypatch.setattr(feedback, 'initialize', lambda path, destination, verbose=False:
                        select(path, None, destination, verbose, True))
    monkeypatch.setattr(feedback, 'update', lambda path, context, destination, verbose=False:
                        select(path, context, destination, verbose, False))

    def execute():
        return weighted.execute(prepared, work, reduction, Path('/synthetic/native'), request, record,
                                native_call=native_call, bind=cli.bind, write_json=cli.write_json)

    state.native_call, state.execute = native_call, execute
    return state


def test_health_summary_keeps_reasons_separate_and_counts_each_detector_once(tmp_path):
    path = tmp_path / 'health.yaml'
    path.write_text(yaml.safe_dump(health_record()))
    result = weighted.health_summary(path)
    assert result['min_relative_response'] == .10
    for array in products.ARRAYS:
        assert result['arrays'][array] == {state: 1 for state in weighted.STATES}
    assert len(result['exceptions']) == 9
    assert {e['state'] for e in result['exceptions']} == {'weak_response', 'sign_mismatch', 'unresolved'}


@pytest.mark.parametrize('defect', ['negative_array', 'high_array', 'boolean_array',
                                  'duplicate_network', 'duplicate_detector', 'unknown_state'])
def test_health_summary_rejects_ambiguous_population_counts(tmp_path, defect):
    health = health_record()
    if defect == 'negative_array':
        health['networks'][0]['array'] = -1
    elif defect == 'high_array':
        health['networks'][0]['array'] = 3
    elif defect == 'boolean_array':
        health['networks'][0]['array'] = True
    elif defect == 'duplicate_network':
        health['networks'].append(deepcopy(health['networks'][0]))
    elif defect == 'duplicate_detector':
        health['networks'][1]['detectors'][0]['detector_id'] = health['networks'][0]['detectors'][0]['detector_id']
    else:
        health['networks'][0]['detectors'][0]['state'] = 'new-unknown-state'
    path = tmp_path / 'health.yaml'
    path.write_text(yaml.safe_dump(health))
    with pytest.raises(ValueError):
        weighted.health_summary(path)


def test_orchestrator_freezes_cal_health_and_replaces_models_for_fixed_budget(orchestration):
    state = orchestration
    final = state.execute()
    assert final == state.work / 'weighted-02/products'
    assert [row['stage'] for row in state.calls] == ['health', 'cycle', 'cycle', 'cycle']
    cycles = state.calls[1:]
    assert [row['cycle_index'] for row in cycles] == [0, 1, 2]
    assert cycles[0]['models'] == {}
    assert all(set(row['models']) == set(products.ARRAYS) for row in cycles[1:])
    assert all(row['networks'] == cycles[0]['networks'] for row in cycles)
    assert all(row['health'] == cycles[0]['health'] for row in cycles)
    assert all('cal_receipt_sha256' in n and 'ptc_receipt_sha256' not in n for n in cycles[0]['networks'])
    for cycle in (1, 2):
        for array, model in cycles[cycle]['models'].items():
            assert set(model) == {'candidate', 'feedback', 'bundle'}
            assert f'weighted-{cycle-1:02d}/feedback-{array}' in model['candidate']['path']
            receipt = json.loads(Path(model['feedback']['path']).read_text())
            assert receipt['model_sha256'] == model['candidate']['sha256']
            assert receipt['input_bundle_sha256'] == model['bundle']['sha256']
            bundle = yaml.safe_load(Path(model['bundle']['path']).read_text())
            assert bundle['cycle_index'] == cycle-1
            assert bundle['files']['map.fits'] == receipt['map_sha256']
    trajectory = state.record['feedback']
    assert trajectory['applied_feedback_applications'] == 2
    assert trajectory['stopping_reason'] == 'fixed_budget_completed'
    assert trajectory['convergence_claim'] is False
    assert state.record['final_cycle'] == 2
    assert state.selections == [(cycle, array, cycle == 0) for cycle in (0, 1) for array in products.ARRAYS]


@pytest.mark.parametrize('stop_cycle', [0, 1])
def test_unavailable_source_stops_at_last_complete_all_array_map(orchestration, stop_cycle):
    state = orchestration
    state.stop = (stop_cycle, 'a1400')
    final = state.execute()
    assert final == state.work / f'weighted-{stop_cycle:02d}/products'
    trajectory = state.record['feedback']
    assert trajectory['requested_feedback_applications'] == 2
    assert trajectory['applied_feedback_applications'] == stop_cycle
    assert trajectory['stopping_reason'] == 'source_feedback_unavailable'
    assert trajectory['convergence_claim'] is False
    assert len(state.calls) == 2 + stop_cycle
    assert all((final / array / 'map.fits').exists() for array in products.ARRAYS)


@pytest.mark.parametrize('defect', ['cycle', 'health', 'missing_array', 'nested_cycle', 'nested_health',
                                  'actual_cycle', 'actual_health', 'actual_observation'])
def test_incomplete_or_foreign_cycle_is_not_accepted(orchestration, defect):
    orchestration.defect = defect
    with pytest.raises(ValueError):
        orchestration.execute()
    assert 'final_cycle' not in orchestration.record
    assert orchestration.selections == []


def test_frozen_health_cannot_be_rebound_to_changed_bytes_on_later_pass(orchestration):
    orchestration.defect = 'mutate_health'
    with pytest.raises(ValueError, match='health'):
        orchestration.execute()
    assert 'final_cycle' not in orchestration.record
    assert len(orchestration.calls) == 2


def test_disabled_feedback_runs_one_weighted_forward_map_only(orchestration):
    orchestration.request.feedback_enabled = False
    final = orchestration.execute()
    assert final == orchestration.work / 'weighted-00/products'
    assert orchestration.selections == []
    assert orchestration.record['feedback']['requested_feedback_applications'] == 0
    assert orchestration.record['feedback']['stopping_reason'] == 'feedback_disabled'


def test_v2_cli_stops_upstream_at_cal_and_uses_weighted_dispatch(tmp_path, weighted_config, monkeypatch):
    weighted_config['runtime']['output_dir'] = str(tmp_path / 'reduced')
    weighted_config['successor']['rtc']['disturbance_exclusion'] = 'disabled'
    weighted_config['successor']['products'].update(fit_diagnostics=False, retain_intermediates=True)
    config_path = tmp_path / 'merged.yaml'
    config_path.write_text(yaml.safe_dump(weighted_config))
    binary = tmp_path / 'native'
    binary.write_text('fake identity\n')
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda *a, **kw: 'synthetic-native\n')
    calls = []

    def prepare(effective, destination, **kwargs):
        destination.mkdir()
        config = destination / 'effective.yaml'
        config.write_text(yaml.safe_dump(effective))
        observation = destination / 'observation.json'
        observation.write_text('{}\n')
        return dict(effective_config=config, observation_request=observation,
                    observation=152389, networks=[0, 7, 11], bindings=[])

    def native_call(native, request_path, directory, **kwargs):
        calls.append(kwargs)
        directory.mkdir()
        return dict(terminal=kwargs['terminal'])

    def execute(prepared, work, reduction, native, request, record, **kwargs):
        assert calls == [dict(terminal='cal', workers=6, method=None, verbose=False)]
        assert request.weighted
        for array in products.ARRAYS:
            weighted_map(work / 'weighted-00/products' / array, array)
        record['final_cycle'] = 0
        return work / 'weighted-00/products'

    monkeypatch.setattr(cli, 'prepare', prepare)
    monkeypatch.setattr(cli, 'native_call', native_call)
    monkeypatch.setattr(weighted, 'execute', execute)
    monkeypatch.setattr(cli, 'map_request', lambda *a, **kw: pytest.fail('legacy PTC MAP route used'))
    reduction = cli.execute(config_path, binary, launch_dir=tmp_path)
    record = json.loads((reduction / 'RUN.json').read_text())
    assert record['status'] == 'completed' and record['final_cycle'] == 0
    assert len(list((reduction / '152389/raw').glob('toltec_*.fits'))) == 3
