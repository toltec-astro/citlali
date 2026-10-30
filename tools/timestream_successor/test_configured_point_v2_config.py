"""Weighted POINT config boundary; v1 remains the explicit uniform profile."""
from copy import deepcopy

import pytest
import yaml

from configured_point_config import (ARRAYS, MIGRATION, MIGRATION_V2, REPO,
                                    SCHEMA, SCHEMA_V2, WEIGHTED_CLEANING, load_request)
from test_configured_point_config import config  # Shared ordinary TolTECA input fixture.


@pytest.fixture
def weighted_config(config):
    overlay = yaml.safe_load((REPO / 'data/configured_point_v2.yaml').read_text())
    config['successor'] = overlay['successor']
    config['runtime'].update(overlay['runtime'])
    return config


def test_v2_overlay_resolves_full_suite_without_mutating_merged_request(weighted_config, tmp_path):
    original = deepcopy(weighted_config)
    request = load_request(weighted_config, tmp_path)
    assert weighted_config == original
    assert request.weighted is True
    assert request.cleaning_method == WEIGHTED_CLEANING
    assert request.ranks == dict(a1100=3, a1400=4, a2000=6)
    assert request.min_relative_response == .10
    assert request.feedback_enabled is True and request.feedback_iterations == 10
    assert request.arrays == ARRAYS
    assert request.verbose is False and request.retain_intermediates is False
    successor = request.effective_config['successor']
    assert successor['schema'] == SCHEMA_V2 and successor['migration'] == MIGRATION_V2
    assert successor['detector_response'] == {'min_relative_response': .10}
    assert successor['feedback'] == {'enabled': True, 'max_iters': 10}
    assert request.effective_config['timestream']['fruit_loops'] == {'enabled': False}
    assert 'adaptive_selector' not in request.effective_config['timestream']['processed_time_chunk']['clean']
    semantics = request.effective_config['successor_effective']
    assert 'not map precision' in semantics['normalization']
    assert 'max(reference' in semantics['map_weight']
    assert 'not convergence' in semantics['feedback_budget']
    assert len(semantics['limitations']) == 3
    dispositions = {row['path']: row['replacement'] for row in request.migration_dispositions}
    assert 'uniform' not in dispositions['mapmaking']
    assert 'empirical inverse-noise' in dispositions['mapmaking']


def test_v1_retains_its_exact_effective_profile_and_new_state_is_inactive(config, tmp_path):
    request = load_request(config, tmp_path)
    assert request.weighted is False and request.min_relative_response is None
    assert request.feedback_enabled is False and request.feedback_iterations == 0
    assert request.effective_config['successor']['schema'] == SCHEMA
    assert request.effective_config['successor']['migration'] == MIGRATION
    assert request.effective_config['successor']['feedback'] == {'enabled': False}
    assert 'detector_response' not in request.effective_config['successor']
    assert 'successor_effective' not in request.effective_config
    assert request.cleaning_method == 'observed-als'


def test_weight_comparison_is_explicit_v2_only(weighted_config, config, tmp_path):
    assert not load_request(weighted_config, tmp_path).weight_comparison
    weighted_config['successor']['weight_comparison'] = True
    parsed = load_request(weighted_config, tmp_path)
    assert parsed.weight_comparison and parsed.effective_config['successor']['weight_comparison'] is True
    for invalid in ('true', 1, None):
        weighted_config['successor']['weight_comparison'] = invalid
        with pytest.raises(ValueError, match='boolean'):
            load_request(weighted_config, tmp_path)


@pytest.mark.parametrize('omit', ['block', 'threshold'])
def test_optical_threshold_default_is_recorded_even_when_omitted(weighted_config, tmp_path, omit):
    if omit == 'block':
        del weighted_config['successor']['detector_response']
    else:
        weighted_config['successor']['detector_response'] = {}
    request = load_request(weighted_config, tmp_path)
    assert request.min_relative_response == .10
    assert request.effective_config['successor']['detector_response'] == {'min_relative_response': .10}


@pytest.mark.parametrize('threshold', [.025, .20, 1.])
def test_optical_threshold_is_free_and_preserved_without_count_retuning(weighted_config, tmp_path, threshold):
    weighted_config['successor']['detector_response']['min_relative_response'] = threshold
    request = load_request(weighted_config, tmp_path)
    assert request.min_relative_response == threshold
    assert request.effective_config['successor']['detector_response']['min_relative_response'] == threshold


@pytest.mark.parametrize('threshold', [0, -1, 1.01, True, False, '0.1', None, [],
                                      float('nan'), float('inf'), -float('inf')])
def test_optical_threshold_domain_is_strict(weighted_config, tmp_path, threshold):
    weighted_config['successor']['detector_response']['min_relative_response'] = threshold
    with pytest.raises(ValueError, match='min_relative_response'):
        load_request(weighted_config, tmp_path)


def test_feedback_defaults_and_forward_only_request_are_explicit(weighted_config, tmp_path):
    weighted_config['successor']['feedback'] = {}
    request = load_request(weighted_config, tmp_path)
    assert request.feedback_enabled and request.feedback_iterations == 10
    weighted_config['successor']['feedback'] = {'enabled': False, 'max_iters': 4}
    request = load_request(weighted_config, tmp_path)
    assert request.weighted and request.feedback_enabled is False
    assert request.feedback_iterations == 4
    assert request.effective_config['successor']['feedback'] == {'enabled': False, 'max_iters': 4}
    weighted_config['successor']['feedback'] = {'enabled': True, 'max_iters': 3}
    assert load_request(weighted_config, tmp_path).feedback_iterations == 3


@pytest.mark.parametrize('field,value', [('enabled', 'true'), ('enabled', 1),
    ('max_iters', True), ('max_iters', 0), ('max_iters', -1), ('max_iters', 2.5),
    ('max_iters', '10'), ('max_iters', None), ('max_iters', float('inf'))])
def test_feedback_controls_are_not_coerced(weighted_config, tmp_path, field, value):
    weighted_config['successor']['feedback'][field] = value
    with pytest.raises(ValueError, match=field):
        load_request(weighted_config, tmp_path)


@pytest.mark.parametrize('section,field', [('detector_response', 'auto_tune'),
    ('cleaning', 'weight_cap'), ('feedback', 'convergence_threshold'), ('map', 'uniform_fallback')])
def test_unknown_weighted_policies_are_rejected(weighted_config, tmp_path, section, field):
    weighted_config['successor'][section][field] = True
    with pytest.raises(ValueError, match='unsupported successor'):
        load_request(weighted_config, tmp_path)


@pytest.mark.parametrize('defect', ['response', 'method', 'feedback_budget', 'migration'])
def test_v1_cannot_silently_consume_v2_controls(config, tmp_path, defect):
    if defect == 'response':
        config['successor']['detector_response'] = {'min_relative_response': .10}
    elif defect == 'method':
        config['successor']['cleaning']['method'] = WEIGHTED_CLEANING
    elif defect == 'feedback_budget':
        config['successor']['feedback']['max_iters'] = 10
    else:
        config['successor']['migration'] = MIGRATION_V2
    with pytest.raises(ValueError):
        load_request(config, tmp_path)


@pytest.mark.parametrize('key,value', [('schema', 'citlali-point-v3'),
    ('migration', MIGRATION), ('cleaning', {'method': 'observed-als', 'ranks': dict(a1100=3, a1400=4, a2000=6)})])
def test_v2_identity_and_science_method_must_be_consistent(weighted_config, tmp_path, key, value):
    weighted_config['successor'][key] = value
    with pytest.raises(ValueError):
        load_request(weighted_config, tmp_path)


def test_v2_rank_and_map_choices_remain_configurable(weighted_config, tmp_path):
    weighted_config['successor']['cleaning']['ranks'] = dict(a1100=2, a1400=5, a2000=7)
    weighted_config['successor']['map'].update(pixel_size_arcsec=2., coverage_cut=.25)
    request = load_request(weighted_config, tmp_path)
    assert request.ranks == dict(a1100=2, a1400=5, a2000=7)
    assert request.pixel_size_arcsec == 2. and request.coverage_cut == .25
    assert request.effective_config['timestream']['processed_time_chunk']['clean']['standard_pca']['n_eig_to_cut'] == {
        'a1100': [2], 'a1400': [5], 'a2000': [7]}
