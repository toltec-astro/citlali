import json
import yaml
import pytest
from configured_point_drops import collect, console_summary
from configured_point_config import DISTURBANCE_POLICY, load_request
from test_configured_point_config import config


def network(root, nw, dropped=0):
    path = root / f'network{nw}' / 'detector-drops.yaml'
    path.parent.mkdir()
    row = dict(schema='citlali-detector-drops-network-v1', policy=DISTURBANCE_POLICY,
               observation=152418, network=nw, array=0 if nw < 7 else 1 if nw < 11 else 2,
               evaluation_state='complete', application_state='applied', input_detectors=100,
               previously_unavailable=5, evaluated_detectors=95,
               selected_for_exclusion=dropped, newly_dropped=dropped,
               retained_after_decision=95-dropped,
               detectors=[dict(network=nw, channel=i, detector_occurrence=f'exact-occurrence-{i}',
                               reason='repeated_level_shifts_unassessable_remainder') for i in range(dropped)])
    path.write_text(yaml.safe_dump(row))
    return path, row


def test_zero_and_missing_are_distinct(tmp_path):
    network(tmp_path, 0)
    s = collect(tmp_path, 152418, [0, 7, 11])
    assert s['evaluation_state'] == 'incomplete'
    assert s['arrays']['a1100']['newly_dropped'] == 0
    assert s['arrays']['a2000']['newly_dropped'] is None
    assert s['totals']['newly_dropped'] is None
    assert 'unavailable' in console_summary(s, 'summary.json')


def test_counts_separate_prior_losses_and_count_each_detector_once(tmp_path):
    for nw in [0, 7, 11]:
        network(tmp_path, nw, int(nw == 11))
    s = collect(tmp_path, 152418, [0, 7, 11])
    assert s['totals']['previously_unavailable'] == 15
    assert s['totals']['newly_dropped'] == 1
    assert s['arrays']['a2000']['retained_after_decision'] == 94
    assert s['evaluation_state'] == 'complete'
    assert 'a2000=1' in console_summary(s, 'summary.json')
    json.dumps(s, allow_nan=False)


@pytest.mark.parametrize('change', ['duplicate', 'foreign', 'unapplied', 'bad_count'])
def test_diagnostic_rejects_misleading_records(tmp_path, change):
    p, row = network(tmp_path, 11, 1)
    if change == 'duplicate':
        row['detectors'] *= 2
    elif change == 'foreign':
        row['observation'] = 152389
    elif change == 'unapplied':
        row['newly_dropped'] = None
    else:
        row['evaluated_detectors'] += 1
    p.write_text(yaml.safe_dump(row))
    with pytest.raises(ValueError):
        collect(tmp_path, 152418, [11])


def test_rule_explicit_and_unknown_rule_rejected(config, tmp_path):
    assert load_request(config, tmp_path).effective_config['successor']['rtc']['disturbance_exclusion'] == 'disabled'
    config['successor']['rtc']['disturbance_exclusion'] = DISTURBANCE_POLICY
    assert load_request(config, tmp_path).effective_config['successor']['rtc']['disturbance_exclusion'] == DISTURBANCE_POLICY
    config['successor']['rtc']['disturbance_exclusion'] = 'try-until-pass'
    with pytest.raises(ValueError):
        load_request(config, tmp_path)
