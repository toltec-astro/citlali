"""Exercise only pure saved-table readers; never run calibration/setup/reduction."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
from astropy.table import Table
import yaml

from tolproj.steps.calibrate_flxscale import _find_pointing_product, _read_recovered_fluxes
from tolproj.errors import TolprojUserError

PROJECT = Path('/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027')
ARRAYS = ('a1100', 'a1400', 'a2000')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apt(manifest):
    table = Table.read(manifest)
    row = table[table['role'] == 'apt'][0]
    path = manifest.parent / str(row['relative_path'])
    assert 'sha256:' + digest(path) == str(row['transport_sha256'])
    return Table.read(path), {'manifest': str(manifest), 'manifest_sha256': digest(manifest),
                             'apt': str(path), 'apt_sha256': digest(path)}


def audit(output):
    recovered = {}
    for obs in (152389, 152391, 152393, 152418):
        path = _find_pointing_product(PROJECT/'reference/v2-science', 'redu00', obs)
        recovered[str(obs)] = _read_recovered_fluxes(path, obs)[0]
    template = Table.read(_find_pointing_product(PROJECT/'reference/v2-science', 'redu00', 152389))
    cases = {}
    with tempfile.TemporaryDirectory(prefix='point-consumer-audit-') as scratch:
        tables = {'valid': template.copy(), 'missing_array': template[:2].copy(),
                  'nonfinite_amp': template.copy(), 'wrong_amp_unit': template.copy(),
                  'wrong_error_unit': template.copy(), 'ineligible_metadata': template.copy(),
                  'fractional_array': template.copy(), 'zero_error': template.copy()}
        tables['nonfinite_amp']['amp'][0] = np.nan
        tables['wrong_amp_unit']['amp'].unit = 'Jy/beam'
        tables['wrong_error_unit']['amp_err'].unit = 'Jy/beam'
        tables['ineligible_metadata'].meta['calibration_eligible'] = False
        tables['fractional_array']['array'] = [0.1, 1.1, 2.1]
        tables['zero_error']['amp_err'][0] = 0.
        for name, table in tables.items():
            path = Path(scratch)/f'{name}.ecsv'
            table.write(path, format='ascii.ecsv')
            try:
                _read_recovered_fluxes(path, 152389)
                cases[name] = 'accepted'
            except TolprojUserError:
                cases[name] = 'rejected'
    assert cases == {'valid': 'accepted', 'missing_array': 'rejected',
        'nonfinite_amp': 'rejected', 'wrong_amp_unit': 'rejected',
        'wrong_error_unit': 'accepted', 'ineligible_metadata': 'accepted',
        'fractional_array': 'accepted', 'zero_error': 'accepted'}
    discovery = {}
    for obs in (152389, 152418):
        try:
            _find_pointing_product(PROJECT, 'redu00', obs)
            discovery[str(obs)] = 'found'
        except TolprojUserError:
            discovery[str(obs)] = 'no standard project POINT table'
        weighted = PROJECT/f'pointings-weighted-{obs}-20261029/reduced/redu00/{obs}/raw'
        assert not list(weighted.glob('ppt_*.ecsv'))
    report_path = PROJECT/'apts/fluxscale_report.json'
    report = json.loads(report_path.read_text())
    bindings = []
    for cohort in report['cohorts']:
        obs = cohort['science_obsnums'][0]
        request = json.loads((PROJECT/f'apts/v2/fluxcal/requests/apt_{obs}_fluxcal_request.json').read_text())
        assert request['report_sha256'] == 'sha256:' + digest(report_path)
        factors = {}
        for array in ARRAYS:
            efficiencies = [recovered[str(p)][array]['flux_mjy']/report['references'][str(p)][array]['flux_mjy']
                            for p in cohort['bracket_obsnums']]
            factor = 1./np.mean(efficiencies)
            assert factor == cohort['correction_factors'][array]
            encoded = next(r['factor_bits'] for r in request['correction_factors'] if r['array_name'] == array)
            assert encoded == struct.pack('>d', factor).hex()
            factors[array] = float(factor)
        parent, parent_id = apt(PROJECT/f'apts/v2/apt_{obs}_matched.apt-v2/manifest.ecsv')
        child, child_id = apt(PROJECT/f'apts/v2/fluxcal/apt_{obs}_fluxcal.apt-v2/manifest.ecsv')
        reference, ref_id = apt(PROJECT/f'reference/v2-science/apts/v2/fluxcal/apt_{obs}_fluxcal.apt-v2/manifest.ecsv')
        assert child_id['manifest_sha256'] == ref_id['manifest_sha256']
        assert child_id['apt_sha256'] == ref_id['apt_sha256']
        assert np.array_equal(parent['uid'], child['uid'])
        assert np.array_equal(parent['array'], child['array'])
        rows = {}
        for index, array in enumerate(ARRAYS):
            mask = np.asarray(parent['array']) == index
            before = np.asarray(parent['flxscale'])[mask]
            after = np.asarray(child['flxscale'])[mask]
            # Missing parent measurements remain missing; compare every finite row.
            assert np.array_equal(np.isfinite(before), np.isfinite(after))
            good = np.isfinite(before)
            assert np.allclose(after[good], before[good]*factors[array], rtol=1e-14, atol=0)
            rows[array] = {'total': int(mask.sum()), 'finite_flxscale': int(good.sum())}
        bindings.append({'science_obsnum': obs, 'brackets': cohort['bracket_obsnums'],
                         'factors': factors, 'matched': parent_id, 'calibrated': child_id,
                         'selected_reference': ref_id, 'rows_checked': rows})
    yaml_path = PROJECT/'refactor/NGC4449/72_science_observation.yaml'
    authored = yaml.safe_load(yaml_path.read_text())['reduce']['steps'][0]['config']
    selected = [v for v in authored['cal_items'] if v['type'] == 'array_prop_table']
    assert {v['filepath'] for v in selected} == {b['selected_reference']['manifest'] for b in bindings}
    project_path = PROJECT/'project.yaml'
    project = yaml.safe_load(project_path.read_text())
    consumer = Path(__import__('tolproj.steps.calibrate_flxscale', fromlist=['x']).__file__)
    result = {'scope': 'read-only helper probes and saved calibration arithmetic; no calibration issued',
        'tolproj_reader': str(consumer), 'reader_sha256': digest(consumer),
        'tolproj_head': subprocess.check_output(['git','-C','/Users/gwilson/GitHub/tolproj','rev-parse','HEAD'], text=True).strip(),
        'probe_results': cases, 'standard_discovery': discovery, 'legacy_recovered': recovered,
        'historical_report_sha256': digest(report_path), 'historical_calibration_bindings': bindings,
        'authored_science_yaml': str(yaml_path), 'authored_science_yaml_sha256': digest(yaml_path),
        'selected_APTs': selected, 'pointing_cal_objs': authored['cal_objs'],
        'project_status': project['status'], 'project_sha256': digest(project_path),
        'new_science_low_level_configs': [str(p) for p in (PROJECT/'refactor/NGC4449/reduced').glob('citlali*.yaml')]}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
