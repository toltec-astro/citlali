"""Reconstruct POINT152420 factors through recorded source associations.
Read-only source audit: no canonical issuance or observation reduction.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import subprocess
import yaml
import numpy as np
from astropy.table import Table

D = Path(__file__).resolve().parent
L = Path('/Users/gwilson/work_toltec/local_data')
H = L/'2025-C1-COM-01'
V = L/'citlali-validation/v2'
P = V/'science/apts/v2/apt_152420_matched.apt-v2'
HB = L/'_apt_library/raw/3c273/152307/apt_commissioning_beammap_152307_citlali.ecsv'
CLI = D.parents[1]/'build/bin/citlali'

def bound(p):
    p = Path(p)
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def rows(p):
    # Canonical v2 is comma ECSV with lowercase bools, not Astropy's bool dialect.
    return list(csv.DictReader(x for x in p.read_text().splitlines() if not x.startswith('#')))

def keyed(items, key):
    result = {}
    for row in items:
        k = key(row)
        if k in result:
            raise ValueError(f'ambiguous identity: {k}')
        result[k] = row
    return result

def quantiles(values):
    return np.quantile(values, [0, .1, .5, .9, 1]).tolist()

def main():
    paths = {role:next(P.glob(f'*.{role}.ecsv')) for role in
             ('apt', 'baseline-apt', 'relation', 'fields', 'exceptions', 'baseline-manifest')}
    h = Table.read(H/'apts/hero/apt_152420_matched.ecsv')
    original = Table.read(H/'apts/apt_152420_matched.ecsv')
    hb = Table.read(HB)
    hist = keyed(h, lambda x:(int(x['nw']), int(x['kids_tone'])))
    successor = keyed(rows(paths['apt']), lambda x:(int(x['nw']), int(x['kids_tone'])))
    relation = keyed(rows(paths['relation']), lambda x:int(x['output_uid']))
    baseline = keyed(rows(paths['baseline-apt']), lambda x:int(x['uid']))
    assert hist.keys() == successor.keys()
    assert np.array_equal(h['flxscale'], original['flxscale'], equal_nan=True)
    assert not rows(paths['exceptions'])
    # Check the recorded artifact-local historical source association, not
    # equality of channel IDs between the two different beammap observations.
    historic_mapped = 0
    for x in original:
        index = int(x['det_id_right'])
        if index < 0:
            continue
        parent = hb[index]
        for field in ('uid', 'nw', 'amp', 'flxscale', 'f_out', 'x_t', 'y_t'):
            assert x[field] == parent[field], (index, field)
        historic_mapped += 1
    request_path = V/'science/apts/v2/requests/apt_152420_matched_match_request.json'
    request = json.loads(request_path.read_text())
    matches = keyed(request['matches'], lambda x:int(x['target_uid']))
    all_matched = 0
    for key, x in successor.items():
        y = hist[key]
        for field in ('array', 'tone_freq', 'kids_f_out'):
            assert float(x[field]) == float(y[field]), (key, field)
        rel = relation[int(x['uid'])]
        if rel['disposition'] != 'matched':
            assert x['flxscale'] == ''
            continue
        parent = baseline[int(rel['seed_uid'])]
        assert rel['seed_uid'] == matches[int(rel['target_uid'])]['seed_uid']
        assert float(parent['flxscale']) == float(x['flxscale'])
        assert int(parent['nw']) == key[0]
        all_matched += 1
    source_provenance = V/'beammaps/3c273/reduced/redu00/beammap_provenance.yaml'
    photometry = yaml.safe_load(source_provenance.read_text())['observations'][0]['photometry']
    sflux = next(float(x['value_mJy']) for x in photometry['fluxes'] if x['array_name']=='a1100')
    assert sflux == 3127.2160728457216
    old_audit = json.loads((D/'results/calibration-audit.json').read_text())
    examples = {(x['network'],x['channel']): x for x in old_audit['representative_native_conversions']}
    detailed = []
    summaries = []
    for nw in range(5):
        group = []
        for key, x in successor.items():
            y = hist[key]
            if key[0] != nw or x['flag'] != '0' or y['flag'] != 0:
                continue
            if not x['flxscale'] or float(x['flxscale']) <= 0 or not y['flxscale'] > 0:
                continue
            rel = relation[int(x['uid'])]
            assert rel['disposition'] == 'matched' and rel['is_good_match'] == 'true'
            hs = hb[int(y['det_id_right'])]
            ss = baseline[int(rel['seed_uid'])]
            hflux = float(hb.meta['a1100_flux'][0])
            hpredict = hflux / float(hs['amp'])
            spredict = sflux / float(ss['cal_amp'])
            hf, sf = float(y['flxscale']), float(x['flxscale'])
            error = (spredict/hpredict)/(sf/hf)-1
            # Legacy decimal serialization and three IEEE double operations:
            # eight epsilons covers the rounding bound, not scientific agreement.
            assert abs(hpredict/hf-1) <= 8*np.finfo(float).eps
            assert spredict == sf
            assert abs(error) <= 8*np.finfo(float).eps
            rec = dict(network=nw, channel=key[1], historical_seed_uid=int(hs['uid']),
                       historical_seed_row=int(y['det_id_right']), successor_seed_uid=int(ss['uid']),
                       successor_output_uid=int(x['uid']), historical_amp_x=float(hs['amp']),
                       successor_cal_amp_x=float(ss['cal_amp']), successor_gaussian_amp_x=float(ss['amp']),
                       successor_cal_amp_method=int(ss['cal_amp_method']), historical_factor=hf,
                       successor_factor=sf, factor_ratio=sf/hf, reconstructed_ratio=spredict/hpredict,
                       relative_ratio_residual=error)
            group.append(rec)
            if key in examples:
                sample=examples[key]
                assert sample['historical_factor']==hf and sample['successor_factor']==sf
                rec=dict(rec, native_sample=sample,
                    independently_reconstructed_historical_sample=sample['native_filtered_x']*hpredict,
                    successor_source_only_sample=sample['native_filtered_x']*spredict,
                    atmosphere_scope='Target atmosphere retained from prior actual-CAL audit; not an independent passband calculation')
                detailed.append(rec)
        summaries.append(dict(network=nw, supported_common_rows=len(group),
            factor_ratio_quantiles=quantiles([x['factor_ratio'] for x in group]),
            successor_cal_amp_over_historical_amp_quantiles=quantiles([x['successor_cal_amp_x']/x['historical_amp_x'] for x in group]),
            successor_cal_amp_over_own_gaussian_amp_quantiles=quantiles([x['successor_cal_amp_x']/x['successor_gaussian_amp_x'] for x in group]),
            empirical_template_rows=sum(x['successor_cal_amp_method']==1 for x in group),
            max_abs_relative_ratio_residual=max(abs(x['relative_ratio_residual']) for x in group)))
    validation = []
    for manifest in [P/'manifest.ecsv', V/'beammaps/3c273/reduced/redu00/148670/raw/apt_commissioning_beammap_148670_citlali.apt-v2/manifest.ecsv']:
        req=dict(protocol='citlali-canonical-apt-protocol-v2',request_id='lineage-audit',operation='validate-bundle-v2',payload=dict(root_manifest=str(manifest)))
        run=subprocess.run([str(CLI),'--canonical-apt-contract-v2'],input=json.dumps(req)+'\n',text=True,capture_output=True,check=True)
        response=json.loads(run.stdout)
        assert response.get('status')=='ok',response
        compact = {k:v for k,v in response['result'].items() if k in
                   ('product_kind','profile','artifact','observation','transport','parser_count','issuance_class')}
        validation.append(dict(manifest=bound(manifest),status=response['status'],result=compact,
                               stdout_sha256=hashlib.sha256(run.stdout.encode()).hexdigest(),
                               stderr=run.stderr))
    inputs = list(paths.values())+[HB,H/'apts/apt_152420_matched.ecsv',H/'apts/hero/apt_152420_matched.ecsv',request_path,
        V/'science/logs/matchapts-v2.log', V/'science/logs/matchapts-v2-63731641.out',
        V/'science/apts/v2/apt_152420_matched_match_report.json',
        V/'science/project.yaml', L/'_apt_library/library_manifest.json',
        H/'pointings/72_reduce.yaml', H/'apts/hero/hero_apply_summary.yaml',
        V/'beammaps/3c273/reduced/redu00/beammap_provenance.yaml',
        V/'beammaps/3c273/reduced/redu00/citlali_o148670_0_2_c1.yaml',D/'results/calibration-audit.json']
    producer_revision = '5fc334cd3d9313106246b0de68b0f7a07ab86909'
    producer_sources = []
    for file, expected in [('tolproj/matched_apt_v2.py', request['software_revision'].split('sha256.')[1]),
                           ('tolproj/legacy_scripts/make_matched_apt.py', request['matcher']['implementation_sha256'].split(':')[1])]:
        content = subprocess.check_output(['git','-C','/Users/gwilson/GitHub/tolproj','show',producer_revision+':'+file])
        assert hashlib.sha256(content).hexdigest() == expected
        producer_sources.append(dict(repository='/Users/gwilson/GitHub/tolproj', revision=producer_revision,path=file,sha256=expected))
    r = L/'citlali-validation/successor-rtc-point-transfer-completion-20260930/B-REF'
    runtime_paths = [r/'freeze/prospective.json',r/'freeze/generated-inputs.json',r/'upstream/RUN.json',
                     r/'upstream/development.yaml']
    generated = json.loads((r/'freeze/generated-inputs.json').read_text())
    for item in generated:
        assert bound(item['path'])['sha256'] == item['sha256']
    runtime = json.loads((r/'upstream/RUN.json').read_text())
    runtime = {k:v for k,v in runtime.items() if k in ['command','executable_sha256','version','input_sha256','workers','thread_environment']}
    result=dict(schema='point152420-apt-lineage-audit-v1',inputs=[bound(p) for p in inputs],
        producer_sources=producer_sources,runtime_inputs=[bound(p) for p in runtime_paths],runtime=runtime,
        source_flux_ratio=sflux/float(hb.meta['a1100_flux'][0]),
        counts=dict(target_rows=len(h),historical_recorded_matches=historic_mapped,historical_no_recorded_match=len(h)-historic_mapped,
                    successor_recorded_matches=all_matched,successor_unmatched=len(h)-all_matched,representatives=len(detailed)),
        identity_scope='Same 152420/0/2 acquisition and 152420/0/1 Tune. Each target has its own recorded source association; no cross-observation channel identity assertion.',
        tolerance='8 binary64 epsilons for reconstructed ratios; exact selected-parent factors and successor S/cal_amp; legacy source division up to 7e-16 relative.',
        historical_limit='Recorded row association and copied fields verified; original matcher executable/ambiguity decision receipt unavailable. Do not promote it to a v2 verified relation.',
        per_network=summaries,representatives=detailed,canonical_validation=validation,validation_cli=bound(CLI),
        validation_cli_version=subprocess.check_output([str(CLI),'--version'],text=True),
        new_observation_starts=0)
    (D/'results/apt-lineage-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(counts=result['counts'],source_flux_ratio=result['source_flux_ratio'],per_network=summaries),indent=2))

if __name__ == '__main__':
    main()
