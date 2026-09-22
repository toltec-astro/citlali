"""Prepare the selected POINT 152389 input, preserving successor treatment.

Inventory and source binding only; C++ realizes native timing, AST and support.
Without an explicit source-protection authority this prepares a non-runnable
request, allowing inventory work before the owner's scientific selection.
"""
import argparse
import copy
import json
from pathlib import Path
import netCDF4
import yaml
from prepare_successor_networks import SUPPLIED, binding, csv_rows, write
from ptc_spod import require


def prepare(point, reference_config, output, repository, protection=None):
    output.mkdir(parents=True, exist_ok=False)
    manifest = point / 'apts/v2/apt_152389_matched.apt-v2/manifest.ecsv'
    roles = {r['role']: manifest.parent / r['relative_path'] for r in csv_rows(manifest)}
    apt, sources = csv_rows(roles['apt']), csv_rows(roles['sources'])
    original = point / 'refactor/reduced/redu00/citlali_o152389_0_2_c1.yaml'
    point_config = yaml.safe_load(original.read_text())
    reference = yaml.safe_load(reference_config.read_text())
    require(reference['timestream']['chunking'] == dict(chunk_mode='duration', force_chunking=True, value=10),
            'reference processing intervals differ')
    selected = [i for i in point_config['inputs'] if i['meta']['name'] == '152389_0_2']
    require(len(selected) == 1, 'POINT requires one exact configuration input')
    config = copy.deepcopy(reference)
    config['inputs'] = copy.deepcopy(selected)
    config['runtime']['reduction_type'] = 'pointing'
    config['runtime']['output_dir'] = str(output / 'unused-legacy-output')
    # These paths identify input items; the successor verifies file digests
    # independently and never executes this effective config as a legacy run.
    for item in config['inputs'][0]['data_items']:
        item['filepath'] = str(point / 'data' / Path(item['filepath']).name)
    for item in config['inputs'][0]['cal_items']:
        if item['type'] == 'array_prop_table':
            item['filepath'] = str(manifest)
    effective = output / 'effective-point.yaml'
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    telescope = point / 'data/tel_toltec_2026-02-19_152389_00_0002.nc'
    tel = binding(telescope)
    require(tel['sha256'] == '2aec5f1f2be8c365127fcf275a02eb0c780eeda9494cd225e50e50c4386a9392',
            'selected POINT telescope differs')
    ast = output / 'ast-source-binding.json'
    write(ast, dict(schema='citlali-ast-motion-source-binding-v1',
        authority_policy_id='wp7-ast-scan-motion-v2', observation=152389,
        subobservation=0, scan=2, telescope=tel,
        meaning='source binding for existing AST owner; not a prior acceptance result'))
    processing = output / 'processing-request.json'
    timing, inventory = [], []
    for nw in SUPPLIED:
        raw = list((point / 'data').glob(f'toltec{nw}_152389_000_0002_*.nc'))
        tunes = list((point / 'data').glob(f'toltec{nw}_152389_000_0001_*_tune.txt'))
        require(len(raw) == len(tunes) == 1, 'ambiguous or missing raw/Tune input')
        raw, tune = binding(raw[0]), binding(tunes[0])
        for role, b in [('raw',raw),('kmp',tune)]:
            matches = [s for s in sources if s['role'] == role and int(s['nw']) == nw]
            require(len(matches) == 1 and matches[0]['content_sha256'] == 'sha256:'+b['sha256'] and
                    int(matches[0]['byte_count']) == Path(b['path']).stat().st_size, 'APT source binding differs')
        with netCDF4.Dataset(raw['path']) as nc:
            scope = [int(nc['Header.Toltec.'+k][...]) for k in ('ObsNum','SubObsNum','ScanNum','RoachIndex')]
            require(scope == [152389,0,2,nw], 'raw header scope differs')
            rows, channels = nc['Data.Toltec.Is'].shape
        members = sorted((r for r in apt if int(r['nw']) == nw), key=lambda r:int(r['kids_tone']))
        require([int(r['kids_tone']) for r in members] == list(range(channels)), 'incomplete APT detector relation')
        arrays = {int(r['array']) for r in members}
        require(len(arrays) == 1, 'network spans arrays')
        name = ('a1100','a1400','a2000')[arrays.pop()]
        timing.append(dict(network=nw, **raw))
        inventory.append(dict(network=nw,array=name,raw=raw,tune=tune,native_rows=rows,detectors=channels))
    write(processing, dict(schema='citlali-native-processing-request-v1',
        observation=152389, subobservation=0, scan=2, realized=False,
        method='preserved-gap-grid-forced-10s-context32',
        effective_config_sha256=binding(effective)['sha256'],
        timing_inputs=timing, manifest=binding(manifest), telescope=tel,
        selected_point_config=binding(original), reference_config=binding(reference_config),
        meaning='request only; native C++ owners realize and publish actual support'))
    requests = []
    for item in inventory:
        nw = item['network']
        cfg = dict(schema='rtc-multidetector-experiment-v1', observation=152389, network=nw,
            raw=item['raw'], tune=item['tune'], manifest=binding(manifest), telescope=tel,
            effective_config=binding(effective), ast_acceptance=binding(ast),
            native_source='exact-raw-kids-v1', terminal_request='rtc-only', prepare_output_grid=True,
            speed_support_comparison='both',
            reassessment_selection=dict(intent='retain-development-candidate',
                authority='owner-POINT-152389-first-MAP-2026-09-22',
                purpose='recognizable central source through the current successor treatment'),
            decision_apply=dict(policy='existing-authorities-with-unavailable-isolated-admission-v1',
                timing_inputs=timing,processing_request=binding(processing)),
            filter_plan=binding(repository/'validation/rtc_development_lowpass_2026-09-19'/f"{item['array']}.json"),
            detectors=[dict(channel=d,filter='lowpass') for d in range(item['detectors'])])
        if protection is not None:
            cfg['source_protection_authority'] = protection
        path = output / f'network{nw}-input.json'
        write(path,cfg)
        requests.append(dict(network=nw,input=binding(path)))
    observation = dict(schema='citlali-observation-networks-v1',observation=152389,subobservation=0,scan=2,networks=requests)
    write(output/'observation.json',observation)
    write(output/'observation-network0.json',dict(observation,networks=requests[:1]))
    write(output/'inventory.json',dict(schema='citlali-point-forward-inventory-v1',observation=152389,
        original_point_config=binding(original),reference_config=binding(reference_config),networks=inventory,
        absent_networks=[6,10],source_protection_selected=protection is not None,
        upstream_numerical_products='not yet executed'))
    return inventory

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--point',type=Path,required=True)
    p.add_argument('--reference-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-protection-authority')
    a=p.parse_args()
    rows=prepare(a.point.resolve(),a.reference_config.resolve(),a.output.resolve(),Path(__file__).resolve().parents[2],a.source_protection_authority)
    print(json.dumps([{k:r[k] for k in ('network','array','native_rows','detectors')} for r in rows],indent=2))
