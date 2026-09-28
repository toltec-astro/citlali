"""Resolve explicit original acquisition inputs without scientific processing.

No numerical processing, calibration issuance, or remote access. Inputs stay in place.
"""
import argparse,copy,json,platform,shutil,subprocess
from pathlib import Path
import netCDF4,numpy as np,yaml
from prepare_successor_networks import binding,csv_rows
from point_additive_diagnostics import save_json
from inspect_forward_map import require,digest

REPO=Path(__file__).resolve().parents[2]
def prepare(out, reference_path, input_root):
    require(not out.exists(), 'fresh output root required')
    r=json.loads(reference_path.read_text());base=reference_path.parent
    require(r['schema']=='citlali-bright-point-reference-v1' and r['mode']=='POINT', 'only bright POINT reference supported')
    require(r['subobservation']==0 and r['scan']==2, 'current saved MAP supports subobservation0/scan2 only')
    require(r['cycles']==10 and r['threshold']==3 and r['coverage_policy']=='high-coverage-informational','frozen downstream recipe differs')
    require(r['ptc']['method']=='observed-als' and isinstance(r['ptc']['rank'],int) and r['ptc']['rank']>0,'unsupported upstream PTC request')
    OBS=r['observation'];array=r['array'];memberships={'a1100':range(7),'a1400':range(7,11),'a2000':range(11,13)}
    require(array in memberships and r['networks'] and len(set(r['networks']))==len(r['networks']) and all(n in memberships[array] for n in r['networks']),'array/network selection differs')
    config=yaml.safe_load((base/r['effective_config']).read_text());require(len(config['inputs'])==1,'one exact acquisition required')
    require(config['inputs'][0]['meta']['name']==f"{OBS}_0_2",'configuration identity differs')
    for item in config['inputs'][0]['data_items']+config['inputs'][0]['cal_items']:
        if 'filepath' in item:
            p=Path(item['filepath']);require(not p.is_absolute() and '..' not in p.parts,'input paths must be relative to explicit root');item['filepath']=str(input_root/p)
    manifest=Path(next(x['filepath'] for x in config['inputs'][0]['cal_items'] if x['type']=='array_prop_table'))
    telescope=Path(next(x['filepath'] for x in config['inputs'][0]['data_items'] if x['meta']['interface']=='lmt'))
    inputs_manifest=json.loads((base/r['input_manifest']).read_text())
    for b in inputs_manifest:
        p=input_root/b['path'];require(p.is_file() and digest(p)==b['sha256'],'missing or changed declared original input: '+str(p))
    data=telescope.parent
    roles={r['role']:manifest.parent/r['relative_path'] for r in csv_rows(manifest)}
    sources=csv_rows(roles['sources']);apt=csv_rows(roles['apt'])
    dependencies={};timing=[];inventory=[];native_cells=0
    def bind(p,role):
        b=binding(p);dependencies[b['path']]=dict(**b,bytes=Path(p).stat().st_size,role=role,local=True,download_bytes=0);return b
    for p in [manifest,*roles.values(),manifest.with_suffix(manifest.suffix+'.sha256')]:
        if p.exists():bind(p,'canonical APT-v2 component/receipt')
    tel=bind(telescope,'original telescope')
    for item in config['inputs'][0]['data_items']:
        name=item['meta']['interface'];path=Path(item['filepath'])
        if name=='lmt':path=telescope
        item['filepath']=str(path)
        if not name.startswith('toltec'):
            bind(path,'auxiliary listed in config; HWPR inert with polarimetry disabled' if name=='hwpr' else 'original telescope');continue
        nw=int(name[6:]);raw=bind(path,'detector raw' if nw in r['networks'] else 'native timestamp-only segmentation input')
        with netCDF4.Dataset(path) as nc:
            require([int(nc['Header.Toltec.'+k][...]) for k in ('ObsNum','SubObsNum','ScanNum','RoachIndex')]==[OBS,0,2,nw],'foreign raw identity')
            rows,channels=nc['Data.Toltec.Is'].shape
            require(nc['Data.Toltec.Qs'].shape==(rows,channels),'paired channel shape')
            require([int(nc['Header.Toltec.TargSweep'+k][...]) for k in ('ObsNum','SubObsNum','ScanNum')]==r['tune'],'foreign tune association')
        s=next(r for r in sources if r['role']=='raw' and int(r['nw'])==nw)
        require(s['content_sha256']=='sha256:'+raw['sha256'] and int(s['byte_count'])==path.stat().st_size,'raw not exact canonical APT source')
        timing.append(dict(network=nw,**raw))
        if nw not in r['networks']:continue
        tunes=list(data.glob(f"toltec{nw}_{r['tune'][0]}_{r['tune'][1]:03d}_{r['tune'][2]:04d}_*_tune.txt"));require(len(tunes)==1,'ambiguous tune')
        tune=bind(tunes[0],'consumed processed KMP text; associated raw/processed NC already available but not required')
        s=next(r for r in sources if r['role']=='kmp' and int(r['nw'])==nw)
        require(s['content_sha256']=='sha256:'+tune['sha256'] and int(s['byte_count'])==tunes[0].stat().st_size,'KMP not exact canonical APT source')
        members=sorted([r for r in apt if int(r['nw'])==nw],key=lambda r:int(r['kids_tone']))
        require([int(r['kids_tone']) for r in members]==list(range(channels)) and all(int(r['array'])=={'a1100':0,'a1400':1,'a2000':2}[array] for r in members),'APT selected-array detector correspondence')
        native_cells+=rows*channels;inventory.append(dict(network=nw,raw=raw,tune=tune,rows=rows,channels=channels))
    require(set(r['networks']) <= {x['network'] for x in timing},'selected network missing timing')
    for item in config['inputs'][0]['cal_items']:
        if item['type']=='array_prop_table':item['filepath']=str(manifest)
    config['runtime']['output_dir']=str(out/'unused-legacy-output')
    out.mkdir(parents=True);inputs=out/'inputs';inputs.mkdir();freeze=out/'freeze';freeze.mkdir()
    ep=inputs/'effective-point.yaml';ep.write_text(yaml.safe_dump(config,sort_keys=False))
    ast=inputs/'ast-source-binding.json';save_json(ast,dict(schema='citlali-ast-motion-source-binding-v1',authority_policy_id='wp7-ast-scan-motion-v2',observation=OBS,subobservation=0,scan=2,telescope=tel))
    processing=inputs/'processing-request.json';save_json(processing,dict(schema='citlali-native-processing-request-v1',observation=OBS,subobservation=0,scan=2,realized=False,method='preserved-gap-grid-forced-10s-context32',effective_config_sha256=digest(ep),timing_inputs=timing,manifest=binding(manifest),telescope=tel,selected_point_config=bind(base/r['effective_config'],'explicit original acquisition configuration'),reference_config=bind(reference_path,'versioned bright-POINT recipe')))
    requests=[]
    for row in inventory:
        nw=row['network'];p=inputs/f'network{nw}-input.json'
        cfg=dict(schema='rtc-multidetector-experiment-v1',development_reference='bright-point-base-v1',observation=OBS,network=nw,raw=row['raw'],tune=row['tune'],manifest=binding(manifest),telescope=tel,effective_config=binding(ep),ast_acceptance=binding(ast),native_source='exact-raw-kids-v1',terminal_request='rtc-only',prepare_output_grid=True,speed_support_comparison='both',reassessment_selection=dict(intent='retain-development-candidate',authority='owner-bright-point-reference-2026-09-26',purpose='frozen bright POINT downstream reference',positive_rationale='accepted reference treatment on explicitly configured acquisition'),decision_apply=dict(policy='existing-authorities-with-unavailable-isolated-admission-v1',timing_inputs=timing,processing_request=binding(processing)),filter_plan=bind(base/r['rtc_plans'][array],'RTC owner filter plan'),detectors=[dict(channel=i,filter='lowpass') for i in range(row['channels'])],source_protection_authority='owner-bright-POINT-no-spatial-mask-2026-09-26')
        save_json(p,cfg);requests.append(dict(network=nw,input=binding(p)))
    save_json(inputs/'observation.json',dict(schema='citlali-observation-networks-v1',development_reference='bright-point-base-v1',observation=OBS,subobservation=0,scan=2,networks=requests))
    shutil.copyfile(base/r['map_owner'],inputs/'map-owner.json');bind(base/r['map_owner'],'MAP frame/beam/coefficient authority')
    # Current native dimensions, conservative serialized allocations, one live cycle.
    limits=r['resource_limits'];budget=dict(native_cells=native_cells,shared_upstream_allowance=256*native_cells,peak_ceiling_bytes=304*native_cells+limits['maximum_retained_bytes'],free_space_reserve_bytes=limits['free_space_reserve_bytes'],network_workers=limits['network_workers'],library_threads=1)
    require(shutil.disk_usage(out).free>budget['peak_ceiling_bytes']+budget['free_space_reserve_bytes'],'insufficient free space')
    save_json(freeze/'budget.json',budget)
    save_json(freeze/'dependencies.json',dict(inputs=list(dependencies.values()),native_inventory=inventory,download_bytes=0))
    r.update(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),reference=bind(reference_path,'versioned config'),input_root=str(input_root),dependencies=binding(freeze/'dependencies.json'),budget=binding(freeze/'budget.json'),environment=dict(python=__import__('sys').executable,platform=platform.platform()))
    save_json(freeze/'generated-inputs.json',[binding(p) for p in sorted(inputs.iterdir()) if p.is_file()]);r['generated_inputs']=binding(freeze/'generated-inputs.json')
    save_json(freeze/'prospective.json',r)
    print(json.dumps(dict(stage='preflight only',observation=OBS,array=array,networks=r['networks'],budget=budget),indent=2))
