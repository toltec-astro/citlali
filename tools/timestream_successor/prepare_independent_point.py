"""Prepare exact local inputs for owner-selected POINT152430/a2000 only.

No numerical processing, calibration issuance, or remote access. Inputs stay in place.
"""
import argparse,copy,json,platform,shutil,subprocess
from pathlib import Path
import netCDF4,numpy as np,yaml
from prepare_successor_networks import binding,csv_rows
from point_additive_diagnostics import save_json
from inspect_forward_map import require,digest

REPO=Path(__file__).resolve().parents[2]
LOCAL=Path('/Users/gwilson/work_toltec/local_data')
OLD=LOCAL/'citlali-validation/development-runs'
OBS=152430

def prepare(out):
    require(not out.exists(),'fresh campaign destination required')
    data=LOCAL/'2025-C1-COM-01/data'
    manifest=LOCAL/'citlali-validation/v2/science/apts/v2/apt_152430_matched.apt-v2/manifest.ecsv'
    original=LOCAL/'citlali-validation/v2/science/pointings/reduced/citlali_o152389_0_2_c8.yaml'
    reference=OLD/'successor-point-152389-forward-map-20260922/inputs/effective-point.yaml'
    selected=[x for x in yaml.safe_load(original.read_text())['inputs'] if x['meta']['name']=='152430_0_2']
    require(len(selected)==1,'exact selected observation configuration absent')
    config=yaml.safe_load(reference.read_text());config['inputs']=copy.deepcopy(selected)
    require(config['timestream']['chunking']==dict(chunk_mode='duration',force_chunking=True,value=10),'changed processing recipe')
    roles={r['role']:manifest.parent/r['relative_path'] for r in csv_rows(manifest)}
    sources=csv_rows(roles['sources']);apt=csv_rows(roles['apt'])
    dependencies={};timing=[];inventory=[];native_cells=0
    def bind(p,role):
        b=binding(p);dependencies[b['path']]=dict(**b,bytes=Path(p).stat().st_size,role=role,local=True,download_bytes=0);return b
    for p in [manifest,*roles.values(),manifest.with_suffix(manifest.suffix+'.sha256')]:
        if p.exists():bind(p,'canonical APT-v2 component/receipt')
    telescope=data/'tel_toltec_2026-02-19_152430_00_0002.nc';tel=bind(telescope,'original telescope')
    require(tel['sha256']=='f6ac27248692af558a16af94f267a84b344e93f30d2248516d4388b9b25e5f19','foreign telescope')
    for item in config['inputs'][0]['data_items']:
        name=item['meta']['interface'];path=data/Path(item['filepath']).name
        if name=='lmt':path=telescope
        item['filepath']=str(path)
        if not name.startswith('toltec'):
            bind(path,'auxiliary listed in config; HWPR inert with polarimetry disabled' if name=='hwpr' else 'original telescope');continue
        nw=int(name[6:]);raw=bind(path,'detector raw' if nw in (11,12) else 'native timestamp-only segmentation input')
        with netCDF4.Dataset(path) as nc:
            require([int(nc['Header.Toltec.'+k][...]) for k in ('ObsNum','SubObsNum','ScanNum','RoachIndex')]==[OBS,0,2,nw],'foreign raw identity')
            rows,channels=nc['Data.Toltec.Is'].shape
            require(nc['Data.Toltec.Qs'].shape==(rows,channels),'paired channel shape')
            require([int(nc['Header.Toltec.TargSweep'+k][...]) for k in ('ObsNum','SubObsNum','ScanNum')]==[OBS,0,1],'foreign tune association')
        s=next(r for r in sources if r['role']=='raw' and int(r['nw'])==nw)
        require(s['content_sha256']=='sha256:'+raw['sha256'] and int(s['byte_count'])==path.stat().st_size,'raw not exact canonical APT source')
        timing.append(dict(network=nw,**raw))
        if nw not in (11,12):continue
        tunes=list(data.glob(f'toltec{nw}_152430_000_0001_*_tune.txt'));require(len(tunes)==1,'ambiguous tune')
        tune=bind(tunes[0],'consumed processed KMP text; associated raw/processed NC already available but not required')
        s=next(r for r in sources if r['role']=='kmp' and int(r['nw'])==nw)
        require(s['content_sha256']=='sha256:'+tune['sha256'] and int(s['byte_count'])==tunes[0].stat().st_size,'KMP not exact canonical APT source')
        members=sorted([r for r in apt if int(r['nw'])==nw],key=lambda r:int(r['kids_tone']))
        require([int(r['kids_tone']) for r in members]==list(range(channels)) and all(int(r['array'])==2 for r in members),'APT a2000 detector correspondence')
        native_cells+=rows*channels;inventory.append(dict(network=nw,raw=raw,tune=tune,rows=rows,channels=channels))
    require(sorted(x['network'] for x in timing)==[0,1,2,3,4,5,7,8,9,11,12],'saved timing population differs')
    for item in config['inputs'][0]['cal_items']:
        if item['type']=='array_prop_table':item['filepath']=str(manifest)
    config['runtime']['output_dir']=str(out/'unused-legacy-output')
    out.mkdir(parents=True);inputs=out/'inputs';inputs.mkdir();freeze=out/'freeze';freeze.mkdir()
    ep=inputs/'effective-point.yaml';ep.write_text(yaml.safe_dump(config,sort_keys=False))
    ast=inputs/'ast-source-binding.json';save_json(ast,dict(schema='citlali-ast-motion-source-binding-v1',authority_policy_id='wp7-ast-scan-motion-v2',observation=OBS,subobservation=0,scan=2,telescope=tel))
    processing=inputs/'processing-request.json';save_json(processing,dict(schema='citlali-native-processing-request-v1',observation=OBS,subobservation=0,scan=2,realized=False,method='preserved-gap-grid-forced-10s-context32',effective_config_sha256=digest(ep),timing_inputs=timing,manifest=binding(manifest),telescope=tel,selected_point_config=bind(original,'saved observation-specific configuration'),reference_config=bind(reference,'accepted numerical recipe')))
    requests=[]
    for row in inventory:
        nw=row['network'];p=inputs/f'network{nw}-input.json'
        cfg=dict(schema='rtc-multidetector-experiment-v1',observation=OBS,network=nw,raw=row['raw'],tune=row['tune'],manifest=binding(manifest),telescope=tel,effective_config=binding(ep),ast_acceptance=binding(ast),native_source='exact-raw-kids-v1',terminal_request='rtc-only',prepare_output_grid=True,speed_support_comparison='both',reassessment_selection=dict(intent='retain-development-candidate',authority='owner-independent-POINT-2026-09-26',purpose='frozen-method independent a2000 comparison',positive_rationale='same accepted treatment on independent acquisition'),decision_apply=dict(policy='existing-authorities-with-unavailable-isolated-admission-v1',timing_inputs=timing,processing_request=binding(processing)),filter_plan=binding(REPO/'validation/rtc_development_lowpass_2026-09-19/a2000.json'),detectors=[dict(channel=i,filter='lowpass') for i in range(row['channels'])],source_protection_authority='owner-152430-frozen-POINT-no-spatial-mask-2026-09-26')
        save_json(p,cfg);requests.append(dict(network=nw,input=binding(p)))
    save_json(inputs/'observation.json',dict(schema='citlali-observation-networks-v1',observation=OBS,subobservation=0,scan=2,networks=requests))
    saved=OLD/'successor-point-152389-guarded-feedback-comparison-20260925/freeze/prospective.json';prior=json.loads(saved.read_text());th=Path(prior['threshold_file']['path']);require(digest(th)==prior['threshold_file']['sha256'],'changed full thresholds')
    shutil.copyfile(th,freeze/'thresholds.json');bind(th,'full precision frozen threshold authority')
    mapowner=OLD/'successor-point-152389-forward-map-20260922/owner-binding.json';shutil.copyfile(mapowner,inputs/'map-owner.json');bind(mapowner,'unchanged MAP frame/nominalbeam/uniform family authority')
    # Conservative byte model bounds current serialized allocations: all native
    # raw cells, even though RTC decimates a2000 by2. No assumption of thinning.
    budget=dict(missing_input_bytes=0,native_cells=native_cells,shared_upstream_allowance=256*native_cells,initial_four_maps_and_injected_parents=4*48*native_cells,reference_cycle_working_allowance=30*32*native_cells,concurrent_cycle_allowance=12*48*native_cells,retained_result_allowance=1_000_000_000,native_network_workers=2,maximum_branch_workers=12,library_threads=1)
    budget['peak_ceiling_bytes']=sum(budget[k] for k in ['shared_upstream_allowance','initial_four_maps_and_injected_parents','reference_cycle_working_allowance','concurrent_cycle_allowance','retained_result_allowance'])
    budget['free_space_reserve_bytes']=2_000_000_000;budget['qualification']='campaign-only ceiling derived from actual raw dimensions and conservative serialized arrays; not a global quota; first REF measured before parallel scheduling'
    require(shutil.disk_usage(out).free>budget['peak_ceiling_bytes']+budget['free_space_reserve_bytes'],'insufficient disk for campaign ceiling')
    save_json(freeze/'dependencies.json',dict(observation=OBS,selected_before_reduction=True,inputs=list(dependencies.values()),unique_input_bytes=sum(r['bytes'] for r in dependencies.values()),download_bytes=0,inventory=inventory,raw_package='complete against saved eleven-network input inventory',a2000_complete=True,reduction_ready='native semantic guardians still gate processing',archive_provenance='saved config /work/toltec/citlali-validation/v2/science/; no live Unity verification',transfer_commands=[],selection='own later tune and complete canonical-v2 route; Januarylegacy-only APT candidates not selected; no policy screening'))
    save_json(freeze/'budget.json',budget)
    save_json(freeze/'prospective.json',dict(observation=OBS,array='a2000',maximum_trajectories=15,maximum_feedback_applications=150,maximum_initial_physical_inputs=4,maximum_ideal_controls=9,maximum_endpoint_covariances=3,cycles=10,policies=['BASE-POINT','CORR','DIAG'],branches=['REF','C-core-plus','C-core-minus','C-shoulder','D-seed-shoulder'],thresholds=prior['thresholds']['a2000'],threshold_file=binding(freeze/'thresholds.json'),reference_tolerances=prior['tolerances'],accounting_atol=1e-8,dependencies=binding(freeze/'dependencies.json'),budget=binding(freeze/'budget.json'),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),selection_reason='independent later acquisition and own tune, canonicalv2 ready, prior source summary bright; shared calibration errors may remain',environment=dict(python=__import__('sys').executable,platform=platform.platform()),method='same scientific owners; new M0 operands frozen once; no transferred noise/covariance/masks or pixelIDs; fixed thresholds portability does not calibrate false admission rate',stage='input preflight only; realized M0 operands pending'))
    print(json.dumps(budget,indent=2))


def prospective_details(out):
    import importlib.metadata as metadata
    from point_additive import NoiseRequest
    from dataclasses import asdict
    p=json.loads((out/'freeze/prospective.json').read_text())
    p['coverage_policy']='high-coverage-informational'
    p['method_change']='owner-authorized removal of upper coverage validity veto for this diagnostic route; numerical amplitude continuation unchanged; not unchanged-method replay'
    p['preparation_base_revision']=p.pop('source_revision',p.get('preparation_base_revision'))
    generated=[binding(f) for f in sorted((out/'inputs').iterdir()) if f.is_file()]
    generated.append(binding(REPO/'validation/rtc_development_lowpass_2026-09-19/a2000.json'))
    save_json(out/'freeze/generated-inputs.json',generated);p['generated_inputs']=binding(out/'freeze/generated-inputs.json')
    p['scientific_recipe']=dict(guard=dict(method='POINT-EXPECTED-v1',optimizer_calls=1,sigma_floor_arcsec=1.5,centroid_radius_arcsec=20,strict_amplitude_snr=10,weights='identity',bootstrap='one geometry-valid attempt, then C0, then complete same fit acceptance'),C0=dict(request=asdict(NoiseRequest(mask_fwhm=4.,high_coverage_usable=True)),stamp_size=49,geometry='singleton available M0 fit; freeze full mask/support and WCS'),BASE_amplitude=dict(request=asdict(NoiseRequest(mask_fwhm=3.,high_coverage_usable=True)),zero_lag_normalization='mean squared standardized background; no second spatial covariance'),PTC=dict(rank=10,max_iterations=100,relative_objective_tolerance=1e-5,relative_rank_tolerance=1e-10,method='observed-entry ALS, fresh residual-specific per-network segment; no historical failure exceptions'),MAP=dict(pixel_arcsec=2,nominal_beam_arcsec=10,frame='FK5 J2000',uniform_gamma=1,coverage_cut=.5,projection='native containing pixel'),feedback=dict(radius_arcsec=40,gain=1,replacement=True,brightness='measured minus own fitted constant',boundary='inherited candidate_guard'),shapes=dict(authority='point_response_cd.shapes',reference='BASE-POINT T10',M0_major=True,amplitudes=[.1,-.1,.05],shoulder_width_factor=2,shoulder_positive_east_offset_M0_major=.5,taper_radii_arcsec=[30,36]),endpoint=dict(maximum=3,source_mask='frozen M0 full mask and support intersect actual support',propagation='seven-parameter local sensitivity with fitted-constant response for template/apertures',unsupported='unresolved'),stops=['missing dependency','unsupported required covariance or projection','new PTC failure','accounting mismatch','nonfinite required operand','expected-source rejection','inherited feedback boundary'])
    p['environment']['packages']={n:metadata.version(n) for n in ('numpy','scipy','astropy','netCDF4','PyYAML','psutil')}
    p['environment']['numeric_threads']=1;p['environment']['native_network_workers']=2
    p['execution_identity']='clean exact commit and rebuilt native binary SHA are authenticated by independent review JSON before every stage'
    save_json(out/'freeze/prospective.json',p)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output.resolve());prospective_details(a.output.resolve())
