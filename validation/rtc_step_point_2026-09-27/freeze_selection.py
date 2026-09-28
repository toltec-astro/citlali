"""Freeze the bounded evidence-selected acquisition and original dependencies, no reduction."""
from pathlib import Path
import json,copy,hashlib,shutil,sys
import yaml,netCDF4
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_additive_diagnostics import save_json
from inspect_forward_map import digest,require
from point_fruit_trajectory import bound
R=Path(__file__).resolve().parent;dest=R/'selected-inputs';dest.mkdir(exist_ok=True)
root=Path('/Users/gwilson/work_toltec/local_data');data=root/'2025-C1-COM-01/data'
prior=root/'citlali-validation/v2/science/pointings/reduced/citlali_o152389_0_2_c8.yaml'
configs=yaml.safe_load(prior.read_text());selected=next(x for x in configs['inputs'] if x['meta']['name']=='152420_0_2')
base=W/'data/development/bright_point_base_v1';cfg=yaml.safe_load((base/'point_152430.yaml').read_text());cfg['inputs']=[copy.deepcopy(selected)]
manifest=root/'citlali-validation/v2/science/apts/v2/apt_152420_matched.apt-v2/manifest.ecsv';files={};headers=[]
def add(p,role):
 require(p.is_file(),'missing original '+str(p));files[str(p)]=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p),role=role)
for item in cfg['inputs'][0]['data_items']:
 name=item['meta']['interface']
 p=data/Path(item['filepath']).name
 if name=='lmt':p=data/'tel_toltec_2026-02-19_152420_00_0002.nc'
 item['filepath']=str(p.relative_to(root));add(p,'original '+name)
 if name.startswith('toltec'):
  nw=int(name[6:])
  with netCDF4.Dataset(p) as nc:
   ids=[int(nc['Header.Toltec.'+k][...]) for k in ('ObsNum','SubObsNum','ScanNum','RoachIndex')];tune=[int(nc['Header.Toltec.TargSweep'+k][...]) for k in ('ObsNum','SubObsNum','ScanNum')]
   require(ids==[152420,0,2,nw] and tune==[152420,0,1],'actual raw/tune identity')
   headers.append(dict(network=nw,identity=ids,tune=tune,shape=list(nc['Data.Toltec.Is'].shape)))
for item in cfg['inputs'][0]['cal_items']:
 if item['type']=='array_prop_table':item['filepath']=str(manifest.relative_to(root))
for p in manifest.parent.iterdir():
 if p.is_file():add(p,'selected canonical APT/calibration component')
for nw in (11,12):
 tunes=list(data.glob(f'toltec{nw}_152420_000_0001_*_tune.txt'));require(len(tunes)==1,'exact tune absent/ambiguous');add(tunes[0],'selected exact associated KMP tune')
(dest/'point_152420.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));save_json(dest/'inputs.json',list(files.values()))
ref=json.loads((base/'reference.json').read_text());ref.update(observation=152420,tune=[152420,0,1],effective_config='point_152420.yaml',input_manifest='inputs.json',map_owner='map-owner.json',rtc_plans={a:a+'-rtc.json' for a in ('a1100','a1400','a2000')})
for name in ['map-owner.json','a1100-rtc.json','a1400-rtc.json','a2000-rtc.json']:shutil.copyfile(base/name,dest/name)
save_json(dest/'reference.json',ref)
disc=json.loads((R/'CANDIDATE_DISCOVERY.json').read_text());e=[x for x in disc['records'] if x['obsnum']==152420]
save_json(R/'SELECTION.json',dict(schema='rtc-step-point-selection-v1',selected=[152420,0,2],array='a2000',networks=[11,12],selection='development challenge; frozen before new final maps or full reductions',rationale='Twelve reviewed detector responses at three separated epochs show finite transitions and sustained post levels through the retained 1.5s window (nw11 survey scans4/10 and nw12scan9); fourth nw12scan2 has a return/short plateau. More original-domain step instances than the nine current admitted groups is plausible, not established by directly comparable production counts yet.',candidate_dispositions={'152420':'selected: clearest sustained original pre/post levels in both networks; multiple separated epochs','152432':'not selected: substantial transitions but closer returns and overlapping context in reviewed examples','152434':'not selected: more recovery/drift and shorter plateaus in reviewed examples'},original_evidence=bound(R/'candidate-original-excerpts.npz'),discovery=bound(R/'CANDIDATE_DISCOVERY.json'),reviewed_lower_bound=dict(persistent_detector_instances=12,distinct_times=3,detector_seconds_original_APT_usable=sum(x['detector_seconds_recv'] for x in [e[0],e[2]]),caveat='purposive original-IQ sample, not total burden or comparison of identical detectors/production detection rates; four simultaneous detectors do not count as four time events'),headers=headers,reference=bound(dest/'reference.json'),input_manifest=bound(dest/'inputs.json'),input_bytes=sum(v['bytes'] for v in files.values()),config_parent=bound(prior),calibration='own152420 canonical APT/tune; CAL atmosphere from own original TEL; same accepted CAL-after-RTC method',readiness='all declared original dependencies local; no Unity recovery',acquisition_parts='one raw scan2 part per available interface; selected nw11/12 complete; original dataset has nw0-5,7-9,11-12 timing inputs',full_reduction_started=False))
print('selected152420; bound',len(files),'inputs',sum(v['bytes'] for v in files.values()),'bytes')
