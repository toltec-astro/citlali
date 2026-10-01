"""Bounded local inventory of the acquisition-defined152385–152387 OOF sequence."""
from pathlib import Path
import json,gzip,hashlib,csv
import numpy as np
import yaml
from netCDF4 import Dataset
D=Path(__file__).resolve().parent;W=D.parents[1];L=Path('/Users/gwilson/work_toltec/local_data');V=L/'citlali-validation/v2'

def bound(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def val(v):
 a=np.ma.asarray(v[:]);a=a.filled(b'' if a.dtype.kind=='S' else np.nan)
 if a.dtype.kind=='S':return b''.join(a.reshape(-1)).decode(errors='replace').strip('\x00 ')
 return a.item() if a.size==1 else a.tolist()
def rows(p):return list(csv.DictReader(x for x in p.read_text().splitlines() if not x.startswith('#')))
def main():
 census_path=W/'validation/local_data_rtc_ptc_purge_2026-09-26/OBSERVATIONS.json.gz';census=json.loads(gzip.decompress(census_path.read_bytes()));selected=[]
 for obs in (152385,152386,152387):
  c=next(x for x in census if x['identity']==[obs,0,1]);p=next((L/'citlali-validation/v1/oof/data').glob(f'tel*{obs}*nc'))
  with Dataset(p) as nc:
   keys=[k for k in nc.variables if (k.startswith('Header.M2.Z') or k in ('Header.Dcs.ObsNum','Header.Dcs.SubObsNum','Header.Dcs.ScanNum','Header.Dcs.ObsPgm','Header.Source.SourceName','Header.Telescope.ElActPos','Header.TimePlace.UTDate'))]
   tel=dict(binding=bound(p),headers={k:dict(value=val(nc[k]),unit=getattr(nc[k],'units',None)) for k in keys})
   tk=[k for k in nc.variables if k.startswith('Data.') and ('Time' in k or 'ElAct' in k or 'TelEl' in k)]
   tel['stream_extents']={k:dict(shape=list(nc[k].shape),unit=getattr(nc[k],'units',None),min=float(np.ma.min(nc[k][:])),max=float(np.ma.max(nc[k][:]))) for k in tk if nc[k].dtype.kind in 'fi' and nc[k].size>1}
  bundle=V/f'oof/apts/v2/apt_{obs}_matched.apt-v2';components={x['role']:bundle/x['relative_path'] for x in rows(bundle/'manifest.ecsv')};sources=rows(components['sources']);raw=[]
  for path in c['selected_raw_paths']:
   f=L/path
   with Dataset(f) as nc:
    nw=int(nc['Header.Toltec.RoachIndex'][:]);shape=nc['Data.Toltec.Is'].shape;identity=[int(nc['Header.Toltec.'+k][:]) for k in ('ObsNum','SubObsNum','ScanNum')];tune=[int(nc['Header.Toltec.TargSweep'+k][:]) for k in ('ObsNum','SubObsNum','ScanNum')];assert identity==[obs,0,1]
   source=next(x for x in sources if x['role']=='raw' and int(x['nw'])==nw);assert f.stat().st_size==int(source['byte_count'])
   text=next(f.parent.glob(f'toltec{nw}_{tune[0]}_{tune[1]:03}_{tune[2]:04}_*_tune.txt'));kmp=next(x for x in sources if x['role']=='kmp' and int(x['nw'])==nw);assert 'sha256:'+bound(text)['sha256']==kmp['content_sha256']
   raw.append(dict(path=str(f),bytes=f.stat().st_size,network=nw,identity=identity,shape=list(shape),tune=tune,kmp=bound(text),declared_raw_sha256=source['content_sha256'],raw_hash_status='not reread in metadata-only OOF inventory; header and byte count checked'))
  assert sorted(x['network'] for x in raw)==sorted(int(x['nw']) for x in sources if x['role']=='raw')
  selected.append(dict(observation=obs,source=c['source'],mode=c['mode'],goal=c['goal'],date_UT=c['date_UT_header'],telescope=tel,raw=raw,networks=c['networks'],raw_bytes=sum(x['bytes'] for x in raw),canonical_manifest=bound(bundle/'manifest.ecsv'),matched_report=bound(V/f'oof/apts/v2/apt_{obs}_matched_match_report.json'),dependency_components=[bound(p) for p in components.values()],missing_paths=[],historical_root=str(V/f'oof/refactor/1146+399/reduced/redu00/{obs}')))
 ref=selected[1]['telescope']['headers']['Header.M2.ZReq']['value']
 for s in selected:s['requested_Z_offset_from_middle_mm']=s['telescope']['headers']['Header.M2.ZReq']['value']-ref
 root=V/'oof/refactor/1146+399/reduced/redu00';cfg=yaml.safe_load((root/'citlali_o152385_0_1_c3.yaml').read_text());artifact=dict(schema='point-to-oof-local-sequence-v1',inventory_mode='read-only local headers, byte counts, canonical manifests and KMP hashes; no reduction/download',census=bound(census_path),sequence=selected,source_project=bound(V/'oof/project.yaml'),historical_config=bound(root/'citlali_o152385_0_1_c3.yaml'),historical_settings={k:cfg.get(k) for k in ['runtime','mapmaking','pointing','source']},historical_timestream={k:cfg['timestream'].get(k) for k in ['fruit_loops','chunking','raw_time_chunk','processed_time_chunk']},total_raw_bytes=sum(x['raw_bytes'] for x in selected),missing_raw_bytes=0,new_oof_starts=0,limits='No best-focus/optical sign inference from actuator values; offsets are requested physical Z relative to the middle acquisition. Focus states stay distinct. No successor OOF admission or morphology policy implied.')
 (D/'results/oof-inventory.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
 print(json.dumps([dict(observation=x['observation'],networks=x['networks'],bytes=x['raw_bytes'],focus=x['requested_Z_offset_from_middle_mm'],tune=sorted(set(tuple(r['tune']) for r in x['raw']))) for x in selected],indent=2))
if __name__=='__main__':main()
