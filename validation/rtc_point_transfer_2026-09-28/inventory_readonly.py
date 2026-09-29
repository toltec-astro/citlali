from pathlib import Path
import json,hashlib,os,shutil,time
import netCDF4
W=Path('/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor');L=Path('/Users/gwilson/work_toltec/local_data');data=L/'2025-C1-COM-01/data';S=data.parent/'docs/pointing-iq-event-survey-20260729/pointing_raw_iq_event_survey.json';survey=json.loads(S.read_text());rows=[r for r in survey['rows'] if r['obsnum']==152420 and r['network'] in (1,2,3,4)]
headers=[]
for n in (1,2,3,4):
 paths=list(data.glob(f'toltec{n}_152420_000_0002_*.nc'));assert len(paths)==1
 with netCDF4.Dataset(paths[0]) as f:
  h={k:int(f['Header.Toltec.'+k][...]) for k in ('ObsNum','SubObsNum','ScanNum','RoachIndex','TargSweepObsNum','TargSweepSubObsNum','TargSweepScanNum')};h.update(path=str(paths[0]),shape=list(f['Data.Toltec.Is'].shape),timing_variables=[k for k in f.variables if 'Time' in k or 'Sample' in k],sample_freq=float(f['Header.Toltec.SampleFreq'][...]) if 'Header.Toltec.SampleFreq' in f.variables else None);headers.append(h)
roots=[L/'citlali-validation'/x for x in ('successor-rtc-retained-residuals-20260928','successor-rtc-two-stage-20260928')];sizes=[]
for root in roots:
 count=total=0
 for folder,dirs,files in os.walk(root):
  for name in files:
   p=Path(folder)/name;assert not p.is_symlink();total+=p.stat().st_size;count+=1
 sizes.append(dict(path=str(root),files=count,bytes=total))
cells=sum(h['shape'][0]*h['shape'][1] for h in headers);single=304*cells+2_000_000_000
out=dict(observation=152420,headers=headers,survey_path=str(S),survey_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),survey_rows=rows,retained_roots=sizes,retained_bytes=sum(q['bytes'] for q in sizes),free_bytes=shutil.disk_usage(L).free,cap_bytes=30_000_000_000,native_cells=cells,inherited_single_trajectory_reservation=single)
Path('/private/tmp/point-transfer-inventory.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='survey_rows'},indent=2));print('survey',len(rows));print(json.dumps(rows[:2],indent=2))
