from pathlib import Path
import json,gzip,time,re,collections
import netCDF4,numpy as np
R=Path('/Users/gwilson/work_toltec/local_data');V=Path('/private/tmp/citlali-point-additive-20260923/validation/local_data_rtc_ptc_purge_2026-09-26')
def value(v):
 a=v[:]
 if getattr(a,'dtype',None) is not None and a.dtype.kind in 'SU':return b''.join(np.ma.filled(a,b'').flatten().tolist()).decode(errors='replace').strip('\0 ')
 if np.ma.is_masked(a):return None
 return np.asarray(a).tolist()
def inspect(row):
 path=row['path'];o=dict(path=path,bytes=row['bytes']);p=R/path
 try:
  with netCDF4.Dataset(p) as d:
   o['format']=d.data_model;o['dimensions']={k:len(v) for k,v in d.dimensions.items()};o['variables']={k:dict(shape=list(v.shape),dims=list(v.dimensions),dtype=str(v.dtype)) for k,v in d.variables.items()};o['attributes']={k:str(d.getncattr(k))[:2000] for k in d.ncattrs()};meta={}
   for k,v in d.variables.items():
    if v.size<=512 and (re.search(r'ObsNum|SubObsNum|ScanNum|RoachIndex|TargSweep|ObsStartTime|ObsEndTime|ObsType|SelectedMask|Master|SampleFreq|ObsPgm|ObsGoal|SourceName|ProjectId|Receiver|UTDate|File.Name|Kids.kind|Kids.source',k,re.I) or k in ['tod_output_type','obsnum','git_version','version']):
     try:meta[k]=value(v)
     except Exception as e:meta[k]={'error':str(e)}
   o['metadata']=meta
   for k in ['Data.Toltec.RecvTime','Data.TelescopeBackend.TelTime','Data.TelescopeBackend.TelUTC','TelTime']:
    if k in d.variables and d.variables[k].ndim==1 and d.variables[k].size:
     v=d.variables[k]
     try:o.setdefault('time_endpoints',{})[k]=[float(v[0]),float(v[-1])]
     except Exception as e:o.setdefault('time_errors',{})[k]=str(e)
   o['groups']=list(d.groups)
 except Exception as e:o['error']=str(e)
 return o
rows=json.load(gzip.open(V/'INVENTORY_BEFORE.json.gz','rt'));selected=[r for r in rows if r['path'].lower().endswith('.nc')];out=[];start=time.monotonic()
for i,row in enumerate(selected):
 out.append(inspect(row))
 if (i+1)%1000==0:print('Headers',i+1,'/',len(selected),round(time.monotonic()-start,1),flush=True)
with gzip.open(V/'NETCDF_HEADERS.json.gz','wt') as f:json.dump(out,f,separators=(',',':'),allow_nan=False)
print('COMPLETE',len(out),'errors',sum('error'in x for x in out),'seconds',time.monotonic()-start,flush=True)
