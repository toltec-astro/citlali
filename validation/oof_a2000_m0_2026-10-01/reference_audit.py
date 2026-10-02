"""Authenticate saved R products and their executed configuration; no reduction."""
import sys,json,subprocess
from pathlib import Path
import yaml,numpy as np,netCDF4
from astropy.io import fits
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W/'tools/timestream_successor'))
from point_fruit_trajectory import bound
from point_additive_diagnostics import save_json
from inspect_forward_map import require
L=Path.home()/'work_toltec/local_data';R=L/'citlali-validation/v2/oof/refactor/1146+399/reduced/redu00'
def main(out):
 config=R/'citlali_o152385_0_1_c3.yaml';c=yaml.safe_load(config.read_text());rows=[]
 rtc=c['timestream']['raw_time_chunk'];ptc=c['timestream']['processed_time_chunk'];m=c['mapmaking']
 require(ptc['clean']['grouping']==['nw'] and ptc['clean']['standard_pca']['n_eig_to_cut']['a2000']==[3],'reference cleaning changed')
 require(ptc['weighting']['validation']['atmospheric_grouping']=='array' and ptc['weighting']['validation']['high_weight_grouping']=='array','cross-array weighting differs')
 require(not c['timestream']['fruit_loops']['enabled'] and not c['coadd']['enabled'],'reference feedback/coadd')
 for obs in (152385,152386,152387):
  acq=next(x for x in c['inputs'] if x['meta']['name']==f'{obs}_0_1');bindings=[];networks=[]
  for item in acq['data_items']+acq['cal_items']:
   if 'filepath' not in item:continue
   p=L/item['filepath'].removeprefix('/work/toltec/').replace('citlali-validation/v2/oof/data/','citlali-validation/v1/oof/data/');require(p.is_file(),'historical operand absent '+str(p));bindings.append(bound(p))
   interface=item.get('meta',{}).get('interface','')
   if interface.startswith('toltec'):
    nw=int(interface[6:]);networks.append(nw)
    if nw in (11,12):
     with netCDF4.Dataset(p) as nc:require(int(nc['Header.Toltec.RoachIndex'][...])==nw and int(nc['Header.Toltec.ObsNum'][...])==obs,'foreign reference raw')
  require([n for n in networks if n>=11]==[11,12],'historical complete a2000 population')
  manifest=next(b for b in bindings if b['path'].endswith('/manifest.ecsv'))
  require(f'apt_{obs}_matched.apt-v2' in manifest['path'],'APT observation binding')
  mapfile=R/str(obs)/'raw'/f'toltec_commissioning_a2000_oof_{obs}_citlali.fits'
  provenance=R/str(obs)/'raw_timestream_provenance.yaml';p=yaml.safe_load(provenance.read_text())
  with fits.open(mapfile) as h:
   require(h[0].header['OOF_ID']==2000,'reference array');header=dict(h['signal_I'].header)
  rows.append(dict(observation=obs,map=bound(mapfile),raw_and_calibration_bindings=bindings,all_recorded_networks=networks,a2000_networks=[11,12],provenance=bound(provenance),software=json.loads(json.dumps(p['software_identity'],default=str)),source_protection_resolution=p['effective']['resolutions']['source_protection'] if 'effective' in p else None,signal_header=header))
 sources={}
 for path in ('include/citlali/core/timestream/ptc/ptcproc.h','include/citlali/core/pipeline/map_group_indexing.h'):
  sources[path]=subprocess.check_output(['git','rev-parse','c31a60a0:'+path],cwd=W,text=True).strip()
 save_json(out,dict(status='reuse operational historical R for all three; zero R starts',config=bound(config),maps=rows,recipe=dict(RTC=rtc,PTC=ptc,MAP=m,feedback=False),cross_band_coupling=dict(cleaning='network blocks; historical ptcproc lines1035–1085',weight_validation='array grouping; historical ptcproc group_for lines2110+, high-weight same grouping',map='separate array index; map_group_indexing',timing='full recorded acquisition timestamp support retained in S too',conclusion='no selected signal cleaning/normalization across arrays; common timing/TEL retained',historical_source_blobs=sources),limits=['effective admitted sample calibration/exposure may differ and is an outcome','historical R is operational reference, not ground truth; source/config/map differences retained']))
if __name__=='__main__':main(Path(sys.argv[1]))
