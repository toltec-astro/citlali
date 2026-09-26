from pathlib import Path
import gzip,json,re,hashlib,collections
R=Path('/Users/gwilson/work_toltec/local_data');V=Path('/private/tmp/citlali-point-additive-20260923/validation/local_data_rtc_ptc_purge_2026-09-26')
def read(n):return json.load(gzip.open(V/n,'rt'))
rows=read('INVENTORY_BEFORE.json.gz');headers={h['path']:h for h in read('NETCDF_HEADERS.json.gz')};targets=[];unknown=[];classes=collections.Counter();count=collections.Counter();proofcache={}
def proof(p,words):
 key=(str(p),tuple(words))
 if key not in proofcache:
  try:
   with p.open('rb') as f:b=f.read(8192)
   ok=any(w.encode()in b for w in words)
   proofcache[key]=(str(p.relative_to(R)),hashlib.sha256(b).hexdigest()) if ok else None
  except OSError:proofcache[key]=None
 return proofcache[key]
def nearest(p,names,words,maxlevels=5):
 for q in [p.parent,*list(p.parents)[1:maxlevels]]:
  for name in names:
   a=proof(q/name,words)
   if a:return a
 return None
for r in rows:
 rel=r['path'];p=R/rel;n=p.name;reason=None;ev=None;extra={};h=headers.get(rel)
 if h:
  typ=h.get('metadata',{}).get('tod_output_type');typ=typ[0] if isinstance(typ,list) and typ else typ
  sig=h.get('variables',{}).get('signal')
  if sig and typ in ['rtc','ptc','ptc_detector_tod'] and sig['dims'] in [['n_pts','n_dets'],['n_dets','n_detector_tod_slots','n_samples_max']]:
   reason='citlali_'+typ+'_mixed_timestream';ev='NetCDF tod_output_type='+typ+';signal dimensions='+str(sig['dims'])
   dim='n_pts' if 'n_pts'in sig['dims'] else 'n_samples_max'
   extra=dict(sample_dimension=dim,removed_variables=[k for k,v in h['variables'].items() if dim in v['dims']],retained_variables=[k for k,v in h['variables'].items() if dim not in v['dims']],retained_companion=rel[:-3]+'_purge_retained.nc')
 elif rel.startswith(('citlali-validation/development-runs/','2026-refactor/citlali-successor-')):
  if re.fullmatch(r'\d+-(conditioned\.f64|filtered\.f64|rows\.i64|(?:input-causes|causes|map-center|support-causes|speed-evidence|state)\.u8)',n) and p.parent.name=='donor-continuity':
   ev=nearest(p,['receipt.yaml','post-lowpass-outcome.yaml','post-notch-outcome.yaml'],['citlali','RTC','rtc']);reason='native_RTC_applied_samples_and_axes' if ev else None
  elif re.fullmatch(r'channel-\d+-(value\.f64|slot\.i64|causes\.u16)',n) and p.parent.name=='cal':
   ev=proof(p.parent/'receipt.yaml',['citlali-cal-output-v1']);reason='calibrated_RTC_cache_samples_and_axes' if ev else None
  elif re.fullmatch(r'segment-\d+-(input\.f64|centered\.f64|cleaned\.f64|eligible\.u8|causes\.u8|slots\.i64)',n):
   ev=proof(p.parent/'receipt.yaml',['citlali-ptc-output']);reason='native_PTC_samples_and_axes' if ev else None
  elif re.fullmatch(r'network-\d+-segment-\d+',p.parent.name) and n in {'cleaned.f64','operand.f64','injection.f64','model-support.u8','causes.u8'}:
   ev=nearest(p,['receipt.yaml','receipt.json'],['point','fruit','citlali','schema']);reason='FRUIT_native_PTC_samples_and_masks' if ev else None
  elif n=='cleaned.f64':
   ev=proof(p.parent/'result.yaml',['iterations','rank','method','shape','fit']);reason='PTC_solver_applied_samples' if ev else None
  elif n in {'operator-pixel.i32','output-rows.u64'} and 'products'in p.parts:
   ev=nearest(p,['receipt.yaml','bundle.yaml'],['point','fruit','citlali','schema']);reason='PTC_to_MAP_occurrence_axes' if ev else None
  elif re.fullmatch(r'network-\d+-segment-\d+-ast\.f64',n) and p.parent.name=='handoff':
   ev=nearest(p,['receipt.yaml','bundle.yaml'],['point','citlali','schema']);reason='PTC_MAP_handoff_sample_axes' if ev else None
  elif re.fullmatch(r'network-\d+-segment-\d+(\.f64|-cal\.u16)',n) and '/ast/'in rel and 'point-152389-map-only-jinc' in rel:
   ev=nearest(p,['receipt.yaml'],['point-152389-native-ast-v1']);reason='JINC_PTC_occurrence_axes' if ev else None
  elif '/prepared/'in rel and 'point-152389-map-only-jinc' in rel and (n in {'clean.f64','operand.f64','total.f64','geometry.f64','network.i16','tangent-arcsec.f64'} or re.fullmatch(r'(phase-\d+|C-(core-plus|core-minus|shoulder)-injection)\.f64',n)):
   ev=nearest(p,['STATE.json'],['array','sha256','schema']);reason='JINC_prepared_PTC_samples_and_axes' if ev else None
  elif '/freeze/'in rel and 'successor-point-152389-'in rel and re.fullmatch(r'network-\d+-segment-\d+-(parent\.f64|injected-parent\.f64|pixel\.i32)',n):
   ev='producer: run_point_guarded_feedback.py/run_point_refseq.py/run_point_frozen_shoulder.py; frozen per-PTC-slot parent/axis';reason='frozen_PTC_parent_samples_and_axes'
 if rel.startswith('fruit-development/fruit-el-f12-response-aware-intervention-r0.1/') and re.fullmatch(r'fruit_response_123424_iter\d+\.bin\.gz',n):
  reason='compressed_final_PTC_occurrence_spool';ev='source10478fa0e include/citlali/core/fruit/response_ledger.h and setup/REGISTRATION_R0.2.json; 8 int64 identities+5 doubles including PTC sample'
 if reason:
  assert r['nlink']==1
  targets.append(dict(**r,classification=reason,producer_evidence=ev,disposition='remove under September26 owner RTC/PTC purge; no replacement or regeneration proof required',**extra))
 elif p.suffix in {'.f64','.u8','.u16','.i64','.i32','.i16','.u64'}:unknown.append(dict(path=rel,bytes=r['bytes']))
 cl=reason or ('protected_raw_acquisition' if h and 'Data.Toltec.Is'in h['variables'] else 'protected_tune_or_telescope_or_other_NetCDF' if h else 'maps' if p.suffix=='.fits' else 'figure' if p.suffix in {'.png','.pdf','.svg'} else 'compact_records_and_config' if p.suffix in {'.md','.yaml','.json','.yml','.toml','.sh'} else 'APT_matching_calibration_tables' if p.suffix=='.ecsv' else 'other_or_unclassified')
 classes[cl]+=r['bytes'];count[cl]+=1
with gzip.open(V/'TARGETS_DRAFT.json.gz','wt') as f:json.dump(targets,f,separators=(',',':'))
with gzip.open(V/'NATIVE_RETAINED_CENSUS.json.gz','wt') as f:json.dump(unknown,f,separators=(',',':'))
(V/'PRODUCT_CLASSES.json').write_text(json.dumps([dict(product_class=k,bytes=v,files=count[k])for k,v in classes.most_common()],indent=2)+'\n')
print('TARGETS',len(targets),sum(x['bytes']for x in targets));print(collections.Counter(x['classification']for x in targets));print('RETAINED NATIVE',len(unknown),sum(x['bytes']for x in unknown));print(collections.Counter(re.sub(r'\d+','N',Path(x['path']).name)for x in unknown).most_common(25))
