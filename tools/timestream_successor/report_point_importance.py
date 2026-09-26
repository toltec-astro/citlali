"""Cached-only POINT measurement-importance report; missing caches fail closed."""
import json, hashlib, math
from pathlib import Path
root=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs')
g=root/'successor-point-152389-guarded-feedback-comparison-20260925'
a=root/'successor-point-152389-post-fruit-ab-20260923'
d=root/'successor-point-152389-a2000-saved-product-diagnosis-20260925'
repo=Path(__file__).resolve().parents[2]
sources={}
def source(path):
 p=Path(path);key=str(p);sources[key]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size};return key
def read(path):source(path);return json.loads(Path(path).read_text())
s=read(g/'report/SUMMARY.json');summary_key=str(g/'report/SUMMARY.json')
def ref(key):return {'file':summary_key,'record_key':key}
for p in ['point_expected.py','point_post_fruit.py','point_guarded_feedback.py','point_admission.py','inspect_point_source.py']:source(repo/'tools/timestream_successor'/p)
source(g/'ASSESSMENT.md')
ab={ar:read(a/'companion'/ar/'source-uncertainty.json') for ar in s['arrays']}
for ar in s['arrays']:
 source(a/'companion'/ar/'background-coupling.json');source(a/'companion'/ar/'propagation-checks.json');source(a/'companion'/ar/'T10/model.json');source(a/'companion'/ar/'frozen-domains.json')
diagnosis=read(d/'RESULTS.json');source(d/'MEASUREMENT_TABLES.md')
measures=['peak','fixed_template','aperture_1','aperture_1p5','aperture_2','east','north','FWHM_major','FWHM_minor']
mapname={'peak':'peak','fixed_template':'template','aperture_1':'aperture1','aperture_1p5':'aperture1.5','aperture_2':'aperture2'}
def val(m,n):
 p=m['fit']['parameters']
 if n in ['peak','east','north']:return p[['peak','east','north'].index(n)]
 if n=='fixed_template':return m['frozen_template_projection']
 if n.startswith('aperture_'):return m['apertures'][['aperture_1','aperture_1p5','aperture_2'].index(n)]['value_mJy_per_beam_arcsec2']
 if n=='FWHM_major':return max(p[3:5])*math.sqrt(8*math.log(2))
 if n=='FWHM_minor':return min(p[3:5])*math.sqrt(8*math.log(2))
def scale(ar,pol,br,n,measurement,stage):
 k=f'arrays.{ar}.policies.{pol}.history.{br}[{stage}].measurement.guard.sigma_A'
 if n=='peak':
  return {'value':measurement['guard']['sigma_A'],'kind':'conditional_model_based','provenance':ref(k),'interpretation':'Corresponding current POINT-EXPECTED-v1 fitted peak, all seven free nuisance directions, W=I, original 40 arcsec fit disk. Frozen full original-M0 C0 transferred from background to source; not endpoint empirical noise, not uncertainty of paired difference.'}
 return {'value':None,'kind':'unavailable_matched_endpoint','interpretation':'No saved current POINT-EXPECTED-v1 refitted-constant uncertainty for this measurement. Historical A+B conditional ensemble is supplied separately.' if n in mapname else 'No saved matching parameter uncertainty. Absolute physical change retained; width fraction uses source width; centroid fraction of offset not scientifically interpreted.'}
rows=[];responses=[];seed_maps=[];refs=[]
for ar,ax in s['arrays'].items():
 for pol,x in ax['policies'].items():
  prefix=f'arrays.{ar}.policies.{pol}'
  mref=x['history']['REF'][-1]['measurement']
  refs.append({'array':ar,'policy':pol,'stage':'T10','values':{n:val(mref,n) for n in measures},'sigma_A':mref['guard']['sigma_A'],'provenance':ref(prefix+'.history.REF[10].measurement')})
  for br in x['history']:
   mh=x['history'][br][-1]['measurement'];prev=x['history'][br][-2]['measurement']
   cases=[]
   if br=='D-seed-shoulder':
    q=x['branches'][br][-1];cases.append(('seed_minus_own_REF',q['measurement_increment'],mref,ref(prefix+f'.branches.{br}[10].measurement_increment'),None))
    seed_maps.append({'array':ar,'policy':pol,'stage':'T10','rms':q['difference_rms'],'maximum_absolute':q['difference_max_abs'],'units':'mJy/nominal beam','shoulder_projection':q['shoulder_projection'],'candidate_shoulder_projection':q['candidate_shoulder_projection'],'provenance':ref(prefix+f'.branches.{br}[10]'),'scale_status':'local map noise at extreme not authenticated; no ratio to fitted peak scatter; maximum is postselected descriptive extreme'})
   elif br!='REF':
    q=x['branches'][br][-1];cs=q['ideal_measurement']['comparisons'];cases.append(('actual_minus_ideal', {v['name']:v['absolute_discrepancy'] for v in cs},mref,ref(prefix+f'.branches.{br}[10].ideal_measurement.comparisons'),{v['name']:v for v in cs}))
    responses.append({'array':ar,'policy':pol,'branch':br,'stage':'T10',**q['benchmark_response'],'provenance':ref(prefix+f'.branches.{br}[10].benchmark_response')})
   cases.append(('own_T10_minus_T9',x['history'][br][-1]['previous_cycle_drift'],prev,ref(prefix+f'.history.{br}[10].previous_cycle_drift'),None))
   for effect_type,delta,denom,prov,ideal in cases:
    for n in measures:
     e=delta[n];den=val(denom,n);sc=scale(ar,pol,br,n,mh,10)
     row={'measurement':n,'array':ar,'policy':pol,'branch':br,'stage':'T10','effect_type':effect_type,'effect':e,'units':s['units'][n],'source_measurement':den,'source_denominator':'own previous T9' if effect_type=='own_T10_minus_T9' else 'own policy REF T10','fraction_of_source':e/den if den and n not in ('east','north') else None,'scale':sc,'noise_scale_ratio':abs(e)/sc['value'] if sc['value'] else None,'provenance':prov}
     if n in mapname:
      i=ab[ar]['order'].index(mapname[n]);h=ab[ar]['measurements'][i];row['historical_context']={'sigma':h['scatter'],'ratio':abs(e)/h['scatter'],'kind':'historical_conditional_additive_scatter','units':h['units'],'record_key':f'measurements[{i}].scatter','file':str(a/'companion'/ar/'source-uncertainty.json'),'trials':ab[ar]['trials'],'failures':ab[ar]['failures'],'bound_limited':ab[ar]['bound_limited'],'not_matching_endpoint':True,'qualification':'Historical BASE T10 covariance transferred from masked background to source; same original frozen aperture/template geometry and trial-refitted seven-parameter constant, but historical fit sigma floor 0.5 arcsec versus POINT-EXPECTED 1.5 arcsec and historical source/endpoint. Includes covariance with refitted background. Not independent pixel RMS.'}
     if ideal:
      row['actual_increment']=ideal[n]['actual_increment'];row['ideal_increment']=ideal[n]['ideal_increment'];row['fraction_of_signed_ideal_increment']=e/ideal[n]['ideal_increment'] if n in mapname and abs(ideal[n]['ideal_increment'])>1e-8 else None
     rows.append(row)
# Authenticate duplicate diagnosis numbers exactly.
checks=[]
for rec in diagnosis['seed']:
 if rec['map_stage']!=10:continue
 pol=rec['policy'];q=s['arrays']['a2000']['policies'][pol]['branches']['D-seed-shoulder'][-1]
 for measure,actual,expected in [('peak',rec['measurement_difference']['peak'],q['measurement_increment']['peak']),('full_map_max',rec['full_map']['maximum_absolute'],q['difference_max_abs']),('aperture_2',rec['measurement_difference']['apertures'][2]['source_relative'],q['measurement_increment']['aperture_2'])]:
  assert actual==expected
  checks.append({'policy':pol,'measurement':measure,'value':actual,'exact_agreement':True})
out={'schema':'point-importance-extraction/v1','date':'2026-09-26','method':'JSON extraction and scalar arithmetic only; no scientific package imported, fits, ensembles, candidate evaluations, covariance construction, or field generation','sources':sources,'reference_measurements':refs,'measurement_effects':rows,'response_metrics':responses,'seed_map_context':seed_maps,'diagnosis_cross_checks':checks,'rules':['All effects are signed; absolute value used only in noise-scale numerator.','Ratios are practical size, not p-values or calibrated significance of paired noise-sharing differences.','Peak conditional scale is current branch guard sigma_A; historical A+B scatter is separately contextual.','Aperture brightness-area units must never be divided by a brightness-only sigma.','Fixed-background analytic operator variances exclude trial-refitted constant and are not substitute uncertainties.','No matching endpoint aperture/template/centroid/width uncertainty is claimed.']}

# Unit normalization only for equivalent display spellings; never change a quantity.
from point_archive_compaction import importance_ratio, fraction
unit_alias={'mJy/nominal beam × arcsec²':'mJy/nominal beam arcsec^2'}
for r in rows:
 u=unit_alias.get(r['units'],r['units'])
 if r['noise_scale_ratio'] is not None:
  assert r['noise_scale_ratio']==importance_ratio(r['effect'],r['scale']['value'],u,u,'conditional_reference')
 if 'historical_context' in r:
  h=r['historical_context'];hu=unit_alias.get(h['units'],h['units'])
  assert h['ratio']==importance_ratio(r['effect'],h['sigma'],u,hu,'historical_context')
 if r['fraction_of_source'] is not None:
  assert abs(100*r['fraction_of_source']-fraction(r['effect'],r['source_measurement'],'own_reference_source'))<1e-12
# Authenticate numerical caches against their original completed root ledgers.
expected={g:'5f6bf979f4e4ee2f4e2d249512c9107e73ed7f8fab8c58ccc257db27c65265a9',a:'8f8d21469ba82e8a5357bcd13eaaff07a51f41d0c257eac782776356615ebebc',d:'0c2f915a5ed6d93b52c6a4ed0c8324b12fb70d67474e4e49787ae45308146462'}
for bundle,wanted in expected.items():
 ledger=bundle/'FILES_SHA256SUMS';assert hashlib.sha256(ledger.read_bytes()).hexdigest()==wanted
 hashes={n:h for h,n in (line.split('  ',1) for line in ledger.read_text().splitlines())}
 for name,info in sources.items():
  path=Path(name)
  if path.is_relative_to(bundle):assert hashes[str(path.relative_to(bundle))]==info['sha256']
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=root/'successor-point-152389-importance-and-archive-compaction-20260926/IMPORTANCE.json');args=p.parse_args()
args.output.write_text(json.dumps(out,indent=2)+'\n')
print('Authenticated cached-only rows',len(rows),'sources',len(sources))
