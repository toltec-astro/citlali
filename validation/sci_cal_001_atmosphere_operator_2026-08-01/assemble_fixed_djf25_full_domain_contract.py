#!/usr/bin/env python3
"""Assemble the owner-directed fixed-DJF25, no-selector contract evidence."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import evaluate_tau025_engineering_evidence as high
import run_am12_successor_adoption_study as low

P=Path(__file__).resolve().parent
ROOT=high.ROOT
LOW=P/'am12_successor_operator_nodes.csv'
OUT=P/'sci_cal_001_fixed_djf25_full_domain_operator_contract.json'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rows=[]
    with LOW.open() as f:
        for r in csv.DictReader(f):
            if r['lane']=='fixed_djf25_v1' and r['passband_id'].startswith('tolteca_v1_'):
                rows.append({k:r[k] for k in ('source_profile','tau225','elevation_deg','passband_id','array','alpha','line_of_sight_optical_depth','extinction_correction')} | {'anchor_id':r['target'],'provenance':'low_fixed_djf25_nodes_v2'})
    assert len(rows)==3*31*3*4
    for side in (ROOT/'sidecars').glob('*.json'):
        s=json.loads(side.read_text()); q=s.get('request',{})
        if not (s.get('run_id','').startswith('tau025e001/construction/LMT_DJF_25/') and q.get('target') in ('tau015','tau020','tau025')): continue
        raw=ROOT/'raw_outputs'/side.with_suffix('.txt').name; parsed=low.parse_am_output(raw.read_bytes(),s['run_id'])
        e=90-int(q['zenith_angle_deg'])
        for b in high.bands():
            tx=__import__('numpy').interp(b.frequency_ghz,parsed.frequency_ghz,parsed.transmission)
            for a in high.ALPHAS:
                lam=-math.log(float(__import__('numpy').dot(b.weights(a),tx)))
                rows.append({'anchor_id':q['target'],'source_profile':'LMT_DJF_25','tau225':str(high.TAU[q['target']]),'elevation_deg':str(e),'passband_id':b.identity,'array':b.array,'alpha':str(a),'line_of_sight_optical_depth':format(lam,'.17e'),'extinction_correction':format(math.exp(lam),'.17e'),'provenance':'tau025_extension_002'})
    assert len(rows)==1116+252
    fields=['anchor_id','source_profile','tau225','elevation_deg','passband_id','array','alpha','line_of_sight_optical_depth','extinction_correction','provenance']
    with (P/'sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    tau_by_anchor={'am_q25':.0504874104674104401,'am_q50':.0883393725904400573,'tau015':.15,'am_q75':.158313198574890929,'tau020':.2,'tau025':.25}
    grouped={}
    for row in rows:
        if float(row['elevation_deg']) < 25: continue
        grouped.setdefault((row['anchor_id'],row['elevation_deg'],row['passband_id']),{})[int(row['alpha'])]=float(row['extinction_correction'])
    sensitivity={}
    for label,predicate in (('science_qualification_regime',lambda t:t<=.15),('engineering_availability_regime',lambda t:t>.15)):
        probes=[]
        for (anchor,elevation,passband),values in grouped.items():
            if predicate(tau_by_anchor[anchor]):
                for alpha,correction in values.items():
                    if alpha: probes.append((abs(correction/values[0]-1),anchor,elevation,passband,alpha,tau_by_anchor[anchor]))
        value,anchor,elevation,passband,alpha,tau=max(probes)
        sensitivity[label]={'maximum_relative_to_alpha0':value,'location':{'anchor_id':anchor,'tau225':tau,'elevation_deg':int(elevation),'passband_id':passband,'alpha':alpha}}
    assert abs(sensitivity['science_qualification_regime']['maximum_relative_to_alpha0']-.037912745) < 1e-9
    assert abs(sensitivity['engineering_availability_regime']['maximum_relative_to_alpha0']-.060784703) < 1e-9
    c={'schema_version':'sci-cal-001-fixed-djf25-full-domain-operator-contract-v1','operator_id':'am12_fixed_djf25_piecewise_linear_los_tau_v1','reference_profile':{'filename':'LMT_DJF_25.amc','sha256':'aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866','meaning':'declared model approximation; not inferred atmospheric truth'},'domain':{'tau225':'0 <= tau225 <= 0.25','elevation_deg':'25 <= elevation_deg <= 80','outside':'fail_closed'},'quality_regimes':[{'tau225':'0 <= tau225 <= 0.15','label':'science_qualification_regime'},{'tau225':'0.15 < tau225 <= 0.25','label':'engineering_availability_regime'}],'convention':{'coordinate':'modified-secant zenith tau225','full_sample_airmass':True,'x_ref':0,'transmission':'finite and 0 < T <= 1','operator_switch_at_tau015':False,'derivative_continuity_gate':False},'anchors':[{'id':'tau0','tau225':'0','source':'analytic unity'},{'id':'am_q25','tau225':'0.0504874104674104401','source':'low fixed-DJF25'},{'id':'am_q50','tau225':'0.0883393725904400573','source':'low fixed-DJF25'},{'id':'tau015','tau225':'0.15','source':'TAU025 fixed-DJF25 target; achieved provenance 0.1499999859125433062628881602402745'},{'id':'am_q75','tau225':'0.158313198574890929','source':'low fixed-DJF25'},{'id':'tau020','tau225':'0.20','source':'TAU025 fixed-DJF25 target; achieved provenance 0.1999999783213567867059666712638576'},{'id':'tau025','tau225':'0.25','source':'TAU025 fixed-DJF25 target; achieved provenance 0.2499999377860148032413478624431719'}],'interpolation':{'opacity':'piecewise linear in LOS optical depth through ordered anchors','elevation':'shape-preserving PCHIP separately at every nonzero anchor; low anchors use even 20..80 evidence, TAU025 anchors use 25,35,..,80 evidence','zero':'analytic unity'},'provenance':{'low_nodes_csv_sha256':sha(LOW),'low_manifest_sha256':sha(P/'am12_successor_adoption_manifest.json'),'tau025_context_sha256':sha(ROOT/'execution_context.json'),'tau025_manifest_sha256':sha(ROOT/'manifests/execution_manifest.json'),'operator_nodes_csv_sha256':sha(P/'sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv')},'owner_accepted_tau015_comparison':{'case_count':84,'max_fractional_correction_difference':0.000338295,'p95':0.000216573,'rms':0.000099135,'array_maxima':{'a1100':0.000338295,'a1400':0.000085375,'a2000':0.000144061},'method':'coordinator fixed-DJF25 TolTECA-v1 band/alpha/elevation comparison; accepted as negligible'},'limitations':['representation contract only; no observational calibration validation','no runtime profile selector','value continuity at .15 is guaranteed by the shared operator anchor; derivative continuity is not a gate']}
    c['spectral_reference']={'configuration_field':'calibration.reference_spectral_index_alpha','semantics':'reference source spectrum S_nu proportional to nu^alpha for calibrated map/product meaning','default_alpha':0,'supported_values':[-1,0,2,4],'selection':'select one already precomputed alpha surface once per reduction; never integrate per sample','alpha_interpolation_or_extrapolation':'prohibited without a separately validated contract','unsupported_or_nonfinite':'fail_closed','omission':'use alpha=0 and record default_applied=true','quicklook':'pointing and OOF quicklook may omit the field and use the default','required_product_provenance':['effective_alpha','alpha_default_applied','tolteca_v1_passband_set_provenance','operator_id','reference_profile_id','calibration_quality_regime'],'map_meaning_limitation':'reference-spectrum convention only; it does not assert every source in a map has that spectrum'}
    c['reference_spectrum_sensitivity']=sensitivity
    c['representation_evidence_by_regime']={'science_qualification_regime':{'interpolation_fidelity':'The inherited low-opacity fixed-DJF25 study is numerical representation evidence only; its manifest records primary_holdout_fidelity_pass=false, so this contract does not claim a passing science numerical-fidelity gate.','reference_spectrum_sensitivity':sensitivity['science_qualification_regime'],'observational_accuracy':'not evaluated; no absolute-flux or repeatability claim.'},'engineering_availability_regime':{'interpolation_fidelity':'TAU025 profile-conditioned held-out study: fixed-DJF25 is included; aggregate all-profile piecewise-linear maximum is 0.532005%, not a science-quality or observational result.','reference_spectrum_sensitivity':sensitivity['engineering_availability_regime'],'observational_accuracy':'not evaluated; engineering availability is not calibrated-science validation.'}}
    OUT.write_text(json.dumps(c,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
