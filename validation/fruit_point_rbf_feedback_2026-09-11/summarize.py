#!/usr/bin/env python3
"""Compact decision evidence from retained metrics; no parameter choices."""
from pathlib import Path
import json
import numpy as np
from astropy.table import Table
import rbf
b=rbf.base;H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def main():
 results={}
 for version in ['R0.1','R0.2']:
  d=read(H/f'NUMERICAL_EVIDENCE_{version}.json');final=d['final'];root=Path(d['root']);geo=np.load(root/'geometry.npz');admissions={};det=[]
  for case in ['real123424','null20260911','null20260912','background20260911','gauss20260911','gauss20260912','mismatch20260911','coma_bright','coma_half']:
   admissions[case]={array:sum(r['decision'].get('admitted',False) for r in d['raw'] if r['arm']=='R' and r['case']==case and r['array']==array) for array in b.ARRAYS}
  for case in ['gauss20260911','coma_bright','coma_half']:
   t=np.load(root/(case+'_truth.npz'))['truth']
   for a,array in enumerate(b.ARRAYS):
    r=next(r for r in d['raw'] if r['arm']=='R' and r['case']==case and r['array']==array and r['pass_index']==0)
    det.append(dict(case=case,array=array,truth_peak=float(t[a].max()),truth_norm_over_map_MAD=float(np.linalg.norm(t[a,geo['D'][a]])/r['decision']['scale']),qualification='oracle shape-norm/scatter scale only; ignores map covariance and cleaning attenuation; not detection significance'))
  compact=[]
  for r in final:
   if r['case'].startswith('background'):continue
   item={k:r[k] for k in ['case','array','arm','passes','joint_target','relative_L2','raw_relative_L2','integrated_brightness_error','raw_integrated_brightness_error','centroid_error_arcsec','direct_D_error_rms','exterior_error_rms','cold_map_seconds','method_map_seconds','next_model_relative_L2','applied_model_relative_L2']}
   item['wing_brightness_error']=r['regions']['wings']['integrated_brightness_error']
   for k in ['amplitude_bias','width_bias','gaussian_centroid_error_arcsec','scientific_Aeff_bias','raw_fitted_Aeff_bias']:item[k]=r.get(k)
   compact.append(item)
  real=[]
  for r in d['raw']:
   if r['case']=='real123424' and r['pass_index']==6:
    f=r['fit'];real.append(dict(array=r['array'],arm=r['arm'],peak=f.get('peak'),centroid=f.get('centroid'),widths=f.get('widths'),fit_boundary=f.get('boundary_rejection'),admission=r['decision']['reason'],cold_map_seconds=r['cold_map_seconds'],warm_map_seconds=r['method_map_seconds'],next_integrated_brightness=r['next_integrated_brightness'],next_rms=r['next_rms'],outside_rms=r['total_O_rms']))
  first=[dict(case=r['case'],array=r['array'],arm=r['arm'],passes=r['first']['passes'] if r['first'] else None,cold_map_seconds=r['first']['cold_map_seconds'] if r['first'] else None) for r in d['first_joint_targets']]
  times={arm:{key:float(np.median([r[key] for r in d['timing'] if r['arm']==arm])) for key in ['clean_seconds','map_seconds','policy_seconds','diagnostic_seconds','evaluation_seconds','output_seconds']} for arm in ['P','G','R']}
  mismatches=[]
  for array in b.ARRAYS:
   R=next(r for r in final if r['case'].startswith('mismatch') and r['array']==array and r['arm']=='R');G=next(r for r in final if r['case'].startswith('mismatch') and r['array']==array and r['arm']=='G')
   mismatches.append(dict(array=array,direct_error_ratio=R['direct_D_error_rms']/G['direct_D_error_rms'],exterior_error_ratio=R['exterior_error_rms']/G['exterior_error_rms']))
  solvers=[fit for row in d['raw'] if row['arm']=='R' for fit in [row['decision']['full_solver']]+[v['solver'] for v in row['decision']['fold_fits']]]
  solver_summary=dict(fits=len(solvers),maximum_iterations=max(s['iterations'] for s in solvers),maximum_relative_KKT=max(s['relative_kkt'] for s in solvers),maximum_condition_bound=max(s['condition_upper_bound'] for s in solvers))
  convergence=[]
  for a,array in enumerate(b.ARRAYS):
   for arm in ['P','G','R']:
    m=np.load(root/'real123424'/arm/'pass06_maps.npz');prev=np.load(root/'real123424'/arm/'pass05_maps.npz');mask=geo['D'][a]
    an=np.linalg.norm(m['applied_model'][a,mask]);tn=np.linalg.norm(prev['total'][a,mask])
    convergence.append(dict(array=array,arm=arm,next_vs_applied_relative_L2=float(np.linalg.norm((m['next_model']-m['applied_model'])[a,mask])/an) if an else None,last_output_step_relative_L2=float(np.linalg.norm((m['total']-prev['total'])[a,mask])/tn) if tn else None))
  results[version]=dict(solver_summary=solver_summary,real_final_step=convergence,campaign=d['campaign'],setup=d['setup'],final=compact,admissions=admissions,real=real,first_joint_targets=first,median_pass_cost=times,mismatch_nondegradation=mismatches,detectability_audit=det,failures=d['failures'])
 og_binding=read(rbf.PREV/'OG_BENCHMARK.json');native_path=Path(og_binding['root'])/'redu06/123424/raw/ppt_commissioning_pointing_123424_citlali.ecsv';table=Table.read(native_path)
 native=dict(path=str(native_path),sha256=b.digest(native_path),observable='native operational POINT fit; distinct from the common free Gaussian evaluation',rows=[{key:float(row[key]) for key in table.colnames} for row in table])
 b.write(H/'DECISION_EVIDENCE.json',dict(OG_native_POINT=native,feedback_decision='REJECT both frozen candidates for POINT adoption; broader RBF hypothesis unresolved',concentration_decision='REVISE empirical diagnostic: algebra verified but known-source measurements unavailable after admission; no operational focus qualification',revision_allowance='exhausted; no further sweep',results=results,OG=read(rbf.PREV/'NUMERICAL_EVIDENCE_R0.2.json')['OG']))
 for key,r in results.items():
  print(key,'campaign',r['campaign'],'setup',r['setup']['basis_seconds']);print('admissions',r['admissions']);print('median cost',r['median_pass_cost']);print('real',r['real']);print('mismatch',r['mismatch_nondegradation'])
  print('compact R amplitudes',[round(100*x['amplitude_bias'],3) for x in r['final'] if x['arm']=='R' and x['case'].startswith('gauss')]);print('raw Aeff coma',[x['raw_fitted_Aeff_bias'] for x in r['final'] if x['arm']=='R' and x['case'].startswith('coma')])
if __name__=='__main__':main()
