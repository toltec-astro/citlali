#!/usr/bin/env python3
"""Compact assessment of the predeclared development gates; no new experiment."""
from pathlib import Path
import json,numpy as np

H=Path(__file__).resolve().parent
d=json.loads((H/'NUMERICAL_EVIDENCE.json').read_text());R=d['response'];P=d['receipts']
arrays=['a1100','a1400','a2000']
def final(case,arm,array):return next(r for r in R if r['case']==case and r['arm']==arm and r['array']==array and r['passes']==7)
recovery=[]
for case in ['gauss20260911','gauss20260912','near20260911']:
    for a in arrays:
        g,j=final(case,'G',a),final(case,'J',a)
        recovery.append(dict(case=case,array=a,joint_target=j['joint_target'],amplitude_degradation_pp=100*(abs(j['amplitude_bias'])-abs(g['amplitude_bias']))))
mismatch=[]
for a in arrays:
    g,j=final('mismatch20260911','G',a),final('mismatch20260911','J',a)
    mismatch.append(dict(array=a,direct_error_G=g['direct_D_error_rms'],direct_error_J=j['direct_D_error_rms'],direct_error_ratio=j['direct_D_error_rms']/g['direct_D_error_rms'],
        exterior_error_G=g['exterior_error_rms'],exterior_error_J=j['exterior_error_rms'],exterior_error_ratio=j['exterior_error_rms']/g['exterior_error_rms']))
nulls=[dec for r in P if r['arm']=='J' and r['case'].startswith('null') for dec in r['decision']]
absent=[r['decision'][2] for r in P if r['arm']=='J' and r['case']=='absent20260911']
contaminant=[r for r in P if r['arm']=='J' and r['case']=='contaminant20260911']
outside=[r for r in P if r['arm']=='J' and r['case']=='outside20260911']
real={r['arm']:r for r in P if r['case']=='real123424' and r['pass_index']==6}
effect=[]
for a in arrays:
    g,j=real['G']['fit'][arrays.index(a)],real['J']['fit'][arrays.index(a)]
    effect.append(dict(array=a,peak_change_fraction=j['peak']/g['peak']-1,centroid_change_arcsec=float(np.linalg.norm(np.array(j['centroid'])-g['centroid'])),
        width_change_fraction=(np.array(j['widths'])/g['widths']-1).tolist(),residual_rms_G=g['residual_rms'],residual_rms_J=j['residual_rms'],exterior_rms_G=g['exterior_rms'],exterior_rms_J=j['exterior_rms']))
root=Path(d['root']);bootstrap=np.load(root/'real123424/J/pass00_maps.npz')['total'][2]
max_real_change=max(float(np.nanmax(abs(np.load(root/'real123424/J'/f'pass{k:02d}_maps.npz')['total'][2]-bootstrap))) for k in range(7))
g,j=final('source_contaminant20260911','G','a2000'),final('source_contaminant20260911','J','a2000')
result=dict(decision='REVISE: positional association shows value; this frozen candidate fails the mismatch nondegradation gate',
    scientific_revision_executed=False,reserved_129081='not evaluated because development gate failed',
    aligned_and_near_recovery=recovery,aligned_near_gate=all(r['joint_target'] and r['amplitude_degradation_pp']<=2 for r in recovery),
    mismatch=mismatch,mismatch_gate=all(r['direct_error_ratio']<=1.1 and r['exterior_error_ratio']<=1.1 for r in mismatch),
    null_array_passes=len(nulls),null_promotions=sum(r['admitted_pixels']>0 for r in nulls),absent_a2000_passes=len(absent),absent_a2000_promotions=sum(r['admitted_pixels']>0 for r in absent),
    contaminant_without_anchor_promotions=sum(r['decision'][2]['admitted_pixels']>0 and not r['prior']['reference_pair_anchor'] for r in contaminant),
    outside_offset_tension_passes=sum(r['prior']['a2000_prior_tension'] for r in outside),outside_offset_final_recovery=final('outside20260911','J','a2000')['joint_target'],
    source_contaminant_a2000=dict(amplitude_bias_G=g['amplitude_bias'],amplitude_bias_J=j['amplitude_bias'],width_bias_G=g['width_bias'],width_bias_J=j['width_bias'],
        centroid_error_arcsec_J=j['centroid_error_arcsec'],direct_D_error_G=g['direct_D_error_rms'],direct_D_error_J=j['direct_D_error_rms']),
    source_contaminant_joint_gate=all(final('source_contaminant20260911','J',a)['joint_target'] for a in arrays),
    real_effect=effect,real_method_seconds={a:r['cumulative_method_map_seconds'] for a,r in real.items()},
    real_method_cost_ratio=real['J']['cumulative_method_map_seconds']/real['G']['cumulative_method_map_seconds'],
    real_a2000_max_map_change_from_bootstrap=max_real_change,real_a2000_feedback_admissions=sum(r['decision'][2]['admitted_pixels']>0 for r in P if r['case']=='real123424' and r['arm']=='J'),
    interpretation='A bound on relative physical alignment is not automatically a bound on a processed-map Gaussian centroid. This hard-constraint candidate cannot establish true geometric misalignment from its boundary hits. The mismatch deterioration is measured; exact equality versus numerical fit branch is not isolated as its cause.')
(H/'DECISION_EVIDENCE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['decision','aligned_near_gate','mismatch_gate','source_contaminant_joint_gate','null_promotions','absent_a2000_promotions','real_method_cost_ratio']},indent=2))
