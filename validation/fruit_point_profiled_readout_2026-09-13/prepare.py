"""Register exact saved cases, validate unchanged evaluator, then bind source/inputs."""
import subprocess
import numpy as np
from common import *
from evaluator import evidence,evaluate
from profiled import SETTINGS,WIDTH_STARTS,QSCALE,OBJECTIVE_ATOL,OBJECTIVE_RTOL,FIRST_ORDER_TOL,NEAR_COST_RTOL

def main():
    assert not (HERE/'FREEZE.json').exists()
    preserve_start=preserve();write(HERE/'PRESERVATION_START.json',preserve_start)
    names=[f'{n}_{s}' for n in ['H','D'] for s in SEEDS]+['H-shift_20260911','N_20260911','E_20260911']
    records={(r['case'],r['arm'],r['pass_index']):r for r in read(OUT/'RECEIPTS.json')}
    with np.load(OUT/'geometry.npz') as z:
        x,y=z['x'],z['y'];S,D,O,Q,central,shape=[z[k] for k in ['S','D','O','Q','central','shape']]
    cal=read(OUT/'SETUP.json')['calibration'];masks={};problems=[];matches=0;paths=[]
    for name in names:
        for arm in ['P','C']:
            path=OUT/name/arm/'pass06_maps.npz';paths.append(path)
            with np.load(path) as z:total=z['total']
            old=records[name,arm,6]
            for a,array in enumerate(ARRAYS):
                m=old['measurement'][a]
                e=evidence(total[a],x,y,S[a],central[a],O[a],Q[a],tuple(shape),cal[a],m)
                v=evaluate(total[a],x,y,D[a],central[a],O[a],m,e,m['original'].get('fit'),m['fixed_inner_domain_probe'].get('fit'))
                assert v['judgments']==m['judgments'],(name,arm,array)
                for key in ['empirical_peak_score','shape_error','source_support_fraction','aperture_brightness']:
                    if key in m['original']:np.testing.assert_allclose(v['original'][key],m['original'][key],rtol=1e-12,atol=1e-12)
                for key in ['centroid_difference_arcsec','peak_relative_difference']:
                    if key in m['fixed_inner_domain_probe']:np.testing.assert_allclose(v['fixed_inner_domain_probe'][key],m['fixed_inner_domain_probe'][key],rtol=1e-12,atol=1e-12)
                matches+=1;identifier=f'{name}_{arm}_{array}'
                masks[identifier]=e['positive']
                for radius in [60,52]:
                    mask=central[a]&(np.hypot(x,y)<=radius)&np.isfinite(total[a])
                    f=m['original'].get('fit') if radius==60 else m['fixed_inner_domain_probe'].get('fit')
                    problems.append(dict(case=name,arm=arm,array=array,array_index=a,pass_index=6,radius=radius,
                        id=f'{identifier}_{radius}',map_path=str(path),pixels=int(mask.sum()),original_fit=f,
                        original_measurement=m,outer_MAD=e['R'],evidence_key=identifier,
                        primary=name.split('_')[0] in ['H','D']))
    assert len(problems)==84 and sum(v['primary'] for v in problems)==48
    np.savez_compressed(HERE/'EVIDENCE.npz',**masks)
    write(HERE/'PROBLEMS.json',problems)
    write(HERE/'REGISTRATION.json',dict(identity='SCI-FRUIT-POINT-PROFILED-READOUT@r0.1',primary=48,safeguards=36,
        cases=names,arms=['P','C'],arrays=ARRAYS,pass_index=6,domains=[60,52],starts=WIDTH_STARTS,
        settings=SETTINGS,geometry_scale=QSCALE,objective_comparison_atol=OBJECTIVE_ATOL,
        objective_comparison_rtol=OBJECTIVE_RTOL,first_order_relative_tolerance=FIRST_ORDER_TOL,
        similar_cost_relative_tolerance=NEAR_COST_RTOL,linear_diagnostic_problems=48,
        fixed_geometry_uses='saved free geometry only',normal_fitter_truth_inputs=False,
        availability='unchanged data-only evaluator; minimum feasible incomplete fit is diagnostic only',
        exact_evaluator_reproductions=matches,maximum_nonlinear_starts=252,maximum_residual_evaluations=63000))
    # Bind existing comparison inputs but do not use their truth-assisted results in fitting.
    paths+=[OUT/'geometry.npz',OUT/'SETUP.json',OUT/'RECEIPTS.json',PRIOR/'CASES.json',
        FIXED/'READOUTS.json',FIXED/'FIT_OBJECTIVE_WITNESSES.json',FIXED/'RESULT_MANIFEST.json',BASE,EVAL,REPAIR,CANDIDATE,
        FIXED/'common.py',FIXED.parent/'fruit_point_peak_failure_diagnosis_2026-09-13/diagnose.py']
    paths += [f for f in HERE.iterdir() if f.is_file()]
    frozen=dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=[dict(path=str(f),sha256=digest(f),bytes=f.stat().st_size) for f in sorted(set(paths))])
    write(HERE/'FREEZE.json',frozen)
    (HERE/'FREEZE.json.sha256').write_text(digest(HERE/'FREEZE.json')+'  FREEZE.json\n')
    print(dict(problems=len(problems),evaluator_reproductions=matches,frozen_files=len(frozen['files']),prior_payloads=sum(v['payloads'] for v in preserve_start['packets'])))
if __name__=='__main__':main()
