"""Descriptive gate audit and repair of the OG metadata lookup; no new run."""
from pathlib import Path
import re
import numpy as np
from common import b, read, HERE, OUT, PREVIOUS

def main():
    rows=read(OUT/'RECEIPTS.json')
    solves=[];fits=[];pairs=[];real=[]
    for r in rows:
        for a,d in enumerate(r['decisions']):
            if r['arm']=='C' and 'solver_success' in d:
                solves.append(dict(case=r['case'],array=b.ARRAYS[a],pass_index=r['pass_index'],
                    **{k:d[k] for k in ['available','iterations','relative_projected_gradient','solver_success','solver_message']}))
        if r['arm']=='P' and r['pass_index']==6 and r['case'].split('_')[0] in ['H','H-shift','D']:
            for a,(m,s) in enumerate(zip(r['measurement'],r['truth_score'])):
                fits.append(dict(case=r['case'],array=b.ARRAYS[a],status=m['status'],
                    measurement_available=m['measurement_available'],centroid_error_arcsec=s['centroid_error_arcsec'],
                    peak_relative_error=s['peak_relative_error'],width_relative_error=s['width_relative_error'],
                    shape_error=m['shape_error'],empirical_peak_score=m['empirical_peak_score']))
    lookup={(r['case'],r['arm'],r['pass_index']):r for r in rows}
    for seed in [20260911,20260912]:
        h,d=lookup[f'H_{seed}','P',6],lookup[f'D_{seed}','P',6]
        for a in range(3):
            ratio=h['measurement'][a]['fit']['peak']/d['measurement'][a]['fit']['peak']
            pairs.append(dict(seed=seed,array=b.ARRAYS[a],ratio=ratio,relative_error=ratio/(1/.9)-1,
                measurement_available=h['measurement'][a]['measurement_available'] and d['measurement'][a]['measurement_available'],
                meaning='Ungated fit diagnostic, not a pass where the common evaluator is unavailable'))
    prior=read(PREVIOUS/'DECISION_EVIDENCE.json')
    binding=read(PREVIOUS/'OG_BENCHMARK.json')
    log=Path(binding['log'])
    record=next(r for r in binding['files'] if r['path']==str(log))
    assert b.digest(log)==record['sha256']
    wall=float(re.search(r'^real\s+([\d.]+)',log.read_text(),re.M)[1])
    native=prior['OG_native_pointing_sequence']
    for arm in ['P','C']:
        r=lookup['real123424',arm,6]
        for a,m in enumerate(r['measurement']):
            og=next(v for v in native if v['array']==b.ARRAYS[a] and v['pass_index']==6)
            real.append(dict(arm=arm,array=b.ARRAYS[a],status=m['status'],fit=m.get('fit'),
                OG_native_centroid=og['centroid'],separation_arcsec=float(np.linalg.norm(np.array(m['fit']['centroid'])-og['centroid'])),
                meaning='Descriptive fitted location, not available operational pointing or truth error'))
    failed=[s for s in solves if not s['available']]
    result=dict(solver_records=solves,solver_counts=dict(nonempty=len(solves),accepted=len(solves)-len(failed),failed=len(failed),
        failed_with_projected_gradient_pass=sum(s['relative_projected_gradient']<=1e-4 for s in failed),
        failed_with_projected_gradient_failure=sum(s['relative_projected_gradient']>1e-4 for s in failed)),
        raw_reference_fit_diagnostics=fits,raw_reference_gain_diagnostics=pairs,
        real_fit_diagnostics=real,OG=dict(identity=binding['identity'],native_sequence=native,runtime=prior['OG_runtime'],
            full_wall_seconds=wall,log_sha256=record['sha256']),
        routine_metadata_repair=dict(original='analyze.py expected an OG key, but the already-bound source provides OG_native_pointing_sequence and OG_runtime.',
            disposition='Original DECISION_EVIDENCE.json retains OG=null. This supplement uses the actual keys and already-bound benchmark log. No method, metric, gate, input population or reduction changed.',
            new_cleaning_calls=0))
    b.write(HERE/'DIAGNOSTIC_EVIDENCE.json',result)
    print('Solver counts:',result['solver_counts'])
    print('Reference centroid error range:',min(r['centroid_error_arcsec'] for r in fits),max(r['centroid_error_arcsec'] for r in fits))
    print('Raw gain diagnostics:',pairs)
    print('Real:',[(r['arm'],r['array'],r['fit']['centroid'],r['separation_arcsec']) for r in real])

if __name__=='__main__':main()
