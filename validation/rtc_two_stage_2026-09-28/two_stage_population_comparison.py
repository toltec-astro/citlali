from pathlib import Path
import json,numpy as np
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-two-stage-20260928/report')
def read(p):return json.loads(p.read_text())
b=read(R/'population.json');q=read(R/'Q/population.json');paired=[];rows=[]
key=lambda r:(r['network'],r['detector'],r['physical_run']);lookup={key(r):r for r in q};assert set(lookup)=={key(r) for r in b}
for a in b:
 c=lookup[key(a)];am=a['stages']['T5'];cm=c['stages']['T5'];paired.append(dict(network=a['network'],detector=a['detector'],physical_run=a['physical_run'],B_retained=am['retained'],Q_retained=cm['retained'],B_background=am['background'],Q_background=cm['background'],B_difference_MAD=am.get('adjacent_difference_MAD'),Q_difference_MAD=cm.get('adjacent_difference_MAD'),B_max_half_second_mean=am.get('max_half_second_mean'),Q_max_half_second_mean=cm.get('max_half_second_mean')))
for nw in range(1,5):
 rr=[r for r in paired if r['network']==nw];same=[r for r in rr if r['B_difference_MAD'] is not None and r['Q_difference_MAD'] is not None]
 rows.append(dict(network=nw,B_retained_detectors=sum(r['B_retained']>0 for r in rr),Q_retained_detectors=sum(r['Q_retained']>0 for r in rr),B_retained_cells=sum(r['B_retained'] for r in rr),Q_retained_cells=sum(r['Q_retained'] for r in rr),B_on_common_detector_median_difference_MAD=float(np.median([r['B_difference_MAD'] for r in same])),Q_on_common_detector_median_difference_MAD=float(np.median([r['Q_difference_MAD'] for r in same])),unchanged_retained_counts_except_selection=all(r['B_retained']==r['Q_retained'] or nw==2 and r['detector']==88 for r in rr)))
(R/'population-comparison.json').write_text(json.dumps(dict(networks=rows,detector_runs=paired,scope='same detector/physical-run identities, source-exclusion rule and descriptive statistics; rejected D88 unavailable, not zero; these are not step-contrast uncertainties'),indent=2)+'\n')
print(json.dumps(rows,indent=2))
