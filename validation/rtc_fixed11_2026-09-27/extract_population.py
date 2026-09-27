"""Extract existing population and pre-RTC response diagnostics; no science runs."""
from pathlib import Path
import sys,json,numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO/'tools/timestream_successor'))
from point_rtc_diagnostics import population
from point_rtc_response import response
root=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-fixed11-20260927')
s=json.loads((Path(__file__).parent/'APPLICABILITY.json').read_text());selected={n['network']:np.array([d['disposition']=='selected' for d in n['detectors']]) for n in s['networks']}
for f in ('B','C'):
 population(root,f,root/'report'/f,background_family='B',selected=selected)
 population(root,f,root/'report'/f'{f}-matched',background_family='B',match_family='C' if f=='B' else 'B',selected=selected)
response(root,'C',root/'report/C')
