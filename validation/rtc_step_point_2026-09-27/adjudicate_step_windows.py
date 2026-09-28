"""Frozen human/model visual review of original x/r and peers, before stage views.
No production event or mask is changed. Only bounded reviewed windows are labeled.
"""
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);rows=json.loads(p.read_text())
labels={
'n11_w0':('plausible step','Finite overshoot then persistent lower x plateau; weak r response and peer discontinuity.'),
'n11_w1':('plausible step','Abrupt large x decrease with supported sloping post plateau; r changes sign at same time, peers respond.'),
'n11_w2':('plausible step','Two-stage finite rise with supported post level in x and r; neighboring transition complicates extent.'),
'n11_w3':('plausible non-step','Source-adjacent ringing/pulse returns to previous level; no supported persistent offset in this window.'),
'n11_w4':('plausible step','Original discovery x rise reaches supported changed level, coherent peers, little r change; no nearby production event.'),
'n11_w5':('unresolved','Source-adjacent x rise followed by curved decay and oscillation; persistence beyond excerpt not established.'),
'n12_w0':('plausible step','Repeated finite transitions; center raises x and r to supported changed levels.'),
'n12_w1':('plausible step','Persistent r-dominated upward transition; x remains near its prior level; nearby earlier transition.'),
'n12_w2':('plausible step','r-dominated lower plateau for about two seconds followed by return; finite discontinuities, not isolated impulse.'),
'n12_w3':('plausible step','Abrupt persistent r rise coincident with source context; x contains smooth source/common variation.'),
'n12_w4':('unresolved','Original discovery x dip has a short low plateau then substantial recovery; peer behavior similar; step versus finite excursion uncertain.'),
'n12_w5':('plausible step','Original discovery x decline reaches supported lower level with peer response and little r change; no nearby production event.'),
'n11_w11':('unresolved','Broad x rise from initial low level, with only short pre-support; finite step versus smooth/common change uncertain.'),
'n12_w6':('plausible step','Candidate-free window contains small r decrease near 1 s, a supported low plateau, and abrupt return near 2.85 s; weak x corroboration.'),
'n12_w7':('plausible step','Same physical epoch as n12_w6: small r decrease then plateau and return; shared occurrence, not independent event epoch.'),
}
for q in rows:
 label,reason=labels.get(q['key'],('plausible non-step','Noise or smoothly changing baseline/oscillation; no supported abrupt persistent offset in reviewed original x/r.'))
 q['adjudication']={'label':label,'reason':reason,'scope':'original-domain observational assessment only; not independent truth or production reclassification'}
p.write_text(json.dumps(rows,indent=2)+'\n')
