from pathlib import Path
import json,yaml,hashlib
R=Path('/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-step-consequences-20260928');p=R/'report/N14-REF-rep2';w=json.loads((p/'windows.json').read_text());events=yaml.load((p/'event-chain.yaml').read_text(),Loader=yaml.CSafeLoader)
labels={
'n1_w0':('plausible step','Abrupt x fall amid oscillatory baseline; r has little persistent change; post-x level initially lower.'),
'n1_w1':('plausible step','Large finite x rise to a supported plateau, followed by smaller later decrease; weak r change and shared peer transition.'),
'n1_w2':('plausible step','Source-adjacent finite x rise to a long plateau; peer and small persistent r response support a physical disturbance; optical attribution not established.'),
'n1_w3':('plausible non-step','Oscillatory x bursts without a supported persistent changed level; r remains near baseline.'),
'n2_w0':('plausible step','Finite shared x decrease reaches a supported lower plateau with weaker r decrease.'),
'n2_w1':('plausible step','Large coincident x/r decrease with supported post levels and peer context.'),
'n2_w2':('plausible step','Both original x and r change substantially, despite r-only production admission; post structure varies and extent is complicated.'),
'n2_w3':('plausible finite excursion','Two finite edges bound a roughly1.2-second lower plateau; later recovery, not an isolated spike or indefinitely persistent offset.'),
'n3_w0':('plausible step','Large finite x increase to supported higher plateau; small opposite r change, shared peers.'),
'n3_w1':('plausible step','Very large two-stage x/r rise; nearby transitions complicate onset, but high post plateau is supported.'),
'n3_w2':('plausible step','Large persistent changes in both original x and r; r-only admission must not be read as absent x evidence.'),
'n3_w3':('plausible step','Two-stage x decline with persistent post offset and smaller r decline; onset/extent not a single sharp edge.'),
'n4_w0':('plausible step','Finite x rise to supported plateau, little r change, shared peer response.'),
'n4_w1':('plausible step','Large coincident x/r increase followed by persistent, slowly varying plateau.'),
'n4_w2':('plausible step','Large x and r increases with supported plateau despite r-only admission.'),
'n4_w3':('plausible step','Small abrupt x rise to a noisy supported higher level; r remains near its baseline; amplitude and interpretation less decisive.'),
'n1_w4':('plausible step','After smooth excursions, x falls finitely to a persistent lower level; little r corroboration and weak peer support do not rule out a detector-local transition.'),
'n1_w5':('plausible finite excursion','Off-center x depression with two finite edges returns before the declared center; central pre/post contrast is not a complete event measure.'),
'n1_w6':('plausible non-step','Local x/r noise and smooth baseline variation; peer structure is not proof of a local step.'),
'n1_w7':('plausible non-step','Quiet x/r control without a supported abrupt level transition.'),
'n2_w4':('plausible step','x rises in stages to a persistent plateau; small r change; strongest edge precedes nominal window center.'),
'n2_w5':('plausible step','Large finite x drop and smaller r drop reach supported lower levels; peer context agrees.'),
'n2_w6':('plausible step','Small abrupt x decrease near1.65s persists, with little r change; lower confidence than largest cases.'),
'n2_w7':('plausible non-step','Noise and smooth common variation; no supported abrupt persistent local offset.'),
'n3_w4':('plausible step','Very large abrupt x/r increases to supported plateaus. No nearby event-group seed does not mean no raw Learn candidate or existing grouped event.'),
'n3_w5':('unresolved','Large smoothly evolving common x structure; no unique sharp transition boundary, persistence versus atmospheric/common-mode change unresolved.'),
'n3_w6':('plausible step','Multiple smaller x transitions, including a sustained rise after2s; r nearly unchanged; central point precedes the strongest transition.'),
'n3_w7':('plausible non-step','Quiet local x/r with mild oscillation, no supported persistent jump.'),
'n4_w4':('plausible finite excursion','Finite x/r rise, short high plateau with a brief interruption, and return; multiple edges rather than a single persistent step.'),
'n4_w5':('plausible step','Finite x rise to persistent curved higher level with peer response; r near baseline; source context enters near excerpt end.'),
'n4_w6':('plausible non-step','Local x/r largely stable while peers show broad structure; peer variation alone is not local step evidence.'),
'n4_w7':('plausible non-step','Smooth x decline/recovery with nearly stationary r; finite step not established.')}
for q in w:
 label,reason=labels[q['key']];q['adjudication']=dict(label=label,reason=reason,scope='original x/r and original peers viewed before cleaned stage plots; targeted observational interpretation, not independent truth')
 q['selection_semantics']='unflagged selection tests proximity of event-group seeds only; raw candidates or existing groups may overlap. Frozen identities unchanged.'
 trace=[]
 for e in events:
  if e['network']!=q['network'] or e['detector']!=q['detector']:continue
  cs=[c for c in e['chain']['candidates'] if q['first']<=c['earlier_row']<q['past_last']]
  lo,hi=e['original']['trial_exclusion'];overlap=max(lo,q['first'])<min(hi,q['past_last'])
  if cs or overlap:trace.append(dict(event=e['event'],seed_native_row=e['seed_native_row'],raw_candidates_in_window=cs,trial_overlap=overlap,consider_class=e['consider']['class'],admitted=e['original']['jump_admitted']))
 q['learn_trace']=trace
 if q['kind']=='unflagged':print(q['key'],[(t['event'],len(t['raw_candidates_in_window']),t['consider_class']) for t in trace])
(p/'windows.json').write_text(json.dumps(w,indent=2)+'\n')
assert (R/'ORIGINAL_REVIEW.json').exists(), 'Historical review receipt must already exist; this script only preserves recorded adjudications'
# Do not recreate a historical viewing-order claim on rerun.
_unused_historical_record = (json.dumps(dict(original_images=[dict(path=str(x.relative_to(R)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in (R/'original-review').glob('*-original.png')],windows=32,stage_images_viewed_before_adjudication=False,selection_correction='candidate-free wording in inherited selector means no nearby event-group seed; raw Learn candidates traced independently; no window identity changed'),indent=2)+'\n')
