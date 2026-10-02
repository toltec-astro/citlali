"""Freeze only after exact-source gates and independent review; never launches."""
import sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run_campaign import *
rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
require(not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip(),'clean source required')
pre=ROOT/'preflight'/rev;passed=read(pre/'PASSED.json');review=read(ROOT/'review.json')
require(passed['source_revision']==rev==review['source_revision'] and review['verdict']=='pass','gates/review not exact source')
require(not (ROOT/'FREEZE.json').exists() and not (ROOT/'LAUNCHED.json').exists(),'preserve existing freeze')
require(digest(BINARY)==passed['binary']['sha256'],'binary changed after tests')
preparation=read(verified(passed['preparation']));consumers=read(verified(passed['consumers']));reference=read(pre/'REFERENCE.json')
bindings=[bound(D/'OWNER_DIRECTIVE.md'),bound(pre/'PASSED.json'),bound(pre/'PREPARATION.json'),bound(pre/'REFERENCE.json'),bound(pre/'consumers/CONSUMERS.json'),bound(ROOT/'review.json')]
bindings += [g['log'] for g in read(pre/'GATES.json')['gates']]
bindings += consumers['sources']+[reference['config']]
for row in reference['maps']:bindings += [row['map'],row['provenance'],*row['raw_and_calibration_bindings']]
for obs in (152385,152386,152387):
 p=pre/str(obs);bindings += [bound(D/f'inputs/reference_{obs}.json'),bound(D/f'inputs/oof_{obs}.yaml'),bound(D/f'inputs/oof_{obs}_inputs.json'),bound(p/'native-preparation/products/PREPARED.yaml')]
 bindings += read(p/'freeze/dependencies.json')['inputs']+read(p/'freeze/generated-inputs.json')
unique={b['path']:dict(path=b['path'],sha256=b['sha256']) for b in bindings}
for b in unique.values():verified(b)
storage=capacity(preparation['storage']['max_native_cells'])
save_json(ROOT/'FREEZE.json',dict(schema='oof-a2000-operational-m0-freeze-v1',source_revision=rev,binary=bound(BINARY),roles=list(ROLES),new_S_starts=3,new_R_starts=0,storage=storage,bindings=list(unique.values()),consumer_configuration=consumers['configuration'],frame_limit='native S FK5: no AltAz solver attachment or paired cross-frame pixel residuals',manual_resume='Owner runs foreground command once, then says done. No automatic retries.'))
print('Frozen; no observational starts. Run run_campaign.py --check-only to test admission.')
