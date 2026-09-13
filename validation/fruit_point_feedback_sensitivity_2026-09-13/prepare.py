"""Preserve existing evidence and bind the six-branch diagnostic before running."""
import datetime
import re
import subprocess
from pathlib import Path
from common import b,read,HERE,ROOT,REPAIR,PRIOR,SAVED,SOLUTIONS,OLD,old

def preserve():
    manifests=[Path(r['path']) for r in read(REPAIR/'PRESERVATION_START.json')['packets']]+[REPAIR/'RESULT_MANIFEST.json']
    records=[]
    for p in manifests:
        rows=read(p)['files']
        for row in rows:
            assert b.digest(p.parent/row['path'])==row['sha256'],row['path']
        records.append(dict(path=str(p),sha256=b.digest(p),payloads=len(rows)))
    external=[]
    for root in [SAVED,SOLUTIONS]:
        for row in read(root/'PRODUCT_MANIFEST.json')['files']:
            assert b.digest(root/row['path'])==row['sha256'],row['path']
        external.append(dict(root=str(root),manifest_sha256=b.digest(root/'PRODUCT_MANIFEST.json'),payloads=len(read(root/'PRODUCT_MANIFEST.json')['files'])))
    frozen=ROOT/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4'
    rows=re.findall(r'^\| `([^`]+)` \| (\d+) \| `([a-f0-9]{64})` \|$',(frozen/'PACKET_MANIFEST.md').read_text(),re.M)
    assert len(rows)==57
    for name,_,sha in rows:
        assert b.digest(frozen/name)==sha
    protected=read(REPAIR/'PRESERVATION_START.json')['protected_worktrees']
    for r in protected:
        assert subprocess.check_output(['git','-C',r['path'],'status','--short'],text=True)==r['status']
    return dict(packets=records,external_products=external,frozen_ordinary_MAP_payloads=57,
                protected_worktrees=protected,archives='presence/status only; not opened or hashed')

def main():
    assert not (HERE/'FREEZE.json').exists()
    b.write(HERE/'PRESERVATION_START.json',preserve())
    paths=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'OWNER_DIRECTION.md',HERE/'WORKTREE_RECOVERY.json',HERE/'PRESERVATION_START.json',
        old.BASE/'INPUT_PREFLIGHT.json',Path(read(old.BASE/'INPUT_PREFLIGHT.json')['input']['path']),OLD/'geometry.npz',
        PRIOR/'CASES.json',REPAIR/'RESULT_MANIFEST.json']
    paths += [Path(r['path']) for r in read(REPAIR/'FREEZE.json')['files']]
    paths += [SOLUTIONS/r['path'] for r in read(SOLUTIONS/'PRODUCT_MANIFEST.json')['files']]
    paths += [SAVED/r['path'] for r in read(SAVED/'PRODUCT_MANIFEST.json')['files']]
    paths += [REPAIR/'reevaluate.py',REPAIR/'common.py',SAVED/'PRODUCT_MANIFEST.json',SOLUTIONS/'PRODUCT_MANIFEST.json']
    b.write(HERE/'FREEZE.json',dict(identity='SCI-FRUIT-POINT-PAIRED-FEEDBACK-SENSITIVITY@r0.1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        maximum_new_cleaning_calls=6,new_feedback_optimizations_allowed=0,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(set(paths))]))
    (HERE/'FREEZE.json.sha256').write_text(b.digest(HERE/'FREEZE.json')+'  FREEZE.json\n')
    print('Bound',len(set(paths)),'files; preserved prior evidence and protected archives')

if __name__=='__main__':main()
