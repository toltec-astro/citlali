"""Preserve prior evidence and freeze this bounded comparison before execution."""
import datetime
import re
import subprocess
from pathlib import Path
from common import b,read,HERE,ROOT,PRIOR,REPAIR,SENSITIVITY,PREVIOUS,OLD,SAVED

def preserve():
    prior=read(SENSITIVITY/'PRESERVATION_START.json')
    manifests=[Path(r['path']) for r in prior['packets']]+[SENSITIVITY/'RESULT_MANIFEST.json']
    packets=[]
    for p in manifests:
        rows=read(p)['files']
        for row in rows:
            assert b.digest(p.parent/row['path'])==row['sha256'],row['path']
        packets.append(dict(path=str(p),sha256=b.digest(p),payloads=len(rows)))
    roots=[Path(r['root']) for r in prior['external_products']]+[Path('/private/tmp/sci-fruit-point-feedback-sensitivity-20260913-r0.1')]
    external=[]
    for root in roots:
        rows=read(root/'PRODUCT_MANIFEST.json')['files']
        for row in rows:
            assert b.digest(root/row['path'])==row['sha256'],row['path']
        external.append(dict(root=str(root),manifest_sha256=b.digest(root/'PRODUCT_MANIFEST.json'),payloads=len(rows)))
    frozen=ROOT/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4'
    rows=re.findall(r'^\| `([^`]+)` \| (\d+) \| `([a-f0-9]{64})` \|$',(frozen/'PACKET_MANIFEST.md').read_text(),re.M)
    assert len(rows)==57
    for name,_,sha in rows:
        assert b.digest(frozen/name)==sha
    for row in prior['protected_worktrees']:
        assert subprocess.check_output(['git','-C',row['path'],'status','--short'],text=True)==row['status']
    return dict(packets=packets,external_products=external,frozen_ordinary_MAP_payloads=57,
                protected_worktrees=prior['protected_worktrees'],archives='presence/status only; not opened or hashed')

def main():
    assert not (HERE/'FREEZE.json').exists()
    b.write(HERE/'PRESERVATION_START.json',preserve())
    paths=list(HERE.glob('*.py'))+[HERE/n for n in ['PROTOCOL.md','OWNER_DIRECTION.txt','CASES.json','PRESERVATION_START.json']]
    paths += [Path(row['path']) for row in read(PRIOR/'FREEZE.json')['files']]
    paths += [REPAIR/'stopping.py',REPAIR/'reevaluate.py',PRIOR/'evaluation.py',SENSITIVITY/'RESULT_MANIFEST.json']
    b.write(HERE/'FREEZE.json',dict(identity='SCI-FRUIT-POINT-NOMINAL-UTILITY@r0.1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        maximum_new_cleaning_calls=238,maximum_trajectories=34,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=b.digest(p)) for p in sorted(set(paths))]))
    (HERE/'FREEZE.json.sha256').write_text(b.digest(HERE/'FREEZE.json')+'  FREEZE.json\n')
    print('Frozen',len(set(paths)),'files. Preservation passed.')
if __name__=='__main__':main()
