#!/usr/bin/env python3
"""Hash frozen scientific payloads; inspect opaque archives by status only."""
from pathlib import Path
import hashlib,json,re,subprocess
import rbf
H=Path(__file__).resolve().parent;repo=H.parent.parent
p=repo/'doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4'
rows=re.findall(r'^\| `([^`]+)` \| (\d+) \| `([a-f0-9]{64})` \|', (p/'PACKET_MANIFEST.md').read_text(),re.M)
assert len(rows)==57,len(rows)
for f,n,h in rows:
 q=p/f;assert q.stat().st_size==int(n) and rbf.base.digest(q)==h,f
statuses=[]
for name in ['4c31','346d']:
 tree=Path('/Users/gwilson/.codex/worktrees')/name/'citlali-refactor'
 status=subprocess.check_output(['git','-C',str(tree),'status','--porcelain'],text=True)
 assert all(line.startswith('?? ') for line in status.splitlines())
 if name=='4c31':
  assert set(line[3:] for line in status.splitlines())=={'SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz','SCI-FRUIT-v0.1-empirical-lane-gate-0-r0.1-owner-review.tar.gz'}
 else:assert not status
 statuses.append(dict(worktree=str(tree),status=status,archive_handling='presence/status only; no read, hash or unpack'))
rbf.base.write(H/'PRESERVATION.json',dict(frozen_ordinary_MAP_payloads_verified=len(rows),protected_worktrees=statuses,production_code_or_profiles_changed=False))
print('57 frozen payloads and protected worktree/archive status verified')
