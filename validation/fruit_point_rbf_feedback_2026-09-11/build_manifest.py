#!/usr/bin/env python3
"""Bind this report and every retained experiment/representation payload."""
from pathlib import Path
import datetime,json
import rbf,run_screen,run_screen_r02
b=rbf.base;H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def record(p,root):return dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=b.digest(p))
def main():
 run_screen.verify_freeze();run_screen_r02.verify_freeze();roots=[]
 for suffix in ['feedback-20260911-r0.1','feedback-20260911-r0.2','representation-20260911-r0.1','representation-20260911-r0.2','representation-20260911-r0.3']:
  root=Path('/private/tmp/sci-fruit-point-rbf-'+suffix);files=[record(p,root) for p in sorted(root.rglob('*')) if p.is_file()]
  roots.append(dict(root=str(root),files=files,bytes=sum(r['bytes'] for r in files)))
  if 'feedback' in suffix:
   for row in read(root/'PRODUCT_MANIFEST.json')['files']:
    p=root/row['path'];assert p.stat().st_size==row['bytes'] and b.digest(p)==row['sha256']
 # Old benchmark bindings remain exact; do not confuse this with reading protected review archives.
 og=read(rbf.PREV/'OG_BENCHMARK.json')
 for row in og['files']:
  p=Path(row['path']);assert p.stat().st_size==row['bytes'] and b.digest(p)==row['sha256'],str(p)
 native=read(H/'DECISION_EVIDENCE.json')['OG_native_POINT'];assert b.digest(native['path'])==native['sha256']
 exclusions={'RESULT_MANIFEST.json','RESULT_MANIFEST.json.sha256'}
 files=[record(p,H) for p in sorted(H.rglob('*')) if p.is_file() and p.name not in exclusions and '__pycache__' not in p.parts]
 d=dict(identity='SCI-FRUIT-POINT-RBF-RESULT@r0.2',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),feedback_decision='REJECT both frozen candidates',concentration_decision='REVISE before empirical use; calculation verified',
   freezes=dict(initial='7d3c41c71',revision='a06db7408'),files=files,external_products=roots,OG_control=og['identity'],OG_bound_files_verified=len(og['files']),
   raw_input_binding='../fruit_point_coherent_feedback_2026-09-11/INPUT_PREFLIGHT.json',preservation='PRESERVATION.json',observation_129081_exposure='historically characterized; reserved for new comparisons; no new RBF trajectory')
 b.write(H/'RESULT_MANIFEST.json',d);(H/'RESULT_MANIFEST.json.sha256').write_text(b.digest(H/'RESULT_MANIFEST.json')+'  RESULT_MANIFEST.json\n')
 for r in d['files']:assert b.digest(H/r['path'])==r['sha256']
 print('Verified',len(files),'packet files;',len(og['files']),'OG bindings;',sum(len(r['files']) for r in roots),'retained external files')
if __name__=='__main__':main()
