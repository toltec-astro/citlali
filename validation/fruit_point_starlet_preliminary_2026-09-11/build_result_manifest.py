"""Bind completed preliminary evidence; exclude this manifest and its sidecar."""
from pathlib import Path
import hashlib,json,datetime,subprocess
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[dict(path=str(p.relative_to(H)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(H.rglob('*')) if p.is_file() and p.name not in ['RESULT_MANIFEST.json','RESULT_MANIFEST.json.sha256'] and '__pycache__' not in p.parts]
manifest=dict(identity='SCI-FRUIT-POINT-STARLET-PRELIMINARY-RESULT@r0.1',state='complete_reject_candidate_no_full_trajectory_authority',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_commit=subprocess.check_output(['git','-C',str(H),'rev-parse','fb8a4612d'],text=True).strip(),
    new_cleaning_passes=1,full_trajectories=0,method_revisions=0,external_products='/private/tmp/sci-fruit-point-starlet-preliminary-20260911-r0.1',files=files,total_payload_bytes=sum(f['bytes'] for f in files))
p=H/'RESULT_MANIFEST.json';p.write_text(json.dumps(manifest,indent=2)+'\n');(H/'RESULT_MANIFEST.json.sha256').write_text(sha(p)+'  RESULT_MANIFEST.json\n')
for r in manifest['files']:assert sha(H/r['path'])==r['sha256']
print('Verified manifest:',len(files),'payloads,',manifest['total_payload_bytes'],'bytes')
