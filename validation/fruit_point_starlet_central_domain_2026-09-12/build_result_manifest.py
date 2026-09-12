"""Bind completed result payloads without changing frozen source/input records."""
from pathlib import Path
import hashlib,json,datetime,subprocess
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[dict(path=str(p.relative_to(H)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(H.rglob('*')) if p.is_file() and p.name not in ['RESULT_MANIFEST.json','RESULT_MANIFEST.json.sha256'] and '__pycache__' not in p.parts]
m=dict(identity='SCI-FRUIT-POINT-STARLET-CENTRAL-DOMAIN-RESULT@r0.1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    freeze_commit=subprocess.check_output(['git','-C',str(H),'rev-parse','f79e93b46'],text=True).strip(),
    state='complete_edge_hypothesis_supported_tested_reconstruction_candidate_not_accepted',new_cleaning_passes=0,estimator_calls=174,full_trajectories=0,
    external_products='/private/tmp/sci-fruit-starlet-central-domain-20260912-r0.1',files=files,total_payload_bytes=sum(x['bytes'] for x in files))
p=H/'RESULT_MANIFEST.json';p.write_text(json.dumps(m,indent=2)+'\n');(H/'RESULT_MANIFEST.json.sha256').write_text(sha(p)+'  RESULT_MANIFEST.json\n')
for row in m['files']:assert sha(H/row['path'])==row['sha256']
for row in json.loads((H/'FREEZE.json').read_text())['files']:assert sha(Path(row['path']))==row['sha256']
print('Manifest verified:',len(files),'payloads;',m['total_payload_bytes'],'bytes; frozen inputs unchanged')
